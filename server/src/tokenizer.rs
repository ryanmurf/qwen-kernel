use std::collections::HashMap;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use fancy_regex::Regex;

use crate::{Result, ServerError};

pub const TOKEN_NORMAL: i32 = 1;
pub const TOKEN_CONTROL: i32 = 3;
pub const TOKEN_USER_DEFINED: i32 = 4;
pub const TOKEN_BYTE: i32 = 6;

const QWEN35_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub add_bos_token: bool,
}

#[derive(Clone)]
pub struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<i32>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    regex: Regex,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    specials_all: SpecialMatcher,
    specials_user: SpecialMatcher,
    eos_token_id: u32,
    bos_token_id: u32,
    add_bos_token: bool,
}

#[derive(Clone)]
struct SpecialMatcher {
    ac: Option<AhoCorasick>,
    ids: Vec<u32>,
}

impl Tokenizer {
    pub fn from_config(config: TokenizerConfig) -> Result<Self> {
        if config.tokens.len() != config.token_types.len() {
            return Err(ServerError::internal(
                "tokenizer token/type length mismatch",
            ));
        }
        let regex = Regex::new(QWEN35_PATTERN)
            .map_err(|err| ServerError::internal(format!("invalid qwen35 regex: {err}")))?;
        let (byte_to_char, char_to_byte) = build_byte_maps();
        let token_to_id = config
            .tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect::<HashMap<_, _>>();
        let mut merges = HashMap::with_capacity(config.merges.len());
        for (rank, merge) in config.merges.iter().enumerate() {
            let Some((left, right)) = merge.split_once(' ') else {
                return Err(ServerError::internal(format!("invalid BPE merge: {merge}")));
            };
            merges.insert((left.to_owned(), right.to_owned()), rank as u32);
        }
        let specials_all = build_special_matcher(&config.tokens, &config.token_types, true)?;
        let specials_user = build_special_matcher(&config.tokens, &config.token_types, false)?;

        Ok(Self {
            tokens: config.tokens,
            token_types: config.token_types,
            token_to_id,
            merges,
            regex,
            byte_to_char,
            char_to_byte,
            specials_all,
            specials_user,
            eos_token_id: config.eos_token_id,
            bos_token_id: config.bos_token_id,
            add_bos_token: config.add_bos_token,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    pub fn tokenize(&self, text: &str, parse_special: bool) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let matcher = if parse_special {
            &self.specials_all
        } else {
            &self.specials_user
        };
        if let Some(ac) = &matcher.ac {
            let mut last = 0;
            for mat in ac.find_iter(text) {
                if mat.start() > last {
                    self.tokenize_fragment(&text[last..mat.start()], &mut out)?;
                }
                let Some(id) = matcher.ids.get(mat.pattern().as_usize()).copied() else {
                    return Err(ServerError::internal("special token matcher id missing"));
                };
                out.push(id);
                last = mat.end();
            }
            if last < text.len() {
                self.tokenize_fragment(&text[last..], &mut out)?;
            }
        } else {
            self.tokenize_fragment(text, &mut out)?;
        }
        Ok(out)
    }

    pub fn token_piece(&self, id: u32, render_special: bool) -> Result<String> {
        let idx =
            usize::try_from(id).map_err(|_| ServerError::bad_request("token id out of range"))?;
        let Some(piece) = self.tokens.get(idx) else {
            return Err(ServerError::bad_request("token id out of range"));
        };
        match self.token_types.get(idx).copied().unwrap_or(TOKEN_NORMAL) {
            TOKEN_CONTROL if !render_special => Ok(String::new()),
            TOKEN_CONTROL | TOKEN_USER_DEFINED => Ok(piece.clone()),
            _ => self.decode_mapped_piece(piece),
        }
    }

    pub fn detokenize(&self, ids: &[u32], render_special: bool) -> Result<String> {
        let mut bytes = Vec::new();
        let mut out = String::new();
        for id in ids {
            let idx = usize::try_from(*id)
                .map_err(|_| ServerError::bad_request("token id out of range"))?;
            let Some(piece) = self.tokens.get(idx) else {
                return Err(ServerError::bad_request("token id out of range"));
            };
            match self.token_types.get(idx).copied().unwrap_or(TOKEN_NORMAL) {
                TOKEN_CONTROL if !render_special => {}
                TOKEN_CONTROL | TOKEN_USER_DEFINED => out.push_str(piece),
                _ => {
                    for ch in piece.chars() {
                        let Some(byte) = self.char_to_byte.get(&ch).copied() else {
                            return Err(ServerError::internal(
                                "token contains byte-map character not in table",
                            ));
                        };
                        bytes.push(byte);
                    }
                    flush_valid_prefix(&mut bytes, &mut out, false);
                }
            }
        }
        flush_valid_prefix(&mut bytes, &mut out, true);
        Ok(out)
    }

    pub fn streaming_decoder(&self) -> Utf8StreamDecoder {
        Utf8StreamDecoder::default()
    }

    fn tokenize_fragment(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let mut covered = String::new();
        let mut pieces = Vec::new();
        for mat in self.regex.find_iter(text) {
            let mat =
                mat.map_err(|err| ServerError::internal(format!("pretokenizer failed: {err}")))?;
            let piece = mat.as_str();
            covered.push_str(piece);
            pieces.push(piece.to_owned());
        }
        if covered != text {
            return Err(ServerError::internal("pretokenizer did not cover input"));
        }
        for piece in pieces {
            let mapped = self.map_bytes(piece.as_bytes());
            for symbol in self.bpe(&mapped) {
                let Some(id) = self.token_to_id.get(&symbol).copied() else {
                    return Err(ServerError::internal(format!(
                        "missing token for BPE symbol {symbol:?}"
                    )));
                };
                out.push(id);
            }
        }
        Ok(())
    }

    fn map_bytes(&self, bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|byte| self.byte_to_char[usize::from(*byte)])
            .collect()
    }

    fn bpe(&self, mapped: &str) -> Vec<String> {
        let mut symbols = mapped.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for pos in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = self
                    .merges
                    .get(&(symbols[pos].clone(), symbols[pos + 1].clone()))
                    .copied()
                    && best.is_none_or(|(best_rank, _)| rank < best_rank)
                {
                    best = Some((rank, pos));
                }
            }
            let Some((_, pos)) = best else {
                break;
            };
            let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
            symbols.splice(pos..=pos + 1, [merged]);
            if symbols.len() < 2 {
                break;
            }
        }
        symbols
    }

    fn decode_mapped_piece(&self, piece: &str) -> Result<String> {
        let mut bytes = Vec::new();
        for ch in piece.chars() {
            let Some(byte) = self.char_to_byte.get(&ch).copied() else {
                return Err(ServerError::internal(
                    "token contains byte-map character not in table",
                ));
            };
            bytes.push(byte);
        }
        String::from_utf8(bytes)
            .map_err(|err| ServerError::internal(format!("token piece is not UTF-8: {err}")))
    }
}

fn build_special_matcher(
    tokens: &[String],
    token_types: &[i32],
    include_control: bool,
) -> Result<SpecialMatcher> {
    let mut patterns = Vec::new();
    let mut ids = Vec::new();
    for (id, (token, token_type)) in tokens.iter().zip(token_types.iter()).enumerate() {
        let is_special =
            *token_type == TOKEN_USER_DEFINED || (include_control && *token_type == TOKEN_CONTROL);
        if is_special && !token.is_empty() {
            patterns.push(token.as_str());
            ids.push(id as u32);
        }
    }
    if patterns.is_empty() {
        return Ok(SpecialMatcher { ac: None, ids });
    }
    let ac = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns)
        .map_err(|err| ServerError::internal(format!("special matcher failed: {err}")))?;
    Ok(SpecialMatcher { ac: Some(ac), ids })
}

pub fn build_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = (0x21..=0x7e)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut cs = bs.iter().map(|byte| u32::from(*byte)).collect::<Vec<_>>();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    for (byte, codepoint) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(codepoint).unwrap_or('\u{fffd}');
        byte_to_char[usize::from(byte)] = ch;
        char_to_byte.insert(ch, byte);
    }
    (byte_to_char, char_to_byte)
}

#[derive(Default)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn push_mapped_piece(&mut self, tokenizer: &Tokenizer, piece: &str) -> Result<String> {
        for ch in piece.chars() {
            let Some(byte) = tokenizer.char_to_byte.get(&ch).copied() else {
                return Err(ServerError::internal(
                    "token contains byte-map character not in table",
                ));
            };
            self.pending.push(byte);
        }
        let mut out = String::new();
        flush_valid_prefix(&mut self.pending, &mut out, false);
        Ok(out)
    }

    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        flush_valid_prefix(&mut self.pending, &mut out, true);
        out
    }
}

fn flush_valid_prefix(bytes: &mut Vec<u8>, out: &mut String, finish: bool) {
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                out.push_str(valid);
                bytes.clear();
                return;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    if let Ok(valid) = std::str::from_utf8(&bytes[..valid_up_to]) {
                        out.push_str(valid);
                    }
                    bytes.drain(..valid_up_to);
                    continue;
                }
                if err.error_len().is_none() && !finish {
                    return;
                }
                out.push('\u{fffd}');
                let drain = err
                    .error_len()
                    .unwrap_or(bytes.len())
                    .min(bytes.len())
                    .max(1);
                bytes.drain(..drain);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_spots() {
        let (map, rev) = build_byte_maps();
        assert_eq!(map[0x20], 'Ġ');
        assert_eq!(map[0x0a], 'Ċ');
        assert_eq!(map[usize::from(b'A')], 'A');
        assert_eq!(rev.get(&'Ġ'), Some(&0x20));
    }

    #[test]
    fn tiny_bpe() {
        let cfg = TokenizerConfig {
            tokens: vec!["a", "b", "ab", "c", "abc"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            token_types: vec![TOKEN_NORMAL; 5],
            merges: vec!["a b".to_owned(), "ab c".to_owned()],
            eos_token_id: 0,
            bos_token_id: 0,
            add_bos_token: false,
        };
        let tokenizer = Tokenizer::from_config(cfg).expect("tiny tokenizer is valid");
        assert_eq!(tokenizer.tokenize("abc", true).expect("tokenize"), vec![4]);
    }

    #[test]
    fn regex_covers_input() {
        let re = Regex::new(QWEN35_PATTERN).expect("regex literal is valid");
        let text = "I'm 你好\n  x!";
        let mut joined = String::new();
        for mat in re.find_iter(text) {
            joined.push_str(mat.expect("regex match").as_str());
        }
        assert_eq!(joined, text);
    }

    #[test]
    fn utf8_holdback() {
        let mut bytes = vec![0xf0, 0x9f];
        let mut out = String::new();
        flush_valid_prefix(&mut bytes, &mut out, false);
        assert_eq!(out, "");
        bytes.extend([0x9a, 0x80]);
        flush_valid_prefix(&mut bytes, &mut out, false);
        assert_eq!(out, "🚀");
        assert!(bytes.is_empty());
    }
}
