use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

use crate::tokenizer::TokenizerConfig;
use crate::{Result, ServerError};

#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub architecture: Option<String>,
    pub name: String,
    pub tokenizer_model: String,
    pub tokenizer_pre: String,
    pub tokenizer: TokenizerConfig,
    pub chat_template: Option<String>,
}

#[derive(Clone, Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

pub fn read_metadata(path: &Path) -> Result<ModelMetadata> {
    let file = File::open(path)
        .map_err(|err| ServerError::internal(format!("failed to open GGUF: {err}")))?;
    let len = file
        .metadata()
        .map_err(|err| ServerError::internal(format!("failed to stat GGUF: {err}")))?
        .len();
    let mut reader = FileReader { file, pos: 0, len };
    parse_metadata(&mut reader)
}

pub fn read_metadata_bytes(bytes: &[u8]) -> Result<ModelMetadata> {
    let mut reader = SliceReader { bytes, pos: 0 };
    parse_metadata(&mut reader)
}

fn parse_metadata(reader: &mut impl GgufRead) -> Result<ModelMetadata> {
    let magic = reader.u32()?;
    if magic != 0x4655_4747 {
        return Err(ServerError::bad_request("invalid GGUF magic"));
    }
    let version = reader.u32()?;
    if version != 3 {
        return Err(ServerError::bad_request("unsupported GGUF version"));
    }
    let _n_tensors = reader.u64()?;
    let n_kv = reader.u64()?;
    let mut kvs = Vec::new();
    for _ in 0..n_kv {
        let key = reader.string()?;
        let ty = reader.u32()?;
        let value = reader.value(ty)?;
        kvs.push((key, value));
    }

    let architecture = take_string(&kvs, "general.architecture").ok();
    let name = take_string(&kvs, "general.name").unwrap_or_else(|_| "qwen-kernel".to_owned());
    let tokenizer_model = take_string(&kvs, "tokenizer.ggml.model")?;
    let tokenizer_pre = take_string(&kvs, "tokenizer.ggml.pre")?;
    if tokenizer_model != "gpt2" {
        return Err(ServerError::bad_request(
            "tokenizer.ggml.model must be gpt2",
        ));
    }
    if !matches!(tokenizer_pre.as_str(), "qwen35" | "qwen2") {
        return Err(ServerError::bad_request(
            "tokenizer.ggml.pre must be qwen35 or qwen2",
        ));
    }
    let tokens = take_string_array(&kvs, "tokenizer.ggml.tokens")?;
    let token_types = take_i32_array(&kvs, "tokenizer.ggml.token_type")?;
    let merges = take_string_array(&kvs, "tokenizer.ggml.merges")?;
    let eos_token_id = take_u32(&kvs, "tokenizer.ggml.eos_token_id")?;
    // Qwen GGUFs with add_bos_token=false may omit the bos id entirely
    // (Qwen3-Next-Instruct does); it is only consulted when add_bos_token is
    // set, so any placeholder is safe.
    let bos_token_id = take_u32(&kvs, "tokenizer.ggml.bos_token_id").unwrap_or(eos_token_id);
    let add_bos_token = take_bool(&kvs, "tokenizer.ggml.add_bos_token").unwrap_or(false);
    let chat_template = take_string(&kvs, "tokenizer.chat_template").ok();

    Ok(ModelMetadata {
        architecture,
        name,
        tokenizer_model,
        tokenizer_pre: tokenizer_pre.clone(),
        tokenizer: TokenizerConfig {
            pre: tokenizer_pre,
            tokens,
            token_types,
            merges,
            eos_token_id,
            bos_token_id,
            add_bos_token,
        },
        chat_template,
    })
}

fn take<'a>(kvs: &'a [(String, GgufValue)], key: &str) -> Result<&'a GgufValue> {
    kvs.iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value)
        .ok_or_else(|| ServerError::bad_request(format!("GGUF metadata missing {key}")))
}

fn take_string(kvs: &[(String, GgufValue)], key: &str) -> Result<String> {
    match take(kvs, key)? {
        GgufValue::String(value) => Ok(value.clone()),
        _ => Err(ServerError::bad_request(format!(
            "GGUF metadata {key} has wrong type"
        ))),
    }
}

fn take_string_array(kvs: &[(String, GgufValue)], key: &str) -> Result<Vec<String>> {
    match take(kvs, key)? {
        GgufValue::Array(values) => values
            .iter()
            .map(|value| match value {
                GgufValue::String(value) => Ok(value.clone()),
                _ => Err(ServerError::bad_request(format!(
                    "GGUF metadata {key} has wrong array type"
                ))),
            })
            .collect(),
        _ => Err(ServerError::bad_request(format!(
            "GGUF metadata {key} has wrong type"
        ))),
    }
}

fn take_i32_array(kvs: &[(String, GgufValue)], key: &str) -> Result<Vec<i32>> {
    match take(kvs, key)? {
        GgufValue::Array(values) => values
            .iter()
            .map(|value| match value {
                GgufValue::I32(value) => Ok(*value),
                _ => Err(ServerError::bad_request(format!(
                    "GGUF metadata {key} has wrong array type"
                ))),
            })
            .collect(),
        _ => Err(ServerError::bad_request(format!(
            "GGUF metadata {key} has wrong type"
        ))),
    }
}

fn take_u32(kvs: &[(String, GgufValue)], key: &str) -> Result<u32> {
    match take(kvs, key)? {
        GgufValue::U32(value) => Ok(*value),
        _ => Err(ServerError::bad_request(format!(
            "GGUF metadata {key} has wrong type"
        ))),
    }
}

fn take_bool(kvs: &[(String, GgufValue)], key: &str) -> Result<bool> {
    match take(kvs, key)? {
        GgufValue::Bool(value) => Ok(*value),
        _ => Err(ServerError::bad_request(format!(
            "GGUF metadata {key} has wrong type"
        ))),
    }
}

trait GgufRead {
    fn take_vec(&mut self, n: usize) -> Result<Vec<u8>>;

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take_vec(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes([self.u8()?]))
    }

    fn u16(&mut self) -> Result<u16> {
        let bytes = self.take_vec(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn i16(&mut self) -> Result<i16> {
        let bytes = self.take_vec(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take_vec(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32> {
        let bytes = self.take_vec(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let bytes = self.take_vec(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn i64(&mut self) -> Result<i64> {
        let bytes = self.take_vec(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?)
            .map_err(|_| ServerError::bad_request("GGUF string too large"))?;
        let bytes = self.take_vec(len)?;
        String::from_utf8(bytes).map_err(|_| ServerError::bad_request("GGUF string is not UTF-8"))
    }

    fn value(&mut self, ty: u32) -> Result<GgufValue> {
        match ty {
            0 => Ok(GgufValue::U8(self.u8()?)),
            1 => Ok(GgufValue::I8(self.i8()?)),
            2 => Ok(GgufValue::U16(self.u16()?)),
            3 => Ok(GgufValue::I16(self.i16()?)),
            4 => Ok(GgufValue::U32(self.u32()?)),
            5 => Ok(GgufValue::I32(self.i32()?)),
            6 => Ok(GgufValue::F32(self.f32()?)),
            7 => Ok(GgufValue::Bool(self.u8()? != 0)),
            8 => Ok(GgufValue::String(self.string()?)),
            9 => {
                let elem_type = self.u32()?;
                let count = usize::try_from(self.u64()?)
                    .map_err(|_| ServerError::bad_request("GGUF array too large"))?;
                let mut values = Vec::with_capacity(count.min(1_000_000));
                for _ in 0..count {
                    values.push(self.value(elem_type)?);
                }
                Ok(GgufValue::Array(values))
            }
            10 => Ok(GgufValue::U64(self.u64()?)),
            11 => Ok(GgufValue::I64(self.i64()?)),
            12 => Ok(GgufValue::F64(self.f64()?)),
            _ => Err(ServerError::bad_request(format!(
                "unsupported GGUF value type {ty}"
            ))),
        }
    }
}

struct SliceReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl GgufRead for SliceReader<'_> {
    fn take_vec(&mut self, n: usize) -> Result<Vec<u8>> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| ServerError::bad_request("GGUF offset overflow"))?;
        if end > self.bytes.len() {
            return Err(ServerError::bad_request("GGUF metadata extends past file"));
        }
        let out = self.bytes[self.pos..end].to_vec();
        self.pos = end;
        Ok(out)
    }
}

struct FileReader {
    file: File,
    pos: u64,
    len: u64,
}

impl GgufRead for FileReader {
    fn take_vec(&mut self, n: usize) -> Result<Vec<u8>> {
        let n_u64 =
            u64::try_from(n).map_err(|_| ServerError::bad_request("GGUF read too large"))?;
        let end = self
            .pos
            .checked_add(n_u64)
            .ok_or_else(|| ServerError::bad_request("GGUF offset overflow"))?;
        if end > self.len {
            return Err(ServerError::bad_request("GGUF metadata extends past file"));
        }
        let mut out = vec![0; n];
        self.file.read_exact(&mut out).map_err(|err| {
            ServerError::bad_request(format!("failed to read GGUF metadata: {err}"))
        })?;
        self.pos = self.file.stream_position().map_err(|err| {
            ServerError::bad_request(format!("failed to seek GGUF metadata: {err}"))
        })?;
        Ok(out)
    }
}
