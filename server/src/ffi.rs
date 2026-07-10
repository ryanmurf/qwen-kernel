use std::ffi::{CStr, CString, c_char, c_int};
use std::path::Path;

use anyhow::{Context, Result, bail};
use libloading::Library;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QkConfig {
    pub n_slots: u32,
    pub n_ctx: u32,
    pub chunk: u32,
}

#[repr(C)]
struct QkEngineOpaque {
    _private: [u8; 0],
}

type QkOpen =
    unsafe extern "C" fn(*const c_char, *const QkConfig, *mut c_char, usize) -> *mut QkEngineOpaque;
type QkClose = unsafe extern "C" fn(*mut QkEngineOpaque);
type QkGetter = unsafe extern "C" fn(*const QkEngineOpaque) -> u32;
type QkSlotStart =
    unsafe extern "C" fn(*mut QkEngineOpaque, u32, *const u32, u32, u32, u32) -> c_int;
type QkSlotCancel = unsafe extern "C" fn(*mut QkEngineOpaque, u32);
type QkStepChunk = unsafe extern "C" fn(*mut QkEngineOpaque, *mut u32, *mut u32, *mut u32) -> c_int;
type QkStageRun = unsafe extern "C" fn(
    *mut QkEngineOpaque,
    u32,        // slot
    *const u32, // toks (first stage) or NULL
    *const f32, // hidden_in (later stages) or NULL
    u32,        // n
    u32,        // base
    *mut f32,   // hidden_out (non-last stage) or NULL
    *mut u32,   // ids_out (last stage) or NULL
) -> c_int;

/// Pipeline-split ABI (engine ≥ 240c63e). Resolved as a group; `None` on
/// older engine libraries, which still serve unsplit.
struct StageSymbols {
    stage_run: QkStageRun,
    layer_first: QkGetter,
    layer_end: QkGetter,
    n_layer: QkGetter,
    n_embd: QkGetter,
}

struct Symbols {
    close: QkClose,
    n_vocab: QkGetter,
    n_ctx: QkGetter,
    n_slots: QkGetter,
    chunk: QkGetter,
    eos_token: QkGetter,
    bos_token: QkGetter,
    slot_start: QkSlotStart,
    slot_cancel: QkSlotCancel,
    step_chunk: QkStepChunk,
    stage: Option<StageSymbols>,
}

pub struct Engine {
    _lib: Library,
    syms: Symbols,
    raw: *mut QkEngineOpaque,
}

pub struct StepOut {
    pub tokens: Vec<u32>,
    pub counts: Vec<u32>,
    pub finished: u32,
    n_slots: u32,
    chunk: u32,
}

impl StepOut {
    pub fn new(n_slots: u32, chunk: u32) -> Self {
        Self {
            tokens: vec![0; n_slots.saturating_mul(chunk) as usize],
            counts: vec![0; n_slots as usize],
            finished: 0,
            n_slots,
            chunk,
        }
    }

    pub fn slot_tokens(&self, slot: u32) -> &[u32] {
        let start = slot.saturating_mul(self.chunk) as usize;
        let count = self.counts.get(slot as usize).copied().unwrap_or(0) as usize;
        let end = start.saturating_add(count).min(self.tokens.len());
        &self.tokens[start..end]
    }
}

unsafe impl Send for Engine {}

impl Engine {
    pub fn open(lib: &Path, model: &Path, cfg: QkConfig) -> Result<Self> {
        let lib = unsafe { Library::new(lib) }
            .with_context(|| format!("failed to open engine library {}", lib.display()))?;
        let open: QkOpen = unsafe { *lib.get(b"qk_open\0")? };
        let syms = Symbols {
            close: unsafe { *lib.get(b"qk_close\0")? },
            n_vocab: unsafe { *lib.get(b"qk_n_vocab\0")? },
            n_ctx: unsafe { *lib.get(b"qk_n_ctx\0")? },
            n_slots: unsafe { *lib.get(b"qk_n_slots\0")? },
            chunk: unsafe { *lib.get(b"qk_chunk\0")? },
            eos_token: unsafe { *lib.get(b"qk_eos_token\0")? },
            bos_token: unsafe { *lib.get(b"qk_bos_token\0")? },
            slot_start: unsafe { *lib.get(b"qk_slot_start\0")? },
            slot_cancel: unsafe { *lib.get(b"qk_slot_cancel\0")? },
            step_chunk: unsafe { *lib.get(b"qk_step_chunk\0")? },
            stage: unsafe {
                match (
                    lib.get(b"qk_stage_run\0"),
                    lib.get(b"qk_layer_first\0"),
                    lib.get(b"qk_layer_end\0"),
                    lib.get(b"qk_n_layer\0"),
                    lib.get(b"qk_n_embd\0"),
                ) {
                    (Ok(run), Ok(first), Ok(end), Ok(layers), Ok(embd)) => Some(StageSymbols {
                        stage_run: *run,
                        layer_first: *first,
                        layer_end: *end,
                        n_layer: *layers,
                        n_embd: *embd,
                    }),
                    _ => None,
                }
            },
        };
        let model = CString::new(model.as_os_str().to_string_lossy().as_bytes())
            .context("model path contains interior NUL")?;
        let mut err = [0i8; 1024];
        let raw = unsafe { open(model.as_ptr(), &cfg, err.as_mut_ptr(), err.len()) };
        if raw.is_null() {
            let message = unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            bail!("qk_open failed: {message}");
        }
        Ok(Self {
            _lib: lib,
            syms,
            raw,
        })
    }

    /// `snap_prefix` is the conversation-history boundary (token count before the
    /// generation scaffold) at which to cache the KV state for cross-turn reuse;
    /// 0 disables it (falls back to caching the full prefill).
    pub fn slot_start(
        &mut self,
        slot: u32,
        prompt: &[u32],
        max_gen: u32,
        snap_prefix: u32,
    ) -> Result<()> {
        if prompt.is_empty() {
            bail!("prompt must not be empty");
        }
        let rc = unsafe {
            (self.syms.slot_start)(
                self.raw,
                slot,
                prompt.as_ptr(),
                u32::try_from(prompt.len()).context("prompt too long")?,
                max_gen,
                snap_prefix,
            )
        };
        if rc < 0 {
            bail!("qk_slot_start failed with code {rc}");
        }
        Ok(())
    }

    pub fn slot_cancel(&mut self, slot: u32) {
        unsafe { (self.syms.slot_cancel)(self.raw, slot) };
    }

    pub fn step_chunk(&mut self, out: &mut StepOut) -> Result<u32> {
        let rc = unsafe {
            (self.syms.step_chunk)(
                self.raw,
                out.tokens.as_mut_ptr(),
                out.counts.as_mut_ptr(),
                &mut out.finished,
            )
        };
        if rc < 0 {
            bail!("qk_step_chunk failed with code {rc}");
        }
        let n_vocab = self.n_vocab();
        for slot in 0..out.n_slots {
            let Some(count) = out.counts.get(slot as usize).copied() else {
                bail!("engine count buffer out of bounds");
            };
            if count > out.chunk {
                bail!("engine emitted too many tokens for slot {slot}");
            }
            for token in out.slot_tokens(slot) {
                if *token >= n_vocab {
                    bail!("engine emitted token id outside vocab");
                }
            }
        }
        Ok(rc as u32)
    }

    pub fn n_vocab(&self) -> u32 {
        unsafe { (self.syms.n_vocab)(self.raw) }
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { (self.syms.n_ctx)(self.raw) }
    }

    pub fn n_slots(&self) -> u32 {
        unsafe { (self.syms.n_slots)(self.raw) }
    }

    pub fn chunk(&self) -> u32 {
        unsafe { (self.syms.chunk)(self.raw) }
    }

    pub fn eos_token(&self) -> u32 {
        unsafe { (self.syms.eos_token)(self.raw) }
    }

    pub fn bos_token(&self) -> u32 {
        unsafe { (self.syms.bos_token)(self.raw) }
    }

    /// Pipeline-split topology, or `None` when the engine library predates
    /// the stage ABI (such a library still serves unsplit requests).
    pub fn stage_info(&self) -> Option<StageInfo> {
        let s = self.syms.stage.as_ref()?;
        Some(StageInfo {
            first: unsafe { (s.layer_first)(self.raw) },
            end: unsafe { (s.layer_end)(self.raw) },
            n_layer: unsafe { (s.n_layer)(self.raw) },
            n_embd: unsafe { (s.n_embd)(self.raw) },
        })
    }

    /// Run n positions [base, base+n) of `slot` through this engine's stage
    /// (see qk_stage_run in qk.h). Exactly one of `toks`/`hidden_in` feeds it
    /// (ids on the first stage, hidden rows otherwise) and exactly one of
    /// `hidden_out`/`ids_out` receives it (ids on the last stage). base == 0
    /// resets the slot. Buffer lengths are validated against n and n_embd.
    pub fn stage_run(
        &mut self,
        slot: u32,
        toks: Option<&[u32]>,
        hidden_in: Option<&[f32]>,
        base: u32,
        hidden_out: Option<&mut [f32]>,
        ids_out: Option<&mut [u32]>,
    ) -> Result<()> {
        let info = self
            .stage_info()
            .context("engine library has no pipeline-split ABI (qk_stage_run)")?;
        let row = info.n_embd as usize;
        let n = match (toks, hidden_in) {
            (Some(t), None) => t.len(),
            (None, Some(h)) if row > 0 && h.len() % row == 0 => h.len() / row,
            _ => bail!("stage_run: need exactly one of toks / whole hidden_in rows"),
        };
        let n = u32::try_from(n).context("stage_run: batch too long")?;
        if n == 0 {
            bail!("stage_run: empty batch");
        }
        match (&hidden_out, &ids_out) {
            (Some(h), None) if h.len() == n as usize * row => {}
            (None, Some(i)) if i.len() == n as usize => {}
            _ => bail!("stage_run: need exactly one of hidden_out / ids_out, sized to n"),
        }
        let stage = self.syms.stage.as_ref().expect("checked above");
        let rc = unsafe {
            (stage.stage_run)(
                self.raw,
                slot,
                toks.map_or(std::ptr::null(), <[u32]>::as_ptr),
                hidden_in.map_or(std::ptr::null(), <[f32]>::as_ptr),
                n,
                base,
                hidden_out.map_or(std::ptr::null_mut(), <[f32]>::as_mut_ptr),
                ids_out.map_or(std::ptr::null_mut(), <[u32]>::as_mut_ptr),
            )
        };
        if rc < 0 {
            bail!("qk_stage_run failed with code {rc}");
        }
        Ok(())
    }
}

/// Which layer range this engine instance owns (QK_LAYERS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInfo {
    pub first: u32,
    pub end: u32,
    pub n_layer: u32,
    pub n_embd: u32,
}

impl StageInfo {
    pub fn is_first(&self) -> bool {
        self.first == 0
    }

    pub fn is_last(&self) -> bool {
        self.end == self.n_layer
    }

    pub fn is_split(&self) -> bool {
        !(self.is_first() && self.is_last())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.syms.close)(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
