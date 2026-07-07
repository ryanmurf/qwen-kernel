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
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.syms.close)(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
