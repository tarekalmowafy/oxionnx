//! Initializers left in the model file instead of being decoded into memory.
//!
//! Large embedding tables are read only a few rows at a time (`Gather` over
//! token ids). Decoding one into an FP32 `Vec` makes all of it resident for the
//! whole session; ORT and llama.cpp instead map the file and only the pages of
//! the rows a batch touches are ever read in. The loader does the same for
//! such tables (see `oxionnx_proto::model::LazyInit`): the session's weight
//! map holds them *released* (shape kept, `data` empty, as after load-time
//! packing) and the bytes stay in the mapped file, reachable by name through
//! [`crate::OperatorRegistry::mapped_weight`].
//!
//! The bytes are the initializer's `raw_data` exactly as stored (little
//! endian), so every value read from them is the one the loader would have
//! decoded: FP32 results are unchanged.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;

/// ONNX `TensorProto.DataType` codes a mapped weight may have.
pub mod data_type {
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const FLOAT16: i32 = 10;
    pub const BFLOAT16: i32 = 16;
}

/// Bytes kept alive for a [`MappedWeight`] (a file mapping, typically).
pub type SharedBytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// One initializer's stored bytes inside a shared buffer (usually a memory
/// map of the `.onnx` file or of its external-data file).
pub struct MappedWeight {
    owner: SharedBytes,
    range: Range<usize>,
    data_type: i32,
}

impl core::fmt::Debug for MappedWeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MappedWeight").field("range", &self.range).field("data_type", &self.data_type).finish()
    }
}

impl MappedWeight {
    /// `owner[range]` holds the initializer's raw data, of ONNX type
    /// `data_type`. `None` if the range is out of bounds or the type is not one
    /// of [`data_type`]'s.
    pub fn new(owner: SharedBytes, range: Range<usize>, data_type: i32) -> Option<Self> {
        let len = (*owner).as_ref().len();
        let known = matches!(data_type, data_type::FLOAT | data_type::UINT8 | data_type::FLOAT16 | data_type::BFLOAT16);
        (known && range.start <= range.end && range.end <= len).then_some(Self { owner, range, data_type })
    }

    /// The stored bytes.
    pub fn bytes(&self) -> &[u8] {
        &(*self.owner).as_ref()[self.range.clone()]
    }

    /// ONNX data type of the stored elements.
    pub fn data_type(&self) -> i32 {
        self.data_type
    }

    /// Bytes per stored element.
    pub fn elem_bytes(&self) -> usize {
        match self.data_type {
            data_type::FLOAT => 4,
            data_type::UINT8 => 1,
            _ => 2,
        }
    }

    /// Number of stored elements.
    pub fn len(&self) -> usize {
        self.range.len() / self.elem_bytes()
    }

    /// Whether no element is stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as an `f32` slice, without copying: FP32 data whose bytes
    /// are 4-byte aligned, on a little-endian target. `None` otherwise.
    pub fn as_f32(&self) -> Option<&[f32]> {
        let b = self.bytes();
        if self.data_type != data_type::FLOAT || cfg!(target_endian = "big") || b.as_ptr().align_offset(4) != 0 {
            return None;
        }
        // SAFETY: the bytes are 4-byte aligned (checked above), every bit
        // pattern is a valid f32, and the slice borrows `self`, which keeps
        // the owner alive.
        #[allow(unsafe_code)]
        Some(unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len() / 4) })
    }

    /// Decode elements `start..start + out.len()` to `f32` into `out`, as the
    /// loader decodes the whole tensor (exact for FP32; FP16 and BF16 widen
    /// exactly). Not for UINT8 data (see [`Self::decode_words`]).
    ///
    /// # Panics
    /// If the range is out of bounds.
    pub fn decode_f32(&self, start: usize, out: &mut [f32]) {
        let w = self.elem_bytes();
        let src = &self.bytes()[start * w..(start + out.len()) * w];
        match self.data_type {
            data_type::FLOAT => {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                    *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            data_type::FLOAT16 => {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                    *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            data_type::BFLOAT16 => {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                    *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            _ => panic!("MappedWeight::decode_f32 on uint8 data"),
        }
    }

    /// The whole tensor in the loader's in-memory form: decoded `f32` values,
    /// or, for UINT8 data, its bytes packed little-endian into `f32` words
    /// (the loader's byte-tensor layout when the innermost dimension is a
    /// multiple of 4, which is the only case it maps).
    pub fn materialize(&self) -> Vec<f32> {
        if self.data_type == data_type::UINT8 {
            return self.decode_words();
        }
        let mut out = alloc::vec![0f32; self.len()];
        self.decode_f32(0, &mut out);
        out
    }

    /// UINT8 bytes packed little-endian into `f32` words (a trailing partial
    /// word is zero-padded).
    pub fn decode_words(&self) -> Vec<f32> {
        self.bytes()
            .chunks(4)
            .map(|w| {
                let mut b = [0u8; 4];
                b[..w.len()].copy_from_slice(w);
                f32::from_bits(u32::from_le_bytes(b))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapped(bytes: Vec<u8>, range: Range<usize>, dt: i32) -> MappedWeight {
        MappedWeight::new(Arc::new(bytes), range, dt).expect("valid")
    }

    #[test]
    fn decodes_like_the_loader() {
        let vals = [1.5f32, -2.25, 3.0e-7, f32::MAX];
        let mut bytes = alloc::vec![0xAAu8];
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Offset 1: misaligned, so only the decoding path works.
        let m = mapped(bytes.clone(), 1..17, data_type::FLOAT);
        assert_eq!(m.len(), 4);
        assert_eq!(m.materialize(), vals);
        let mut two = [0f32; 2];
        m.decode_f32(1, &mut two);
        assert_eq!(two, [-2.25, 3.0e-7]);

        let h = [half::f16::from_f32(0.5), half::f16::from_f32(-7.0)];
        let hb: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(mapped(hb, 0..4, data_type::FLOAT16).materialize(), [0.5, -7.0]);
        let b = [half::bf16::from_f32(0.25), half::bf16::from_f32(-3.0)];
        let bb: Vec<u8> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(mapped(bb, 0..4, data_type::BFLOAT16).materialize(), [0.25, -3.0]);
    }

    #[test]
    fn uint8_packs_words_and_rejects_bad_ranges() {
        let m = mapped((1u8..=8).collect(), 0..8, data_type::UINT8);
        let words: Vec<u32> = m.materialize().iter().map(|f| f.to_bits()).collect();
        assert_eq!(words, [0x0403_0201, 0x0807_0605]);
        assert!(MappedWeight::new(Arc::new(alloc::vec![0u8; 4]), 0..8, data_type::FLOAT).is_none());
        assert!(MappedWeight::new(Arc::new(alloc::vec![0u8; 4]), 0..4, 7).is_none());
    }

    #[test]
    fn aligned_fp32_is_borrowed() {
        let v: Vec<f32> = alloc::vec![1.0, 2.0, 3.0];
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let owner: SharedBytes = Arc::new(bytes);
        let base = (*owner).as_ref().as_ptr() as usize;
        let start = base.next_multiple_of(4) - base;
        if start == 0 {
            let m = MappedWeight::new(owner, 0..12, data_type::FLOAT).unwrap();
            assert_eq!(m.as_f32(), Some(&[1.0f32, 2.0, 3.0][..]));
        }
    }
}
