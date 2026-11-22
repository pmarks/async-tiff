//! Decoders for different TIFF compression methods.

use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{Cursor, Read};

use bytes::Bytes;
use flate2::bufread::ZlibDecoder;

use crate::error::{AsyncTiffError, AsyncTiffResult, TiffError, TiffUnsupportedError};
use crate::tags::{Compression, PhotometricInterpretation};

/// A registry of decoders.
///
/// This allows end users to register their own decoders, for custom compression methods, or
/// override the default decoder implementations.
///
/// ```
/// use async_tiff::decoder::DecoderRegistry;
///
/// // Default registry includes Deflate, LZW, JPEG, ZSTD.
/// let registry = DecoderRegistry::default();
///
/// // Empty registry for manual configuration.
/// let empty = DecoderRegistry::empty();
/// ```
#[derive(Debug)]
pub struct DecoderRegistry(HashMap<Compression, Box<dyn Decoder>>);

impl DecoderRegistry {
    /// Create a new decoder registry with no decoders registered
    pub fn empty() -> Self {
        Self(HashMap::new())
    }
}

impl AsRef<HashMap<Compression, Box<dyn Decoder>>> for DecoderRegistry {
    fn as_ref(&self) -> &HashMap<Compression, Box<dyn Decoder>> {
        &self.0
    }
}

impl AsMut<HashMap<Compression, Box<dyn Decoder>>> for DecoderRegistry {
    fn as_mut(&mut self) -> &mut HashMap<Compression, Box<dyn Decoder>> {
        &mut self.0
    }
}

impl Default for DecoderRegistry {
    fn default() -> Self {
        let mut registry = HashMap::with_capacity(6);
        registry.insert(Compression::None, Box::new(UncompressedDecoder) as _);
        registry.insert(Compression::Deflate, Box::new(DeflateDecoder) as _);
        registry.insert(Compression::OldDeflate, Box::new(DeflateDecoder) as _);
        #[cfg(feature = "lerc")]
        registry.insert(Compression::LERC, Box::new(LercDecoder) as _);
        #[cfg(feature = "lzma")]
        registry.insert(Compression::LZMA, Box::new(LZMADecoder) as _);
        registry.insert(Compression::LZW, Box::new(LZWDecoder) as _);
        registry.insert(Compression::ModernJPEG, Box::new(JPEGDecoder) as _);
        #[cfg(feature = "jpeg2k")]
        registry.insert(Compression::JPEG2k, Box::new(JPEG2kDecoder) as _);
        #[cfg(feature = "webp")]
        registry.insert(Compression::WebP, Box::new(WebPDecoder) as _);
        registry.insert(Compression::ZSTD, Box::new(ZstdDecoder) as _);
        Self(registry)
    }
}

/// A trait to decode a TIFF tile.
pub trait Decoder: Debug + Send + Sync {
    /// Decode a TIFF tile into at most `max_out` bytes, the most its samples
    /// fill. A tile that decodes to more is an error, found before much
    /// more than `max_out` bytes are made: a small hostile tile must not
    /// expand to gigabytes.
    #[allow(clippy::too_many_arguments)]
    fn decode_tile(
        &self,
        buffer: Bytes,
        photometric_interpretation: PhotometricInterpretation,
        jpeg_tables: Option<&[u8]>,
        samples_per_pixel: u16,
        bits_per_sample: u16,
        lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>>;
}

/// The error for a tile that decodes to more than its `max_out` bytes.
fn too_large(max_out: usize) -> AsyncTiffError {
    AsyncTiffError::General(format!("a tile decodes to more than its {max_out} bytes"))
}

/// `reader` read to its end, or an error once it gives more than `max_out`
/// bytes.
pub fn read_capped(reader: impl Read, max_out: usize) -> AsyncTiffResult<Vec<u8>> {
    let mut out = Vec::new();
    reader
        .take((max_out as u64).saturating_add(1))
        .read_to_end(&mut out)?;
    if out.len() > max_out {
        return Err(too_large(max_out));
    }
    Ok(out)
}

/// A decoder for the Deflate compression method.
#[derive(Debug, Clone)]
pub struct DeflateDecoder;

impl Decoder for DeflateDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        read_capped(ZlibDecoder::new(Cursor::new(buffer)), max_out)
    }
}

/// A decoder for the JPEG compression method.
#[derive(Debug, Clone)]
pub struct JPEGDecoder;

impl Decoder for JPEGDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        photometric_interpretation: PhotometricInterpretation,
        jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        decode_modern_jpeg(buffer, photometric_interpretation, jpeg_tables, max_out)
    }
}

/// A decoder for the LERC compression method.
#[cfg(feature = "lerc")]
#[derive(Debug, Clone)]
pub struct LercDecoder;

/// Helper to decode and convert to bytes
#[cfg(feature = "lerc")]
fn decode_lerc<T: lerc::LercDataType + bytemuck::Pod>(
    buffer: &[u8],
    info: &lerc::BlobInfo,
) -> AsyncTiffResult<Vec<u8>> {
    let (data, _mask) = lerc::decode::<T>(
        buffer,
        info.width as usize,
        info.height as usize,
        info.depth as usize,
        info.bands as usize,
        info.masks as usize,
    )
    .map_err(|e| AsyncTiffError::General(format!("LERC decode failed: {e}")))?;

    // TODO: in the future we could avoid this copy by allowing the return type of the decoder to
    // be a typed array, not just Vec<u8>
    Ok(bytemuck::cast_slice(&data).to_vec())
}

#[cfg(feature = "lerc")]
impl Decoder for LercDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        // LercParameters[1] is the inner compression type:
        //   0 = none, 1 = deflate, 2 = zstd
        // Decompress the outer wrapper before passing to the LERC decoder.
        // A LERC blob is near its samples' size: twice it, and its header,
        // bounds the wrapper.
        let blob_cap = max_out.saturating_mul(2).saturating_add(1 << 16);
        let lerc_blob: Vec<u8> = match lerc_parameters.and_then(|p| p.get(1).copied()) {
            Some(1) => read_capped(ZlibDecoder::new(Cursor::new(buffer)), blob_cap)?,
            Some(2) => read_capped(zstd::Decoder::new(Cursor::new(buffer))?, blob_cap)?,
            _ => buffer.to_vec(),
        };

        let info = lerc::get_blob_info(&lerc_blob)
            .map_err(|e| AsyncTiffError::General(format!("LERC get_blob_info failed: {e}")))?;
        let width = match info.data_type {
            0 | 1 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            _ => 8,
        };
        let bytes = [info.width, info.height, info.depth, info.bands]
            .iter()
            .try_fold(width, |n: usize, &d| n.checked_mul(d as usize));
        if bytes.is_none_or(|n| n > max_out) {
            return Err(too_large(max_out));
        }

        // LERC data_type mapping (from LERC C API):
        // 0=i8, 1=u8, 2=i16, 3=u16, 4=i32, 5=u32, 6=f32, 7=f64
        match info.data_type {
            0 => decode_lerc::<i8>(&lerc_blob, &info),
            1 => decode_lerc::<u8>(&lerc_blob, &info),
            2 => decode_lerc::<i16>(&lerc_blob, &info),
            3 => decode_lerc::<u16>(&lerc_blob, &info),
            4 => decode_lerc::<i32>(&lerc_blob, &info),
            5 => decode_lerc::<u32>(&lerc_blob, &info),
            6 => decode_lerc::<f32>(&lerc_blob, &info),
            7 => decode_lerc::<f64>(&lerc_blob, &info),
            _ => Err(AsyncTiffError::General(format!(
                "Unsupported LERC data type: {}",
                info.data_type
            ))),
        }
    }
}

/// A decoder for the LZMA compression method.
#[derive(Debug, Clone)]
#[cfg(feature = "lzma")]
pub struct LZMADecoder;

#[cfg(feature = "lzma")]
impl Decoder for LZMADecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        use bytes::Buf;
        use lzma_rust2::XzReader;

        let buffer = xz_within(buffer, max_out)?;
        read_capped(XzReader::new(buffer.reader(), false), max_out)
    }
}

/// `stream`, an XZ stream, with each block's LZMA2 dictionary no larger than the `max_out`
/// bytes it may decode to: a decoder allocates the dictionary a block declares, up to 4 GiB,
/// before it reads a byte, and no match reaches further back than the output. The blocks are
/// found as the reader will meet them, each after the last one's LZMA2 chunks (whose headers
/// give their sizes), never from the index, which could lie. Each changed block header gets its
/// CRC again, so the reader still checks the whole stream. A stream whose chunks decode to more
/// than `max_out` is refused before any is decoded.
#[cfg(feature = "lzma")]
fn xz_within(stream: Bytes, max_out: usize) -> AsyncTiffResult<Bytes> {
    let malformed = || AsyncTiffError::General("a malformed XZ stream".into());
    let mut data = stream.to_vec();
    if data.len() < 12 || data[..6] != *b"\xFD7zXZ\0" {
        return Err(malformed());
    }
    let check = match data[7] & 0x0F {
        0 => 0,
        id => 4usize << ((id - 1) / 3),
    };
    let varint = |bytes: &mut &[u8]| -> Option<u64> {
        let mut value = 0u64;
        for i in 0..9 {
            let (&b, rest) = bytes.split_first()?;
            *bytes = rest;
            value |= u64::from(b & 0x7F) << (7 * i);
            if b & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    };
    // The smallest LZMA2 dictionary code whose size holds `max_out`.
    let dict = |code: u8| -> u64 {
        match code {
            40.. => u64::from(u32::MAX),
            _ => u64::from(2 | (code & 1)) << (code / 2 + 11),
        }
    };
    let enough = (0..40)
        .find(|&code| dict(code) >= max_out as u64)
        .unwrap_or(40);
    let be16 = |data: &[u8], at: usize| -> Option<u64> {
        Some(u64::from(u16::from_be_bytes([
            *data.get(at)?,
            *data.get(at + 1)?,
        ])))
    };
    let (mut at, mut total) = (12usize, 0u64);
    // A block header's first byte is its size; a zero starts the index.
    while *data.get(at).ok_or_else(malformed)? != 0 {
        let size = (usize::from(data[at]) + 1) * 4;
        let header = data.get(at..at + size).ok_or_else(malformed)?;
        let flags = header[1];
        let mut fields = &header[2..size - 4];
        for present in [0x40, 0x80] {
            if flags & present != 0 {
                varint(&mut fields).ok_or_else(malformed)?;
            }
        }
        // Where in the header each oversized LZMA2 dictionary code is.
        let mut oversized = vec![];
        for _ in 0..=(flags & 0x03) {
            let id = varint(&mut fields).ok_or_else(malformed)?;
            let props = usize::try_from(varint(&mut fields).ok_or_else(malformed)?)
                .map_err(|_| malformed())?;
            if id == 0x21 && props == 1 && fields.first().is_some_and(|&code| code > enough) {
                oversized.push(size - 4 - fields.len());
            }
            fields = fields.get(props..).ok_or_else(malformed)?;
        }
        if !oversized.is_empty() {
            let header = &mut data[at..at + size];
            for i in oversized {
                header[i] = enough;
            }
            let crc = crc32fast::hash(&header[..size - 4]);
            header[size - 4..].copy_from_slice(&crc.to_le_bytes());
        }
        // The block's LZMA2 chunks, to the end marker: uncompressed ones (control 1, 2) and
        // compressed ones (0x80 up, with properties from 0xC0).
        let mut chunk = at + size;
        loop {
            let control = *data.get(chunk).ok_or_else(malformed)?;
            let (out, skip) = match control {
                0 => break,
                1 | 2 => {
                    let n = be16(&data, chunk + 1).ok_or_else(malformed)? + 1;
                    (n, 3 + n)
                }
                0x80.. => {
                    let high = u64::from(control & 0x1F) << 16;
                    let out = high + be16(&data, chunk + 1).ok_or_else(malformed)? + 1;
                    let packed = be16(&data, chunk + 3).ok_or_else(malformed)? + 1;
                    (out, 5 + u64::from(control >= 0xC0) + packed)
                }
                _ => return Err(malformed()),
            };
            total = total.saturating_add(out);
            if total > max_out as u64 {
                return Err(too_large(max_out));
            }
            chunk = usize::try_from(skip)
                .ok()
                .and_then(|skip| chunk.checked_add(skip))
                .ok_or_else(malformed)?;
        }
        // The end marker, the block's padding to four bytes, its check.
        let end = chunk + 1;
        at = (end - at)
            .checked_next_multiple_of(4)
            .and_then(|block| at.checked_add(block)?.checked_add(check))
            .ok_or_else(malformed)?;
    }
    Ok(data.into())
}

/// A decoder for the LZW compression method.
#[derive(Debug, Clone)]
pub struct LZWDecoder;

impl Decoder for LZWDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        // https://github.com/image-rs/image-tiff/blob/90ae5b8e54356a35e266fb24e969aafbcb26e990/src/decoder/stream.rs#L147
        let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
        let mut out = Vec::new();
        let mut step = vec![0u8; max_out.clamp(1, 1 << 16)];
        let mut input = &buffer[..];
        loop {
            let result = decoder.decode_bytes(input, &mut step);
            input = &input[result.consumed_in..];
            out.extend_from_slice(&step[..result.consumed_out]);
            if out.len() > max_out {
                return Err(too_large(max_out));
            }
            let status = result
                .status
                .map_err(|e| AsyncTiffError::General(format!("bad LZW data: {e}")))?;
            match status {
                weezl::LzwStatus::Ok => {}
                // A TIFF writer may leave out the end code.
                weezl::LzwStatus::Done | weezl::LzwStatus::NoProgress => return Ok(out),
            }
        }
    }
}

/// A decoder for the JPEG2000 compression method.
#[cfg(feature = "jpeg2k")]
#[derive(Debug, Clone)]
pub struct JPEG2kDecoder;

#[cfg(feature = "jpeg2k")]
impl Decoder for JPEG2kDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        let header = Jpeg2kHeader::parse(&buffer)?;
        if header.decoded_bytes().is_none_or(|n| n > max_out) {
            return Err(too_large(max_out));
        }
        let decoder = jpeg2k::DecodeParameters::new();

        let image = jpeg2k::Image::from_bytes_with(&buffer, decoder)?;

        let id = image.get_pixels(None)?;
        match id.data {
            jpeg2k::ImagePixelData::L8(items)
            | jpeg2k::ImagePixelData::La8(items)
            | jpeg2k::ImagePixelData::Rgb8(items)
            | jpeg2k::ImagePixelData::Rgba8(items) => Ok(items),
            jpeg2k::ImagePixelData::L16(ref items)
            | jpeg2k::ImagePixelData::La16(ref items)
            | jpeg2k::ImagePixelData::Rgb16(ref items)
            | jpeg2k::ImagePixelData::Rgba16(ref items) => Ok(bytemuck::cast_slice(items).to_vec()),
        }
    }
}

/// What a JPEG 2000 codestream's SIZ marker declares: read before it is
/// decoded, to refuse one too large for its tile.
#[cfg(feature = "jpeg2k")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jpeg2kHeader {
    /// The image's width and height on the reference grid.
    pub size: [u32; 2],
    /// Its components, in order.
    pub components: Vec<Jpeg2kComponent>,
}

/// One component of a [`Jpeg2kHeader`].
#[cfg(feature = "jpeg2k")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Jpeg2kComponent {
    /// A sample's precision in bits.
    pub bits: u8,
    /// Whether its samples are signed.
    pub signed: bool,
    /// Its subsampling across and down: 1 for none.
    pub subsampling: [u8; 2],
}

#[cfg(feature = "jpeg2k")]
impl Jpeg2kHeader {
    /// The header of `buffer`, a codestream or a JP2 file holding one.
    pub fn parse(buffer: &[u8]) -> AsyncTiffResult<Jpeg2kHeader> {
        let bad = |why: &str| AsyncTiffError::General(format!("a JPEG 2000 header {why}"));
        let cs = codestream(buffer).ok_or_else(|| bad("isn't there"))?;
        let u16_at = |i: usize| cs.get(i..i + 2).map(|b| u16::from_be_bytes([b[0], b[1]]));
        let u32_at = |i: usize| {
            cs.get(i..i + 4)
                .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        };
        if u16_at(0) != Some(0xFF4F) || u16_at(2) != Some(0xFF51) {
            return Err(bad("doesn't start with SOC and SIZ"));
        }
        let read = || {
            let size = [
                u32_at(8)?.checked_sub(u32_at(16)?)?,
                u32_at(12)?.checked_sub(u32_at(20)?)?,
            ];
            let components = (0..usize::from(u16_at(40)?))
                .map(|c| {
                    let [s, dx, dy] = cs.get(42 + 3 * c..45 + 3 * c)?.try_into().ok()?;
                    Some(Jpeg2kComponent {
                        bits: (s & 0x7F) + 1,
                        signed: s & 0x80 != 0,
                        subsampling: [dx, dy],
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Jpeg2kHeader { size, components })
        };
        read().ok_or_else(|| bad("is cut short"))
    }

    /// The most bytes it can decode to: every component at full size, in
    /// whole bytes per sample. `None` past a `usize`.
    pub fn decoded_bytes(&self) -> Option<usize> {
        let bits = self.components.iter().map(|c| c.bits).max().unwrap_or(0);
        let width = usize::from(bits.div_ceil(8));
        [
            self.size[0] as usize,
            self.size[1] as usize,
            self.components.len(),
        ]
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d))
    }
}

/// The codestream in `buffer`: all of it, or a JP2 file's `jp2c` box.
#[cfg(feature = "jpeg2k")]
fn codestream(buffer: &[u8]) -> Option<&[u8]> {
    const JP2: [u8; 12] = [0, 0, 0, 12, b'j', b'P', b' ', b' ', 13, 10, 0x87, 10];
    if !buffer.starts_with(&JP2) {
        return Some(buffer);
    }
    let mut at = 0usize;
    while at < buffer.len() {
        let head = buffer.get(at..at + 8)?;
        let (len, kind) = (u32::from_be_bytes(head[..4].try_into().ok()?), &head[4..8]);
        let (start, end) = match len {
            0 => (at + 8, buffer.len()),
            1 => {
                let xl = u64::from_be_bytes(buffer.get(at + 8..at + 16)?.try_into().ok()?);
                (at + 16, at.checked_add(usize::try_from(xl).ok()?)?)
            }
            n => (at + 8, at.checked_add(n as usize)?),
        };
        if kind == b"jp2c" {
            return buffer.get(start..end);
        }
        if end <= at {
            return None;
        }
        at = end;
    }
    None
}

/// A decoder for the WebP compression method.
#[cfg(feature = "webp")]
#[derive(Debug, Clone)]
pub struct WebPDecoder;

#[cfg(feature = "webp")]
impl Decoder for WebPDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        samples_per_pixel: u16,
        bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        let features = webp::BitstreamFeatures::new(&buffer)
            .ok_or(AsyncTiffError::General("WebP decoding failed".to_string()))?;
        let bytes = (features.width() as usize)
            .checked_mul(features.height() as usize)
            .and_then(|n| n.checked_mul(if samples_per_pixel == 4 { 4 } else { 3 }));
        if bytes.is_none_or(|n| n > max_out) {
            return Err(too_large(max_out));
        }
        let decoded = webp::Decoder::new(&buffer)
            .decode()
            .ok_or(AsyncTiffError::General("WebP decoding failed".to_string()))?;

        let data = decoded.to_vec();

        // WebP lossy compression may discard fully-opaque alpha channels.
        // If the TIFF expects 4 samples but WebP decoded to 3, expand RGB to RGBA.
        // Only do this for 8-bit data since WebP only supports 8-bit.
        if samples_per_pixel == 4 && bits_per_sample == 8 && !decoded.is_alpha() {
            let mut rgba = Vec::with_capacity(data.len() / 3 * 4);
            for chunk in data.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255); // opaque alpha
            }
            Ok(rgba)
        } else {
            Ok(data)
        }
    }
}

/// A decoder for uncompressed data.
#[derive(Debug, Clone)]
pub struct UncompressedDecoder;

impl Decoder for UncompressedDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        // Past its samples, a tile's bytes are padding.
        Ok(buffer[..buffer.len().min(max_out)].to_vec())
    }
}

/// A decoder for the Zstd compression method.
#[derive(Debug, Clone)]
pub struct ZstdDecoder;

impl Decoder for ZstdDecoder {
    fn decode_tile(
        &self,
        buffer: Bytes,
        _photometric_interpretation: PhotometricInterpretation,
        _jpeg_tables: Option<&[u8]>,
        _samples_per_pixel: u16,
        _bits_per_sample: u16,
        _lerc_parameters: Option<&[u32]>,
        max_out: usize,
    ) -> AsyncTiffResult<Vec<u8>> {
        read_capped(zstd::Decoder::new(Cursor::new(buffer))?, max_out)
    }
}

// https://github.com/image-rs/image-tiff/blob/3bfb43e83e31b0da476832067ada68a82b378b7b/src/decoder/image.rs#L389-L450
fn decode_modern_jpeg(
    buf: Bytes,
    photometric_interpretation: PhotometricInterpretation,
    jpeg_tables: Option<&[u8]>,
    max_out: usize,
) -> AsyncTiffResult<Vec<u8>> {
    // Construct new jpeg_reader wrapping a SmartReader.
    //
    // JPEG compression in TIFF allows saving quantization and/or huffman tables in one central
    // location. These `jpeg_tables` are simply prepended to the remaining jpeg image data. Because
    // these `jpeg_tables` start with a `SOI` (HEX: `0xFFD8`) or __start of image__ marker which is
    // also at the beginning of the remaining JPEG image data and would confuse the JPEG renderer,
    // one of these has to be taken off. In this case the first two bytes of the remaining JPEG
    // data is removed because it follows `jpeg_tables`. Similary, `jpeg_tables` ends with a `EOI`
    // (HEX: `0xFFD9`) or __end of image__ marker, this has to be removed as well (last two bytes
    // of `jpeg_tables`).
    let reader = Cursor::new(buf);

    let jpeg_reader = match jpeg_tables {
        Some(jpeg_tables) => {
            let mut reader = reader;
            reader.read_exact(&mut [0; 2])?;

            let end = jpeg_tables
                .len()
                .checked_sub(2)
                .ok_or(AsyncTiffError::General(
                    "JPEGTables is too short to hold an EOI marker".to_string(),
                ))?;
            Box::new(Cursor::new(&jpeg_tables[..end]).chain(reader)) as Box<dyn Read>
        }
        None => Box::new(reader),
    };

    let mut decoder = jpeg::Decoder::new(jpeg_reader);
    decoder.set_max_decoding_buffer_size(max_out);

    match photometric_interpretation {
        PhotometricInterpretation::RGB => decoder.set_color_transform(jpeg::ColorTransform::RGB),
        PhotometricInterpretation::WhiteIsZero
        | PhotometricInterpretation::BlackIsZero
        | PhotometricInterpretation::TransparencyMask => {
            decoder.set_color_transform(jpeg::ColorTransform::None)
        }
        PhotometricInterpretation::CMYK => decoder.set_color_transform(jpeg::ColorTransform::CMYK),
        PhotometricInterpretation::YCbCr => {
            decoder.set_color_transform(jpeg::ColorTransform::YCbCr)
        }
        photometric_interpretation => {
            return Err(TiffError::UnsupportedError(
                TiffUnsupportedError::UnsupportedInterpretation(photometric_interpretation),
            )
            .into());
        }
    }

    // The decoder checks its limit only once the planes are made: read the
    // frame's size first.
    decoder.read_info()?;
    let bytes = decoder.info().and_then(|info| {
        let width = match info.pixel_format {
            jpeg::PixelFormat::L8 => 1,
            jpeg::PixelFormat::L16 => 2,
            jpeg::PixelFormat::RGB24 => 3,
            jpeg::PixelFormat::CMYK32 => 4,
        };
        (info.width as usize)
            .checked_mul(info.height as usize)
            .and_then(|n| n.checked_mul(width))
    });
    if bytes.is_none_or(|n| n > max_out) {
        return Err(too_large(max_out));
    }
    let data = decoder.decode()?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn decode(decoder: &dyn Decoder, stream: Vec<u8>, max_out: usize) -> AsyncTiffResult<Vec<u8>> {
        decoder.decode_tile(
            Bytes::from(stream),
            PhotometricInterpretation::BlackIsZero,
            None,
            1,
            8,
            None,
            max_out,
        )
    }

    /// An XZ stream's LZMA2 dictionary is shrunk to what the tile can hold (preset 9 declares
    /// 64 MiB), the stream still decodes and checks, and one its index says is larger than the
    /// tile is refused.
    #[cfg(feature = "lzma")]
    #[test]
    fn an_xz_dictionary_is_no_larger_than_the_tile() {
        use lzma_rust2::{XzOptions, XzWriter};
        let tile: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut stream = vec![];
        let mut writer = XzWriter::new(&mut stream, XzOptions::with_preset(9)).unwrap();
        writer.write_all(&tile).unwrap();
        writer.finish().unwrap();

        // The block header follows the 12-byte stream header: size, flags, then the LZMA2
        // filter's id, property size and dictionary code.
        let code = |stream: &[u8]| {
            assert_eq!(stream[12 + 2..12 + 4], [0x21, 1], "one LZMA2 filter");
            stream[12 + 4]
        };
        assert_eq!(code(&stream), 28, "64 MiB");
        let within = xz_within(Bytes::from(stream.clone()), tile.len()).unwrap();
        assert_eq!(code(&within), 0, "4 KiB");

        assert_eq!(
            decode(&LZMADecoder, stream.clone(), tile.len()).unwrap(),
            tile
        );
        assert!(decode(&LZMADecoder, stream, tile.len() - 1).is_err());
    }

    /// Every block of a stream is shrunk, found by its chunks: a stream of three blocks, each
    /// declaring a 4 GiB dictionary, decodes within the tile.
    #[cfg(feature = "lzma")]
    #[test]
    fn every_xz_block_is_shrunk() {
        use lzma_rust2::{XzOptions, XzWriter};
        let tile: Vec<u8> = (0..150_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut options = XzOptions::with_preset(1);
        options.lzma_options.dict_size = 64 << 10;
        options.set_block_size(std::num::NonZeroU64::new(64 << 10));
        let mut stream = vec![];
        let mut writer = XzWriter::new(&mut stream, options).unwrap();
        writer.write_all(&tile).unwrap();
        writer.finish().unwrap();

        // Where each block header is: after the last block's chunks, padding and CRC64.
        let headers = |stream: &[u8]| {
            let (mut at, mut found) = (12, vec![]);
            while stream[at] != 0 {
                found.push(at);
                let n = |i: usize| usize::from(u16::from_be_bytes([stream[i], stream[i + 1]]));
                let mut chunk = at + (usize::from(stream[at]) + 1) * 4;
                while stream[chunk] != 0 {
                    chunk += match stream[chunk] {
                        1 | 2 => 3 + n(chunk + 1) + 1,
                        c => 5 + usize::from(c >= 0xC0) + n(chunk + 3) + 1,
                    };
                }
                at += (chunk + 1 - at).next_multiple_of(4) + 8;
            }
            found
        };
        let blocks = headers(&stream);
        assert_eq!(blocks.len(), 3, "three blocks");
        let unbounded = xz_within(Bytes::from(stream.clone()), usize::MAX).unwrap();
        assert_eq!(unbounded, stream, "nothing to shrink for an unbounded tile");
        // Each block declares a 4 GiB dictionary (code 40), its CRC again.
        let mut hostile = stream.clone();
        for &at in &blocks {
            let size = (usize::from(hostile[at]) + 1) * 4;
            assert_eq!(hostile[at + 2..at + 4], [0x21, 1], "one LZMA2 filter");
            hostile[at + 4] = 40;
            let crc = crc32fast::hash(&hostile[at..at + size - 4]);
            hostile[at + size - 4..at + size].copy_from_slice(&crc.to_le_bytes());
        }
        let within = xz_within(Bytes::from(hostile.clone()), tile.len()).unwrap();
        for &at in &blocks {
            assert!(within[at + 4] < 40, "block at {at} shrunk");
        }
        assert_eq!(decode(&LZMADecoder, hostile, tile.len()).unwrap(), tile);
    }

    /// A damaged XZ stream, any byte of it, or cut short anywhere, is an error or a tile,
    /// never a panic.
    #[cfg(feature = "lzma")]
    #[test]
    fn a_damaged_xz_stream_never_panics() {
        use lzma_rust2::{XzOptions, XzWriter};
        let mut stream = vec![];
        let mut writer = XzWriter::new(&mut stream, XzOptions::with_preset(6)).unwrap();
        writer.write_all(&[7; 300]).unwrap();
        writer.finish().unwrap();
        for len in 0..stream.len() {
            let _ = xz_within(Bytes::copy_from_slice(&stream[..len]), 300);
        }
        for at in 0..stream.len() {
            for value in [0, 1, 2, 3, 0x7F, 0x80, 0xC0, 0xFF] {
                let mut damaged = stream.clone();
                damaged[at] = value;
                let _ = xz_within(Bytes::from(damaged), 300);
            }
        }
    }

    /// Each general-purpose codec gives a tile its bytes, and refuses one
    /// that inflates past them before it is made.
    #[test]
    fn a_tile_decodes_to_at_most_max_out() {
        let tile = vec![7u8; 4096];
        let mut deflate = flate2::write::ZlibEncoder::new(vec![], flate2::Compression::best());
        deflate.write_all(&tile).unwrap();
        let deflate = deflate.finish().unwrap();
        let zstd = zstd::encode_all(&tile[..], 3).unwrap();
        let lzw = weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
            .encode(&tile)
            .unwrap();
        let cases: [(&dyn Decoder, Vec<u8>); 3] = [
            (&DeflateDecoder, deflate),
            (&ZstdDecoder, zstd),
            (&LZWDecoder, lzw),
        ];
        for (decoder, stream) in cases {
            assert_eq!(
                decode(decoder, stream.clone(), 4096).unwrap(),
                tile,
                "{decoder:?}"
            );
            assert!(decode(decoder, stream, 4095).is_err(), "{decoder:?}");
        }
        assert_eq!(
            decode(&UncompressedDecoder, tile.clone(), 100)
                .unwrap()
                .len(),
            100
        );
    }

    /// Garbage given to LZW is an error, not a panic.
    #[test]
    fn bad_lzw_is_an_error() {
        assert!(decode(&LZWDecoder, vec![0xFF; 64], 1 << 20).is_err());
    }

    /// A JPEGTables shorter than its EOI marker is an error, not a panic.
    #[test]
    fn short_jpeg_tables_are_an_error() {
        let decoded = JPEGDecoder.decode_tile(
            Bytes::from_static(&[0xFF, 0xD8, 0xFF, 0xD9]),
            PhotometricInterpretation::BlackIsZero,
            Some(&[0xFF]),
            1,
            8,
            None,
            64,
        );
        assert!(decoded.is_err());
    }

    /// A JPEG 2000 header is read from a codestream or a JP2 file.
    #[cfg(feature = "jpeg2k")]
    #[test]
    fn a_jpeg2000_header_is_read() {
        let mut cs = vec![0xFF, 0x4F, 0xFF, 0x51, 0, 47, 0, 0];
        for v in [2048u32, 1024, 0, 0, 2048, 1024, 0, 0] {
            cs.extend(v.to_be_bytes());
        }
        cs.extend([0, 3, 7, 1, 1, 7, 2, 1, 7, 2, 1]);
        let header = Jpeg2kHeader::parse(&cs).unwrap();
        assert_eq!(header.size, [2048, 1024]);
        assert_eq!(header.components[1].subsampling, [2, 1]);
        assert_eq!(header.decoded_bytes(), Some(2048 * 1024 * 3));
        let mut jp2 = vec![0, 0, 0, 12, b'j', b'P', b' ', b' ', 13, 10, 0x87, 10];
        jp2.extend((8 + cs.len() as u32).to_be_bytes());
        jp2.extend(b"jp2c");
        jp2.extend(&cs);
        assert_eq!(Jpeg2kHeader::parse(&jp2).unwrap(), header);
        assert!(Jpeg2kHeader::parse(&cs[..30]).is_err());
    }
}
