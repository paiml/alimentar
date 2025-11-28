//! Alimentar Dataset Format (.ald)
//!
//! A binary format for secure, verifiable dataset distribution.
//! See `docs/specifications/dataset-format-spec.md` for full specification.
//!
//! # Format Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Header (32 bytes, fixed)                │
//! ├─────────────────────────────────────────┤
//! │ Metadata (variable, MessagePack)        │
//! ├─────────────────────────────────────────┤
//! │ Schema (variable, Arrow IPC)            │
//! ├─────────────────────────────────────────┤
//! │ Payload (variable, Arrow IPC + zstd)    │
//! ├─────────────────────────────────────────┤
//! │ Checksum (4 bytes, CRC32)               │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use alimentar::format::{save, load, SaveOptions, DatasetType};
//!
//! // Save dataset
//! save(&dataset, DatasetType::Tabular, "data.ald", SaveOptions::default())?;
//!
//! // Load dataset
//! let dataset = load("data.ald")?;
//! ```

#[cfg(feature = "format-encryption")]
pub mod encryption;
pub mod license;
pub mod piracy;
#[cfg(feature = "format-signing")]
pub mod signing;
#[cfg(feature = "format-streaming")]
pub mod streaming;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Magic bytes: "ALDF" (0x414C4446)
pub const MAGIC: [u8; 4] = [0x41, 0x4C, 0x44, 0x46];

/// Current format version major number
pub const FORMAT_VERSION_MAJOR: u8 = 1;
/// Current format version minor number
pub const FORMAT_VERSION_MINOR: u8 = 2;

/// Header size in bytes (fixed)
pub const HEADER_SIZE: usize = 32;

/// Header flags (bit positions)
pub mod flags {
    /// Payload encrypted (AES-256-GCM)
    pub const ENCRYPTED: u8 = 0b0000_0001;
    /// Has digital signature (Ed25519)
    pub const SIGNED: u8 = 0b0000_0010;
    /// Supports chunked/mmap loading (native only)
    pub const STREAMING: u8 = 0b0000_0100;
    /// Has commercial license block
    pub const LICENSED: u8 = 0b0000_1000;
    /// 64-byte aligned arrays for zero-copy SIMD (native only)
    pub const TRUENO_NATIVE: u8 = 0b0001_0000;
}

/// Dataset type identifiers (§3.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum DatasetType {
    // Structured Data (0x0001-0x000F)
    /// Generic columnar data
    Tabular = 0x0001,
    /// Temporal sequences
    TimeSeries = 0x0002,
    /// Node/edge structures
    Graph = 0x0003,
    /// Geospatial coordinates
    Spatial = 0x0004,

    // Text & NLP (0x0010-0x001F)
    /// Raw text documents
    TextCorpus = 0x0010,
    /// Labeled text
    TextClassification = 0x0011,
    /// Sentence pairs (NLI, STS)
    TextPairs = 0x0012,
    /// Token-level labels (NER)
    SequenceLabeling = 0x0013,
    /// QA datasets (SQuAD-style)
    QuestionAnswering = 0x0014,
    /// Document + summary pairs
    Summarization = 0x0015,
    /// Parallel corpora
    Translation = 0x0016,

    // Computer Vision (0x0020-0x002F)
    /// Images + class labels
    ImageClassification = 0x0020,
    /// Images + bounding boxes
    ObjectDetection = 0x0021,
    /// Images + pixel masks
    Segmentation = 0x0022,
    /// Image matching/similarity
    ImagePairs = 0x0023,
    /// Sequential frames
    Video = 0x0024,

    // Audio (0x0030-0x003F)
    /// Audio + class labels
    AudioClassification = 0x0030,
    /// Audio + transcripts (ASR)
    SpeechRecognition = 0x0031,
    /// Audio + speaker labels
    SpeakerIdentification = 0x0032,

    // Recommender Systems (0x0040-0x004F)
    /// Collaborative filtering
    UserItemRatings = 0x0040,
    /// Click/view interactions
    ImplicitFeedback = 0x0041,
    /// Session-based sequences
    SequentialRecs = 0x0042,

    // Multimodal (0x0050-0x005F)
    /// Image captioning/VQA
    ImageText = 0x0050,
    /// Speech-to-text pairs
    AudioText = 0x0051,
    /// Video descriptions
    VideoText = 0x0052,

    // Special
    /// User extensions
    Custom = 0x00FF,
}

impl DatasetType {
    /// Convert from u16 value
    #[must_use]
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::Tabular),
            0x0002 => Some(Self::TimeSeries),
            0x0003 => Some(Self::Graph),
            0x0004 => Some(Self::Spatial),
            0x0010 => Some(Self::TextCorpus),
            0x0011 => Some(Self::TextClassification),
            0x0012 => Some(Self::TextPairs),
            0x0013 => Some(Self::SequenceLabeling),
            0x0014 => Some(Self::QuestionAnswering),
            0x0015 => Some(Self::Summarization),
            0x0016 => Some(Self::Translation),
            0x0020 => Some(Self::ImageClassification),
            0x0021 => Some(Self::ObjectDetection),
            0x0022 => Some(Self::Segmentation),
            0x0023 => Some(Self::ImagePairs),
            0x0024 => Some(Self::Video),
            0x0030 => Some(Self::AudioClassification),
            0x0031 => Some(Self::SpeechRecognition),
            0x0032 => Some(Self::SpeakerIdentification),
            0x0040 => Some(Self::UserItemRatings),
            0x0041 => Some(Self::ImplicitFeedback),
            0x0042 => Some(Self::SequentialRecs),
            0x0050 => Some(Self::ImageText),
            0x0051 => Some(Self::AudioText),
            0x0052 => Some(Self::VideoText),
            0x00FF => Some(Self::Custom),
            _ => None,
        }
    }

    /// Convert to u16 value
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }
}

/// Compression algorithm identifiers (§3.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum Compression {
    /// No compression (debugging)
    None = 0x00,
    /// Zstd level 3 (standard distribution)
    #[default]
    ZstdL3 = 0x01,
    /// Zstd level 19 (archival, max compression)
    ZstdL19 = 0x02,
    /// LZ4 (high-throughput streaming)
    Lz4 = 0x03,
}

impl Compression {
    /// Convert from u8 value
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::None),
            0x01 => Some(Self::ZstdL3),
            0x02 => Some(Self::ZstdL19),
            0x03 => Some(Self::Lz4),
            _ => None,
        }
    }

    /// Convert to u8 value
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// File header (32 bytes, fixed)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 4 | magic |
/// | 4 | 2 | format_version (major.minor) |
/// | 6 | 2 | dataset_type |
/// | 8 | 4 | metadata_size |
/// | 12 | 4 | payload_size (compressed) |
/// | 16 | 4 | uncompressed_size |
/// | 20 | 1 | compression |
/// | 21 | 1 | flags |
/// | 22 | 2 | schema_size |
/// | 24 | 8 | num_rows |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Format version (major, minor)
    pub version: (u8, u8),
    /// Dataset type identifier
    pub dataset_type: DatasetType,
    /// Metadata block size in bytes
    pub metadata_size: u32,
    /// Compressed payload size in bytes
    pub payload_size: u32,
    /// Uncompressed payload size in bytes
    pub uncompressed_size: u32,
    /// Compression algorithm
    pub compression: Compression,
    /// Feature flags
    pub flags: u8,
    /// Schema block size in bytes
    pub schema_size: u16,
    /// Total row count
    pub num_rows: u64,
}

impl Header {
    /// Create a new header with default values
    #[must_use]
    pub fn new(dataset_type: DatasetType) -> Self {
        Self {
            version: (FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR),
            dataset_type,
            metadata_size: 0,
            payload_size: 0,
            uncompressed_size: 0,
            compression: Compression::default(),
            flags: 0,
            schema_size: 0,
            num_rows: 0,
        }
    }

    /// Serialize header to 32 bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];

        // Magic (0-3)
        buf[0..4].copy_from_slice(&MAGIC);

        // Version (4-5)
        buf[4] = self.version.0;
        buf[5] = self.version.1;

        // Dataset type (6-7)
        let dt = self.dataset_type.as_u16().to_le_bytes();
        buf[6..8].copy_from_slice(&dt);

        // Metadata size (8-11)
        buf[8..12].copy_from_slice(&self.metadata_size.to_le_bytes());

        // Payload size (12-15)
        buf[12..16].copy_from_slice(&self.payload_size.to_le_bytes());

        // Uncompressed size (16-19)
        buf[16..20].copy_from_slice(&self.uncompressed_size.to_le_bytes());

        // Compression (20)
        buf[20] = self.compression.as_u8();

        // Flags (21)
        buf[21] = self.flags;

        // Schema size (22-23)
        buf[22..24].copy_from_slice(&self.schema_size.to_le_bytes());

        // Num rows (24-31)
        buf[24..32].copy_from_slice(&self.num_rows.to_le_bytes());

        buf
    }

    /// Deserialize header from bytes
    ///
    /// # Errors
    ///
    /// Returns error if magic is invalid, version is unsupported, or types are
    /// unknown.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(Error::Format(format!(
                "Header too short: {} bytes, expected {}",
                buf.len(),
                HEADER_SIZE
            )));
        }

        // Validate magic
        if buf[0..4] != MAGIC {
            return Err(Error::Format(format!(
                "Invalid magic: expected {:?}, got {:?}",
                MAGIC,
                &buf[0..4]
            )));
        }

        // Version
        let version = (buf[4], buf[5]);
        if version.0 > FORMAT_VERSION_MAJOR {
            return Err(Error::Format(format!(
                "Unsupported version: {}.{}, max supported: {}.{}",
                version.0, version.1, FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR
            )));
        }

        // Dataset type
        let dt_value = u16::from_le_bytes([buf[6], buf[7]]);
        let dataset_type = DatasetType::from_u16(dt_value)
            .ok_or_else(|| Error::Format(format!("Unknown dataset type: 0x{:04X}", dt_value)))?;

        // Metadata size
        let metadata_size = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);

        // Payload size
        let payload_size = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

        // Uncompressed size
        let uncompressed_size = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);

        // Compression
        let compression = Compression::from_u8(buf[20])
            .ok_or_else(|| Error::Format(format!("Unknown compression: 0x{:02X}", buf[20])))?;

        // Flags
        let flags = buf[21];

        // Schema size
        let schema_size = u16::from_le_bytes([buf[22], buf[23]]);

        // Num rows
        let num_rows = u64::from_le_bytes([
            buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
        ]);

        Ok(Self {
            version,
            dataset_type,
            metadata_size,
            payload_size,
            uncompressed_size,
            compression,
            flags,
            schema_size,
            num_rows,
        })
    }

    /// Check if encrypted flag is set
    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.flags & flags::ENCRYPTED != 0
    }

    /// Check if signed flag is set
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        self.flags & flags::SIGNED != 0
    }

    /// Check if streaming flag is set
    #[must_use]
    pub const fn is_streaming(&self) -> bool {
        self.flags & flags::STREAMING != 0
    }

    /// Check if licensed flag is set
    #[must_use]
    pub const fn is_licensed(&self) -> bool {
        self.flags & flags::LICENSED != 0
    }

    /// Check if trueno-native flag is set
    #[must_use]
    pub const fn is_trueno_native(&self) -> bool {
        self.flags & flags::TRUENO_NATIVE != 0
    }
}

/// Dataset metadata (MessagePack-encoded)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Human-readable name
    pub name: Option<String>,
    /// Version (semver)
    pub version: Option<String>,
    /// SPDX license identifier
    pub license: Option<String>,
    /// Searchable tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// Markdown description
    pub description: Option<String>,
    /// Citation (BibTeX)
    pub citation: Option<String>,
    /// Creation timestamp (RFC 3339)
    pub created_at: Option<String>,
    /// SHA-256 hash of the payload data (hex string, 64 chars)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// Computes SHA-256 hash of data and returns it as a hex string.
///
/// # Example
///
/// ```
/// use alimentar::format::sha256_hex;
///
/// let hash = sha256_hex(b"Hello, World!");
/// assert_eq!(hash.len(), 64); // 256 bits = 32 bytes = 64 hex chars
/// ```
#[cfg(feature = "provenance")]
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();

    // Convert to hex string
    result.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// CRC32 checksum calculation (IEEE polynomial)
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for byte in data {
        let index = ((crc ^ u32::from(*byte)) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    !crc
}

/// CRC32 lookup table (IEEE polynomial 0xEDB88320)
const CRC32_TABLE: [u32; 256] = [
    0x0000_0000,
    0x7707_3096,
    0xEE0E_612C,
    0x9909_51BA,
    0x076D_C419,
    0x706A_F48F,
    0xE963_A535,
    0x9E64_95A3,
    0x0EDB_8832,
    0x79DC_B8A4,
    0xE0D5_E91E,
    0x97D2_D988,
    0x09B6_4C2B,
    0x7EB1_7CBD,
    0xE7B8_2D07,
    0x90BF_1D91,
    0x1DB7_1064,
    0x6AB0_20F2,
    0xF3B9_7148,
    0x84BE_41DE,
    0x1ADA_D47D,
    0x6DDD_E4EB,
    0xF4D4_B551,
    0x83D3_85C7,
    0x136C_9856,
    0x646B_A8C0,
    0xFD62_F97A,
    0x8A65_C9EC,
    0x1401_5C4F,
    0x6306_6CD9,
    0xFA0F_3D63,
    0x8D08_0DF5,
    0x3B6E_20C8,
    0x4C69_105E,
    0xD560_41E4,
    0xA267_7172,
    0x3C03_E4D1,
    0x4B04_D447,
    0xD20D_85FD,
    0xA50A_B56B,
    0x35B5_A8FA,
    0x42B2_986C,
    0xDBBB_C9D6,
    0xACBC_F940,
    0x32D8_6CE3,
    0x45DF_5C75,
    0xDCD6_0DCF,
    0xABD1_3D59,
    0x26D9_30AC,
    0x51DE_003A,
    0xC8D7_5180,
    0xBFD0_6116,
    0x21B4_F4B5,
    0x56B3_C423,
    0xCFBA_9599,
    0xB8BD_A50F,
    0x2802_B89E,
    0x5F05_8808,
    0xC60C_D9B2,
    0xB10B_E924,
    0x2F6F_7C87,
    0x5868_4C11,
    0xC161_1DAB,
    0xB666_2D3D,
    0x76DC_4190,
    0x01DB_7106,
    0x98D2_20BC,
    0xEFD5_102A,
    0x71B1_8589,
    0x06B6_B51F,
    0x9FBF_E4A5,
    0xE8B8_D433,
    0x7807_C9A2,
    0x0F00_F934,
    0x9609_A88E,
    0xE10E_9818,
    0x7F6A_0DBB,
    0x086D_3D2D,
    0x9164_6C97,
    0xE663_5C01,
    0x6B6B_51F4,
    0x1C6C_6162,
    0x8565_30D8,
    0xF262_004E,
    0x6C06_95ED,
    0x1B01_A57B,
    0x8208_F4C1,
    0xF50F_C457,
    0x65B0_D9C6,
    0x12B7_E950,
    0x8BBE_B8EA,
    0xFCB9_887C,
    0x62DD_1DDF,
    0x15DA_2D49,
    0x8CD3_7CF3,
    0xFBD4_4C65,
    0x4DB2_6158,
    0x3AB5_51CE,
    0xA3BC_0074,
    0xD4BB_30E2,
    0x4ADF_A541,
    0x3DD8_95D7,
    0xA4D1_C46D,
    0xD3D6_F4FB,
    0x4369_E96A,
    0x346E_D9FC,
    0xAD67_8846,
    0xDA60_B8D0,
    0x4404_2D73,
    0x3303_1DE5,
    0xAA0A_4C5F,
    0xDD0D_7CC9,
    0x5005_713C,
    0x2702_41AA,
    0xBE0B_1010,
    0xC90C_2086,
    0x5768_B525,
    0x206F_85B3,
    0xB966_D409,
    0xCE61_E49F,
    0x5EDE_F90E,
    0x29D9_C998,
    0xB0D0_9822,
    0xC7D7_A8B4,
    0x59B3_3D17,
    0x2EB4_0D81,
    0xB7BD_5C3B,
    0xC0BA_6CAD,
    0xEDB8_8320,
    0x9ABF_B3B6,
    0x03B6_E20C,
    0x74B1_D29A,
    0xEAD5_4739,
    0x9DD2_77AF,
    0x04DB_2615,
    0x73DC_1683,
    0xE363_0B12,
    0x9464_3B84,
    0x0D6D_6A3E,
    0x7A6A_5AA8,
    0xE40E_CF0B,
    0x9309_FF9D,
    0x0A00_AE27,
    0x7D07_9EB1,
    0xF00F_9344,
    0x8708_A3D2,
    0x1E01_F268,
    0x6906_C2FE,
    0xF762_575D,
    0x8065_67CB,
    0x196C_3671,
    0x6E6B_06E7,
    0xFED4_1B76,
    0x89D3_2BE0,
    0x10DA_7A5A,
    0x67DD_4ACC,
    0xF9B9_DF6F,
    0x8EBE_EFF9,
    0x17B7_BE43,
    0x60B0_8ED5,
    0xD6D6_A3E8,
    0xA1D1_937E,
    0x38D8_C2C4,
    0x4FDF_F252,
    0xD1BB_67F1,
    0xA6BC_5767,
    0x3FB5_06DD,
    0x48B2_364B,
    0xD80D_2BDA,
    0xAF0A_1B4C,
    0x3603_4AF6,
    0x4104_7A60,
    0xDF60_EFC3,
    0xA867_DF55,
    0x316E_8EEF,
    0x4669_BE79,
    0xCB61_B38C,
    0xBC66_831A,
    0x256F_D2A0,
    0x5268_E236,
    0xCC0C_7795,
    0xBB0B_4703,
    0x2202_16B9,
    0x5505_262F,
    0xC5BA_3BBE,
    0xB2BD_0B28,
    0x2BB4_5A92,
    0x5CB3_6A04,
    0xC2D7_FFA7,
    0xB5D0_CF31,
    0x2CD9_9E8B,
    0x5BDE_AE1D,
    0x9B64_C2B0,
    0xEC63_F226,
    0x756A_A39C,
    0x026D_930A,
    0x9C09_06A9,
    0xEB0E_363F,
    0x7207_6785,
    0x0500_5713,
    0x95BF_4A82,
    0xE2B8_7A14,
    0x7BB1_2BAE,
    0x0CB6_1B38,
    0x92D2_8E9B,
    0xE5D5_BE0D,
    0x7CDC_EFB7,
    0x0BDB_DF21,
    0x86D3_D2D4,
    0xF1D4_E242,
    0x68DD_B3F8,
    0x1FDA_836E,
    0x81BE_16CD,
    0xF6B9_265B,
    0x6FB0_77E1,
    0x18B7_4777,
    0x8808_5AE6,
    0xFF0F_6A70,
    0x6606_3BCA,
    0x1101_0B5C,
    0x8F65_9EFF,
    0xF862_AE69,
    0x6169_FFD3,
    0x166E_CF45,
    0xA00A_E278,
    0xD70D_D2EE,
    0x4E04_8354,
    0x3903_B3C2,
    0xA767_2661,
    0xD060_16F7,
    0x4969_474D,
    0x3E6E_77DB,
    0xAED1_6A4A,
    0xD9D6_5ADC,
    0x40DF_0B66,
    0x37D8_3BF0,
    0xA9BC_AE53,
    0xDEBB_9EC5,
    0x47B2_CF7F,
    0x30B5_FFE9,
    0xBDBD_F21C,
    0xCABA_C28A,
    0x53B3_9330,
    0x24B4_A3A6,
    0xBAD0_3605,
    0xCDD7_0693,
    0x54DE_5729,
    0x23D9_67BF,
    0xB366_7A2E,
    0xC461_4AB8,
    0x5D68_1B02,
    0x2A6F_2B94,
    0xB40B_BE37,
    0xC30C_8EA1,
    0x5A05_DF1B,
    0x2D02_EF8D,
];

/// Options for saving datasets
#[derive(Debug, Clone)]
pub struct SaveOptions {
    /// Compression algorithm to use
    pub compression: Compression,
    /// Optional metadata to include
    pub metadata: Option<Metadata>,
    /// Encryption parameters (requires `format-encryption` feature)
    #[cfg(feature = "format-encryption")]
    pub encryption: Option<encryption::EncryptionParams>,
    /// Signing key pair (requires `format-signing` feature)
    #[cfg(feature = "format-signing")]
    pub signing_key: Option<signing::SigningKeyPair>,
    /// License block for commercial distribution
    pub license: Option<license::LicenseBlock>,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            compression: Compression::ZstdL3,
            metadata: None,
            #[cfg(feature = "format-encryption")]
            encryption: None,
            #[cfg(feature = "format-signing")]
            signing_key: None,
            license: None,
        }
    }
}

impl SaveOptions {
    /// Set compression algorithm
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set password-based encryption (requires `format-encryption` feature)
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.encryption = Some(encryption::EncryptionParams::password(password));
        self
    }

    /// Set recipient-based encryption (requires `format-encryption` feature)
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_recipient(mut self, public_key: [u8; 32]) -> Self {
        self.encryption = Some(encryption::EncryptionParams::recipient(public_key));
        self
    }

    /// Set signing key (requires `format-signing` feature)
    #[cfg(feature = "format-signing")]
    #[must_use]
    pub fn with_signing_key(mut self, key: signing::SigningKeyPair) -> Self {
        self.signing_key = Some(key);
        self
    }

    /// Set license block
    #[must_use]
    pub fn with_license(mut self, license: license::LicenseBlock) -> Self {
        self.license = Some(license);
        self
    }
}

/// Options for loading datasets
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Decryption parameters (required if dataset is encrypted)
    #[cfg(feature = "format-encryption")]
    pub decryption: Option<encryption::DecryptionParams>,
    /// Trusted public keys for signature verification (if empty, skip
    /// verification)
    #[cfg(feature = "format-signing")]
    pub trusted_keys: Vec<[u8; 32]>,
    /// Whether to verify license expiration
    pub verify_license: bool,
}

impl LoadOptions {
    /// Set password for decryption
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.decryption = Some(encryption::DecryptionParams::password(password));
        self
    }

    /// Set private key for decryption
    #[cfg(feature = "format-encryption")]
    #[must_use]
    pub fn with_private_key(mut self, key: [u8; 32]) -> Self {
        self.decryption = Some(encryption::DecryptionParams::private_key(key));
        self
    }

    /// Add trusted public key for signature verification
    #[cfg(feature = "format-signing")]
    #[must_use]
    pub fn with_trusted_key(mut self, key: [u8; 32]) -> Self {
        self.trusted_keys.push(key);
        self
    }

    /// Enable license verification
    #[must_use]
    pub fn verify_license(mut self) -> Self {
        self.verify_license = true;
        self
    }
}

/// Save an Arrow dataset to the .ald format
///
/// # Errors
///
/// Returns error if serialization or I/O fails.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub fn save<W: std::io::Write>(
    writer: &mut W,
    batches: &[arrow::array::RecordBatch],
    dataset_type: DatasetType,
    options: &SaveOptions,
) -> Result<()> {
    use arrow::ipc::writer::StreamWriter;

    if batches.is_empty() {
        return Err(Error::EmptyDataset);
    }

    let schema = batches[0].schema();

    // Serialize schema via Arrow IPC
    let mut schema_buf = Vec::new();
    {
        let mut schema_writer =
            StreamWriter::try_new(&mut schema_buf, &schema).map_err(Error::Arrow)?;
        schema_writer.finish().map_err(Error::Arrow)?;
    }

    // Serialize payload via Arrow IPC
    let mut payload_buf = Vec::new();
    {
        let mut payload_writer =
            StreamWriter::try_new(&mut payload_buf, &schema).map_err(Error::Arrow)?;
        for batch in batches {
            payload_writer.write(batch).map_err(Error::Arrow)?;
        }
        payload_writer.finish().map_err(Error::Arrow)?;
    }

    let uncompressed_size = payload_buf.len() as u32;

    // Compress payload if needed
    let compressed_payload = match options.compression {
        Compression::None => payload_buf,
        Compression::ZstdL3 => {
            zstd::encode_all(payload_buf.as_slice(), 3).map_err(Error::io_no_path)?
        }
        Compression::ZstdL19 => {
            zstd::encode_all(payload_buf.as_slice(), 19).map_err(Error::io_no_path)?
        }
        Compression::Lz4 => {
            let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
            std::io::Write::write_all(&mut encoder, &payload_buf).map_err(Error::io_no_path)?;
            encoder
                .finish()
                .map_err(|e| Error::Format(format!("LZ4 compression error: {e}")))?
        }
    };

    // Build flags
    let mut header_flags: u8 = 0;

    // Encryption block (mode + salt/ephemeral_key + nonce)
    #[cfg(feature = "format-encryption")]
    let encryption_block: Option<Vec<u8>> = if let Some(ref enc_params) = options.encryption {
        header_flags |= flags::ENCRYPTED;
        Some(build_encryption_block(&compressed_payload, enc_params)?)
    } else {
        None
    };
    #[cfg(not(feature = "format-encryption"))]
    #[allow(clippy::no_effect_underscore_binding)]
    let _encryption_block: Option<Vec<u8>> = None;

    // Get the final payload (encrypted or not)
    #[cfg(feature = "format-encryption")]
    let final_payload: Vec<u8> = if let Some(ref block) = encryption_block {
        // Block contains: mode(1) + salt/ephemeral(16/32) + nonce(12) + ciphertext
        // We store the ciphertext portion as the payload
        let header_size = if block[0] == encryption::mode::PASSWORD {
            1 + 16 + 12 // mode + salt + nonce
        } else {
            1 + 32 + 12 // mode + ephemeral_pub + nonce
        };
        block[header_size..].to_vec()
    } else {
        compressed_payload
    };
    #[cfg(not(feature = "format-encryption"))]
    let final_payload: Vec<u8> = compressed_payload;

    // Encryption header (salt/nonce portion, written before payload)
    #[cfg(feature = "format-encryption")]
    let encryption_header: Vec<u8> = if let Some(ref block) = encryption_block {
        let header_size = if block[0] == encryption::mode::PASSWORD {
            1 + 16 + 12 // mode + salt + nonce
        } else {
            1 + 32 + 12 // mode + ephemeral_pub + nonce
        };
        block[..header_size].to_vec()
    } else {
        Vec::new()
    };
    #[cfg(not(feature = "format-encryption"))]
    let encryption_header: Vec<u8> = Vec::new();

    // Signing setup
    #[cfg(feature = "format-signing")]
    if options.signing_key.is_some() {
        header_flags |= flags::SIGNED;
    }

    // License setup
    if options.license.is_some() {
        header_flags |= flags::LICENSED;
    }

    // Serialize metadata
    let metadata_buf = if let Some(ref meta) = options.metadata {
        rmp_serde::to_vec(meta).map_err(|e| Error::Format(e.to_string()))?
    } else {
        rmp_serde::to_vec(&Metadata::default()).map_err(|e| Error::Format(e.to_string()))?
    };

    // Count total rows
    let num_rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();

    // Build header
    let header = Header {
        version: (FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR),
        dataset_type,
        metadata_size: metadata_buf.len() as u32,
        payload_size: final_payload.len() as u32,
        uncompressed_size,
        compression: options.compression,
        flags: header_flags,
        schema_size: schema_buf.len() as u16,
        num_rows,
    };

    // Build all data for checksum and signature
    let mut all_data = Vec::new();
    let header_bytes = header.to_bytes();
    all_data.extend_from_slice(&header_bytes);
    all_data.extend_from_slice(&metadata_buf);
    all_data.extend_from_slice(&schema_buf);
    all_data.extend_from_slice(&encryption_header);
    all_data.extend_from_slice(&final_payload);

    // Add signature block if signing
    #[cfg(feature = "format-signing")]
    let signature_block: Option<[u8; signing::SignatureBlock::SIZE]> =
        if let Some(ref key) = options.signing_key {
            let sig_block = signing::SignatureBlock::sign(&all_data, key);
            let sig_bytes = sig_block.to_bytes();
            all_data.extend_from_slice(&sig_bytes);
            Some(sig_bytes)
        } else {
            None
        };
    #[cfg(not(feature = "format-signing"))]
    let signature_block: Option<[u8; 96]> = None;

    // Add license block if present
    let license_bytes: Option<Vec<u8>> = if let Some(ref lic) = options.license {
        let lic_bytes = lic.to_bytes();
        all_data.extend_from_slice(&lic_bytes);
        Some(lic_bytes)
    } else {
        None
    };

    // Calculate checksum over all preceding data
    let checksum = crc32(&all_data);

    // Write everything
    writer.write_all(&header_bytes).map_err(Error::io_no_path)?;
    writer.write_all(&metadata_buf).map_err(Error::io_no_path)?;
    writer.write_all(&schema_buf).map_err(Error::io_no_path)?;
    writer
        .write_all(&encryption_header)
        .map_err(Error::io_no_path)?;
    writer
        .write_all(&final_payload)
        .map_err(Error::io_no_path)?;

    if let Some(ref sig) = signature_block {
        writer.write_all(sig).map_err(Error::io_no_path)?;
    }

    if let Some(ref lic) = license_bytes {
        writer.write_all(lic).map_err(Error::io_no_path)?;
    }

    writer
        .write_all(&checksum.to_le_bytes())
        .map_err(Error::io_no_path)?;

    Ok(())
}

/// Build encryption block: mode + key_material + nonce + ciphertext
#[cfg(feature = "format-encryption")]
fn build_encryption_block(
    plaintext: &[u8],
    params: &encryption::EncryptionParams,
) -> Result<Vec<u8>> {
    match &params.mode {
        encryption::EncryptionMode::Password(password) => {
            let (mode, salt, nonce, ciphertext) =
                encryption::encrypt_password(plaintext, password)?;
            let mut block = Vec::with_capacity(1 + 16 + 12 + ciphertext.len());
            block.push(mode);
            block.extend_from_slice(&salt);
            block.extend_from_slice(&nonce);
            block.extend_from_slice(&ciphertext);
            Ok(block)
        }
        encryption::EncryptionMode::Recipient {
            recipient_public_key,
        } => {
            let (mode, ephemeral_pub, nonce, ciphertext) =
                encryption::encrypt_recipient(plaintext, recipient_public_key)?;
            let mut block = Vec::with_capacity(1 + 32 + 12 + ciphertext.len());
            block.push(mode);
            block.extend_from_slice(&ephemeral_pub);
            block.extend_from_slice(&nonce);
            block.extend_from_slice(&ciphertext);
            Ok(block)
        }
    }
}

/// Loaded dataset from .ald format
#[derive(Debug)]
pub struct LoadedDataset {
    /// Parsed header
    pub header: Header,
    /// Dataset metadata
    pub metadata: Metadata,
    /// Arrow record batches
    pub batches: Vec<arrow::array::RecordBatch>,
    /// License block (if present)
    pub license: Option<license::LicenseBlock>,
    /// Signer public key (if signed and verified)
    pub signer_public_key: Option<[u8; 32]>,
}

/// Load an Arrow dataset from the .ald format (unencrypted only)
///
/// For encrypted or signed datasets, use `load_with_options`.
///
/// # Errors
///
/// Returns error if dataset is encrypted, or if deserialization,
/// decompression, or checksum validation fails.
pub fn load<R: std::io::Read>(reader: &mut R) -> Result<LoadedDataset> {
    load_with_options(reader, &LoadOptions::default())
}

/// Load an Arrow dataset with decryption and verification options
///
/// # Errors
///
/// Returns error if deserialization, decompression, decryption,
/// signature verification, license validation, or checksum validation fails.
#[allow(clippy::too_many_lines)]
pub fn load_with_options<R: std::io::Read>(
    reader: &mut R,
    options: &LoadOptions,
) -> Result<LoadedDataset> {
    use arrow::ipc::reader::StreamReader;

    // Read all data
    let mut all_data = Vec::new();
    reader
        .read_to_end(&mut all_data)
        .map_err(Error::io_no_path)?;

    if all_data.len() < HEADER_SIZE + 4 {
        return Err(Error::Format("File too small".to_string()));
    }

    // Split off checksum (last 4 bytes)
    let checksum_offset = all_data.len() - 4;
    let stored_checksum = u32::from_le_bytes([
        all_data[checksum_offset],
        all_data[checksum_offset + 1],
        all_data[checksum_offset + 2],
        all_data[checksum_offset + 3],
    ]);

    // Verify checksum
    let computed_checksum = crc32(&all_data[..checksum_offset]);
    if stored_checksum != computed_checksum {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }

    // Parse header
    let header = Header::from_bytes(&all_data[..HEADER_SIZE])?;

    // Parse metadata
    let metadata_start = HEADER_SIZE;
    let metadata_end = metadata_start + header.metadata_size as usize;
    let metadata: Metadata = rmp_serde::from_slice(&all_data[metadata_start..metadata_end])
        .map_err(|e| Error::Format(format!("Metadata parse error: {e}")))?;

    // Skip schema (embedded in payload IPC stream)
    let schema_end = metadata_end + header.schema_size as usize;

    // Determine encryption header size
    let encryption_header_size = if header.is_encrypted() {
        // Need to peek at mode byte to determine header size
        if all_data.len() <= schema_end {
            return Err(Error::Format("Missing encryption header".to_string()));
        }
        let mode = all_data[schema_end];
        #[cfg(feature = "format-encryption")]
        let size = if mode == encryption::mode::PASSWORD {
            1 + 16 + 12 // mode + salt + nonce
        } else {
            1 + 32 + 12 // mode + ephemeral_pub + nonce
        };
        #[cfg(not(feature = "format-encryption"))]
        {
            let _ = mode;
            return Err(Error::Format(
                "Dataset is encrypted but format-encryption feature is not enabled".to_string(),
            ));
        }
        #[cfg(feature = "format-encryption")]
        size
    } else {
        0
    };

    let payload_start = schema_end + encryption_header_size;
    let payload_end = payload_start + header.payload_size as usize;

    if payload_end > checksum_offset {
        return Err(Error::Format("Payload extends beyond data".to_string()));
    }

    // Extract and decrypt payload if encrypted
    let compressed_payload: Vec<u8> = if header.is_encrypted() {
        #[cfg(feature = "format-encryption")]
        {
            let enc_header = &all_data[schema_end..payload_start];
            let ciphertext = &all_data[payload_start..payload_end];

            let decryption_params = options.decryption.as_ref().ok_or_else(|| {
                Error::Format("Dataset is encrypted but no decryption params provided".to_string())
            })?;

            decrypt_payload(enc_header, ciphertext, decryption_params)?
        }
        #[cfg(not(feature = "format-encryption"))]
        {
            return Err(Error::Format(
                "Dataset is encrypted but format-encryption feature is not enabled".to_string(),
            ));
        }
    } else {
        all_data[payload_start..payload_end].to_vec()
    };

    // Parse trailing blocks (signature, license) working backwards from checksum
    #[allow(unused_mut)]
    let mut trailing_offset = payload_end;
    #[allow(unused_mut)]
    let mut signer_public_key: Option<[u8; 32]> = None;
    let mut license_block: Option<license::LicenseBlock> = None;

    // Signature block comes right after payload (if present)
    if header.is_signed() {
        #[cfg(feature = "format-signing")]
        {
            let sig_start = trailing_offset;
            let sig_end = sig_start + signing::SignatureBlock::SIZE;

            if sig_end > checksum_offset {
                return Err(Error::Format(
                    "Signature block extends beyond data".to_string(),
                ));
            }

            let sig_block = signing::SignatureBlock::from_bytes(&all_data[sig_start..sig_end])?;

            // Verify signature if trusted keys provided
            if !options.trusted_keys.is_empty() {
                // Signature covers: header + metadata + schema + enc_header + payload
                let signed_data = &all_data[..sig_start];

                // Check if signer is in trusted keys
                if !options.trusted_keys.contains(&sig_block.public_key) {
                    return Err(Error::Format("Signer not in trusted keys list".to_string()));
                }

                sig_block.verify(signed_data)?;
            }

            // Always record the signer's public key
            signer_public_key = Some(sig_block.public_key);
            trailing_offset = sig_end;
        }
        #[cfg(not(feature = "format-signing"))]
        {
            return Err(Error::Format(
                "Dataset is signed but format-signing feature is not enabled".to_string(),
            ));
        }
    }

    // License block comes after signature (if present)
    if header.is_licensed() {
        let lic_start = trailing_offset;
        // License block is variable size, need to parse it
        if lic_start >= checksum_offset {
            return Err(Error::Format("Missing license block".to_string()));
        }

        let lic_data = &all_data[lic_start..checksum_offset];
        license_block = Some(license::LicenseBlock::from_bytes(lic_data)?);

        // Verify license if requested
        if options.verify_license {
            if let Some(ref lic) = license_block {
                lic.verify()?;
            }
        }
    }

    // Decompress payload
    let decompressed_payload = match header.compression {
        Compression::None => compressed_payload,
        Compression::ZstdL3 | Compression::ZstdL19 => {
            zstd::decode_all(compressed_payload.as_slice())
                .map_err(|e| Error::Format(format!("Zstd decompression error: {e}")))?
        }
        Compression::Lz4 => {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(compressed_payload.as_slice());
            let mut decompressed = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut decompressed)
                .map_err(|e| Error::Format(format!("LZ4 decompression error: {e}")))?;
            decompressed
        }
    };

    // Parse Arrow IPC stream
    let cursor = std::io::Cursor::new(decompressed_payload);
    let stream_reader = StreamReader::try_new(cursor, None).map_err(Error::Arrow)?;

    let batches: Vec<_> = stream_reader
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Error::Arrow)?;

    Ok(LoadedDataset {
        header,
        metadata,
        batches,
        license: license_block,
        signer_public_key,
    })
}

/// Decrypt payload using the provided parameters
#[cfg(feature = "format-encryption")]
fn decrypt_payload(
    enc_header: &[u8],
    ciphertext: &[u8],
    params: &encryption::DecryptionParams,
) -> Result<Vec<u8>> {
    if enc_header.is_empty() {
        return Err(Error::Format("Empty encryption header".to_string()));
    }

    let mode = enc_header[0];

    match (mode, params) {
        (encryption::mode::PASSWORD, encryption::DecryptionParams::Password(password)) => {
            if enc_header.len() < 1 + 16 + 12 {
                return Err(Error::Format(
                    "Invalid password encryption header".to_string(),
                ));
            }
            let mut salt = [0u8; 16];
            let mut nonce = [0u8; 12];
            salt.copy_from_slice(&enc_header[1..17]);
            nonce.copy_from_slice(&enc_header[17..29]);

            encryption::decrypt_password(ciphertext, password, &salt, &nonce)
        }
        (encryption::mode::RECIPIENT, encryption::DecryptionParams::PrivateKey(private_key)) => {
            if enc_header.len() < 1 + 32 + 12 {
                return Err(Error::Format(
                    "Invalid recipient encryption header".to_string(),
                ));
            }
            let mut ephemeral_pub = [0u8; 32];
            let mut nonce = [0u8; 12];
            ephemeral_pub.copy_from_slice(&enc_header[1..33]);
            nonce.copy_from_slice(&enc_header[33..45]);

            encryption::decrypt_recipient(ciphertext, private_key, &ephemeral_pub, &nonce)
        }
        (encryption::mode::PASSWORD, encryption::DecryptionParams::PrivateKey(_)) => Err(
            Error::Format("Dataset encrypted with password but private key provided".to_string()),
        ),
        (encryption::mode::RECIPIENT, encryption::DecryptionParams::Password(_)) => Err(
            Error::Format("Dataset encrypted for recipient but password provided".to_string()),
        ),
        _ => Err(Error::Format(format!("Unknown encryption mode: {mode}"))),
    }
}

/// Save dataset to a file path
///
/// # Errors
///
/// Returns error if file creation or serialization fails.
pub fn save_to_file<P: AsRef<std::path::Path>>(
    path: P,
    batches: &[arrow::array::RecordBatch],
    dataset_type: DatasetType,
    options: &SaveOptions,
) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())
        .map_err(|e| Error::io(e, path.as_ref().to_path_buf()))?;
    let mut writer = std::io::BufWriter::new(file);
    save(&mut writer, batches, dataset_type, options)
}

/// Load dataset from a file path
///
/// # Errors
///
/// Returns error if file reading or deserialization fails.
pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<LoadedDataset> {
    load_from_file_with_options(path, &LoadOptions::default())
}

/// Load dataset from a file path with decryption and verification options
///
/// # Errors
///
/// Returns error if file reading, decryption, verification, or deserialization
/// fails.
pub fn load_from_file_with_options<P: AsRef<std::path::Path>>(
    path: P,
    options: &LoadOptions,
) -> Result<LoadedDataset> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| Error::io(e, path.as_ref().to_path_buf()))?;
    let mut reader = std::io::BufReader::new(file);
    load_with_options(&mut reader, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_bytes() {
        assert_eq!(&MAGIC, b"ALDF");
    }

    #[test]
    fn test_header_roundtrip() {
        let header = Header {
            version: (1, 2),
            dataset_type: DatasetType::ImageClassification,
            metadata_size: 1024,
            payload_size: 65536,
            uncompressed_size: 131072,
            compression: Compression::ZstdL3,
            flags: flags::SIGNED | flags::ENCRYPTED,
            schema_size: 256,
            num_rows: 50000,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);

        let parsed = Header::from_bytes(&bytes).expect("parse failed");
        assert_eq!(parsed, header);
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"XXXX");

        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid magic"));
    }

    #[test]
    fn test_header_flags() {
        let mut header = Header::new(DatasetType::Tabular);
        assert!(!header.is_encrypted());
        assert!(!header.is_signed());

        header.flags |= flags::ENCRYPTED;
        assert!(header.is_encrypted());
        assert!(!header.is_signed());

        header.flags |= flags::SIGNED;
        assert!(header.is_encrypted());
        assert!(header.is_signed());
    }

    #[test]
    fn test_dataset_type_roundtrip() {
        for dt in [
            DatasetType::Tabular,
            DatasetType::ImageClassification,
            DatasetType::QuestionAnswering,
            DatasetType::Custom,
        ] {
            let value = dt.as_u16();
            let parsed = DatasetType::from_u16(value);
            assert_eq!(parsed, Some(dt));
        }
    }

    #[test]
    fn test_compression_roundtrip() {
        for c in [
            Compression::None,
            Compression::ZstdL3,
            Compression::ZstdL19,
            Compression::Lz4,
        ] {
            let value = c.as_u8();
            let parsed = Compression::from_u8(value);
            assert_eq!(parsed, Some(c));
        }
    }

    #[test]
    fn test_crc32_known_values() {
        // Empty
        assert_eq!(crc32(&[]), 0x0000_0000);
        // "123456789"
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_header_size_is_32() {
        let header = Header::new(DatasetType::Tabular);
        assert_eq!(header.to_bytes().len(), 32);
    }

    fn create_test_batch() -> arrow::array::RecordBatch {
        use std::sync::Arc;

        use arrow::{
            array::{Int32Array, StringArray},
            datatypes::{DataType, Field, Schema},
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie", "Diana", "Eve"]);

        arrow::array::RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(name_array)])
            .expect("create batch")
    }

    #[test]
    fn test_save_load_roundtrip_no_compression() {
        let batch = create_test_batch();
        let batches = vec![batch];

        let options = SaveOptions {
            compression: Compression::None,
            metadata: Some(Metadata {
                name: Some("test-dataset".to_string()),
                version: Some("1.0.0".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

        assert_eq!(loaded.header.dataset_type, DatasetType::Tabular);
        assert_eq!(loaded.header.compression, Compression::None);
        assert_eq!(loaded.header.num_rows, 5);
        assert_eq!(loaded.metadata.name, Some("test-dataset".to_string()));
        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.batches[0].num_rows(), 5);
    }

    #[test]
    fn test_save_load_roundtrip_zstd() {
        let batch = create_test_batch();
        let batches = vec![batch];

        let options = SaveOptions {
            compression: Compression::ZstdL3,
            metadata: None,
            ..Default::default()
        };

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

        assert_eq!(loaded.header.compression, Compression::ZstdL3);
        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.batches[0].num_rows(), 5);
    }

    #[test]
    fn test_save_load_roundtrip_lz4() {
        let batch = create_test_batch();
        let batches = vec![batch];

        let options = SaveOptions {
            compression: Compression::Lz4,
            metadata: None,
            ..Default::default()
        };

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

        assert_eq!(loaded.header.compression, Compression::Lz4);
        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.batches[0].num_rows(), 5);
    }

    #[test]
    fn test_save_empty_batches_fails() {
        let batches: Vec<arrow::array::RecordBatch> = vec![];
        let options = SaveOptions::default();

        let mut buf = Vec::new();
        let result = save(&mut buf, &batches, DatasetType::Tabular, &options);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_checksum_mismatch_detected() {
        let batch = create_test_batch();
        let batches = vec![batch];

        let mut buf = Vec::new();
        save(
            &mut buf,
            &batches,
            DatasetType::Tabular,
            &SaveOptions::default(),
        )
        .expect("save failed");

        // Corrupt a byte in the payload
        if buf.len() > 50 {
            buf[50] ^= 0xFF;
        }

        let result = load(&mut std::io::Cursor::new(&buf));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Checksum"));
    }

    #[test]
    fn test_multiple_batches() {
        let batch1 = create_test_batch();
        let batch2 = create_test_batch();
        let batches = vec![batch1, batch2];

        let mut buf = Vec::new();
        save(
            &mut buf,
            &batches,
            DatasetType::Tabular,
            &SaveOptions::default(),
        )
        .expect("save failed");

        let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

        assert_eq!(loaded.header.num_rows, 10);
        assert_eq!(loaded.batches.len(), 2);
    }

    #[cfg(feature = "format-encryption")]
    #[test]
    fn test_save_load_password_encrypted() {
        let batch = create_test_batch();
        let batches = vec![batch];

        let options = SaveOptions::default().with_password("test-password-123");

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Verify it's marked as encrypted
        let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
        assert!(header.is_encrypted());

        // Load with correct password
        let load_opts = LoadOptions::default().with_password("test-password-123");
        let loaded =
            load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.batches[0].num_rows(), 5);
    }

    #[cfg(feature = "format-encryption")]
    #[test]
    fn test_save_load_password_wrong_password() {
        let batch = create_test_batch();
        let batches = vec![batch];

        let options = SaveOptions::default().with_password("correct-password");

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Load with wrong password should fail
        let load_opts = LoadOptions::default().with_password("wrong-password");
        let result = load_with_options(&mut std::io::Cursor::new(&buf), &load_opts);

        assert!(result.is_err());
    }

    #[cfg(feature = "format-encryption")]
    #[test]
    fn test_save_load_recipient_encrypted() {
        use x25519_dalek::{PublicKey, StaticSecret};

        let batch = create_test_batch();
        let batches = vec![batch];

        // Generate recipient key pair
        let mut key_bytes = [0u8; 32];
        getrandom::getrandom(&mut key_bytes).expect("rng failed");
        let recipient_secret = StaticSecret::from(key_bytes);
        let recipient_public = PublicKey::from(&recipient_secret);

        let options = SaveOptions::default().with_recipient(*recipient_public.as_bytes());

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Verify it's marked as encrypted
        let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
        assert!(header.is_encrypted());

        // Load with correct private key
        let load_opts = LoadOptions::default().with_private_key(key_bytes);
        let loaded =
            load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.batches[0].num_rows(), 5);
    }

    #[cfg(feature = "format-signing")]
    #[test]
    fn test_save_load_signed() {
        use crate::format::signing::SigningKeyPair;

        let batch = create_test_batch();
        let batches = vec![batch];

        let key_pair = SigningKeyPair::generate().expect("keygen");
        let public_key = key_pair.public_key_bytes();

        let options = SaveOptions::default().with_signing_key(key_pair);

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Verify it's marked as signed
        let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
        assert!(header.is_signed());

        // Load and verify signature
        let load_opts = LoadOptions::default().with_trusted_key(public_key);
        let loaded =
            load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.signer_public_key, Some(public_key));
    }

    #[cfg(feature = "format-signing")]
    #[test]
    fn test_save_load_signed_untrusted_signer() {
        use crate::format::signing::SigningKeyPair;

        let batch = create_test_batch();
        let batches = vec![batch];

        let key_pair = SigningKeyPair::generate().expect("keygen");
        let other_key_pair = SigningKeyPair::generate().expect("keygen2");
        let other_public_key = other_key_pair.public_key_bytes();

        let options = SaveOptions::default().with_signing_key(key_pair);

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Load with different trusted key should fail
        let load_opts = LoadOptions::default().with_trusted_key(other_public_key);
        let result = load_with_options(&mut std::io::Cursor::new(&buf), &load_opts);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not in trusted keys"));
    }

    #[cfg(feature = "format-encryption")]
    #[test]
    fn test_save_load_with_license() {
        use std::time::{SystemTime, UNIX_EPOCH};

        use crate::format::license::{generate_license_id, hash_licensee, LicenseBuilder};

        let batch = create_test_batch();
        let batches = vec![batch];

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let license_id = generate_license_id().expect("license id gen failed");
        let licensee_hash = hash_licensee("test@example.com");
        let license = LicenseBuilder::new(license_id, licensee_hash)
            .expires_at(now + 3600) // Valid for 1 hour
            .build();

        let options = SaveOptions::default().with_license(license);

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Verify it's marked as licensed
        let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
        assert!(header.is_licensed());

        // Load with license verification
        let load_opts = LoadOptions::default().verify_license();
        let loaded =
            load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

        assert_eq!(loaded.batches.len(), 1);
        assert!(loaded.license.is_some());
    }

    #[cfg(feature = "format-encryption")]
    #[test]
    fn test_save_load_with_expired_license() {
        use std::time::{SystemTime, UNIX_EPOCH};

        use crate::format::license::{generate_license_id, hash_licensee, LicenseBuilder};

        let batch = create_test_batch();
        let batches = vec![batch];

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let license_id = generate_license_id().expect("license id gen failed");
        let licensee_hash = hash_licensee("test@example.com");
        let license = LicenseBuilder::new(license_id, licensee_hash)
            .expires_at(now - 3600) // Expired 1 hour ago
            .build();

        let options = SaveOptions::default().with_license(license);

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Load with license verification should fail
        let load_opts = LoadOptions::default().verify_license();
        let result = load_with_options(&mut std::io::Cursor::new(&buf), &load_opts);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("License expired"));
    }

    #[cfg(all(feature = "format-encryption", feature = "format-signing"))]
    #[test]
    fn test_save_load_encrypted_and_signed() {
        use crate::format::signing::SigningKeyPair;

        let batch = create_test_batch();
        let batches = vec![batch];

        let key_pair = SigningKeyPair::generate().expect("keygen");
        let public_key = key_pair.public_key_bytes();

        let options = SaveOptions::default()
            .with_password("secure-password")
            .with_signing_key(key_pair);

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Verify flags
        let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
        assert!(header.is_encrypted());
        assert!(header.is_signed());

        // Load with correct credentials
        let load_opts = LoadOptions::default()
            .with_password("secure-password")
            .with_trusted_key(public_key);
        let loaded =
            load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.signer_public_key, Some(public_key));
    }

    #[cfg(all(feature = "format-encryption", feature = "format-signing"))]
    #[test]
    fn test_save_load_full_security_suite() {
        use std::time::{SystemTime, UNIX_EPOCH};

        use crate::format::{
            license::{generate_license_id, hash_licensee, LicenseBuilder},
            signing::SigningKeyPair,
        };

        let batch = create_test_batch();
        let batches = vec![batch];

        let key_pair = SigningKeyPair::generate().expect("keygen");
        let public_key = key_pair.public_key_bytes();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let license_id = generate_license_id().expect("license id gen failed");
        let licensee_hash = hash_licensee("enterprise@company.com");
        let license = LicenseBuilder::new(license_id, licensee_hash)
            .expires_at(now + 86400)
            .seat_limit(100)
            .build();

        let options = SaveOptions::default()
            .with_password("enterprise-secret")
            .with_signing_key(key_pair)
            .with_license(license)
            .with_metadata(Metadata {
                name: Some("enterprise-dataset".to_string()),
                version: Some("2.0.0".to_string()),
                ..Default::default()
            });

        let mut buf = Vec::new();
        save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

        // Verify all flags
        let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
        assert!(header.is_encrypted());
        assert!(header.is_signed());
        assert!(header.is_licensed());

        // Load with full verification
        let load_opts = LoadOptions::default()
            .with_password("enterprise-secret")
            .with_trusted_key(public_key)
            .verify_license();
        let loaded =
            load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

        assert_eq!(loaded.batches.len(), 1);
        assert_eq!(loaded.signer_public_key, Some(public_key));
        assert!(loaded.license.is_some());
        assert_eq!(loaded.metadata.name, Some("enterprise-dataset".to_string()));
    }

    #[test]
    fn test_dataset_type_all_variants() {
        // Test all DatasetType variants
        let types = [
            (DatasetType::Tabular, 0x0001),
            (DatasetType::TimeSeries, 0x0002),
            (DatasetType::Graph, 0x0003),
            (DatasetType::Spatial, 0x0004),
            (DatasetType::TextCorpus, 0x0010),
            (DatasetType::TextClassification, 0x0011),
            (DatasetType::TextPairs, 0x0012),
            (DatasetType::SequenceLabeling, 0x0013),
            (DatasetType::QuestionAnswering, 0x0014),
            (DatasetType::Summarization, 0x0015),
            (DatasetType::Translation, 0x0016),
            (DatasetType::ImageClassification, 0x0020),
            (DatasetType::ObjectDetection, 0x0021),
            (DatasetType::Segmentation, 0x0022),
            (DatasetType::ImagePairs, 0x0023),
            (DatasetType::Video, 0x0024),
            (DatasetType::AudioClassification, 0x0030),
            (DatasetType::SpeechRecognition, 0x0031),
            (DatasetType::SpeakerIdentification, 0x0032),
            (DatasetType::UserItemRatings, 0x0040),
            (DatasetType::ImplicitFeedback, 0x0041),
            (DatasetType::SequentialRecs, 0x0042),
            (DatasetType::ImageText, 0x0050),
            (DatasetType::AudioText, 0x0051),
            (DatasetType::VideoText, 0x0052),
            (DatasetType::Custom, 0x00FF),
        ];

        for (dt, expected_value) in types {
            assert_eq!(dt.as_u16(), expected_value);
            assert_eq!(DatasetType::from_u16(expected_value), Some(dt));
        }
    }

    #[test]
    fn test_dataset_type_invalid_value() {
        assert_eq!(DatasetType::from_u16(0x9999), None);
    }

    #[test]
    fn test_compression_invalid_value() {
        assert_eq!(Compression::from_u8(0x99), None);
    }

    #[test]
    fn test_header_is_streaming() {
        let mut header = Header::new(DatasetType::Tabular);
        assert!(!header.is_streaming());

        header.flags |= flags::STREAMING;
        assert!(header.is_streaming());
    }

    #[test]
    fn test_header_is_licensed() {
        let mut header = Header::new(DatasetType::Tabular);
        assert!(!header.is_licensed());

        header.flags |= flags::LICENSED;
        assert!(header.is_licensed());
    }

    #[test]
    #[cfg(feature = "provenance")]
    fn test_sha256_hex_known_value() {
        // Test against known SHA-256 hash
        let hash = super::sha256_hex(b"Hello, World!");
        // SHA-256 of "Hello, World!" is known
        assert_eq!(
            hash,
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        );
    }

    #[test]
    #[cfg(feature = "provenance")]
    fn test_sha256_hex_length() {
        let hash = super::sha256_hex(b"test data");
        assert_eq!(hash.len(), 64); // 256 bits = 32 bytes = 64 hex chars
    }

    #[test]
    #[cfg(feature = "provenance")]
    fn test_sha256_hex_empty() {
        let hash = super::sha256_hex(b"");
        // SHA-256 of empty string is known
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    #[cfg(feature = "provenance")]
    fn test_sha256_hex_different_inputs() {
        let hash1 = super::sha256_hex(b"input1");
        let hash2 = super::sha256_hex(b"input2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_metadata_sha256_field() {
        let mut metadata = Metadata::default();
        assert!(metadata.sha256.is_none());

        metadata.sha256 = Some("abc123".to_string());
        assert_eq!(metadata.sha256, Some("abc123".to_string()));
    }
}
