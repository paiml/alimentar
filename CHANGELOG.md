# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of alimentar
- Core `ArrowDataset` for in-memory datasets backed by Arrow RecordBatches
- `StreamingDataset` for lazy loading large datasets
- `DataLoader` with batching, shuffling, and drop_last support
- Storage backends: Local, Memory, S3, HTTP
- HuggingFace Hub integration for importing datasets
- Dataset transforms: Select, Filter, Sort, Sample, Shuffle, Take, Skip, Rename, Drop, Cast, Normalize, FillNull, Unique, Map, Chain
- Local dataset registry with versioning and metadata
- CLI tool for convert, info, head, schema, import, and registry operations
- WASM support for browser environments
- Comprehensive test suite with 85%+ coverage

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-XX-XX

### Added
- Initial public release
- Arrow 53 and Parquet 53 support
- Zero-copy data access throughout
- Memory-mapped file support
- Feature flags for optional dependencies (s3, http, hf-hub, wasm)

[Unreleased]: https://github.com/paiml/alimentar/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/alimentar/releases/tag/v0.1.0
