# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2025-11-29

### Added
- `DatasetCardValidator` utility methods: `is_valid_task_category()`, `is_valid_license()`, `is_valid_size_category()`, `suggest_task_category()`
- Native HuggingFace Hub upload API with `HfHubUploader`
- Abstract quality profile system for configurable quality scoring
- 100-point weighted quality scoring system with grade calculation
- Function signature extraction in doctest parser
- REPL module with command parsing and tab completion
- Doctest extraction from Python source files

### Fixed
- Handle nullable columns in constant check quality validation
- Clippy lint warnings across codebase
- Duplicate `DatasetCardValidator` struct definitions

### Changed
- Test suite expanded from 571 to 1648 tests
- Test coverage improved to 92.10%

## [0.2.0] - 2025-11-27

### Added
- REPL command interface with session management
- Quality scoring profiles (ML training, doctest corpus)
- Drift detection CLI commands
- Federated learning split strategies

## [0.1.0] - 2025-11-26

### Added
- Core `ArrowDataset` for in-memory datasets backed by Arrow RecordBatches
- `StreamingDataset` for lazy loading large datasets
- `DataLoader` with batching, shuffling, and drop_last support
- Storage backends: Local, Memory, S3, HTTP
- HuggingFace Hub integration for importing datasets
- Dataset transforms: Select, Filter, Sort, Sample, Shuffle, Take, Skip, Rename, Drop, Cast, Normalize, FillNull, Unique, Map, Chain
- Local dataset registry with versioning and metadata
- CLI tool for convert, info, head, schema, import, and registry operations
- WASM support for browser environments
- Data quality checking with null/duplicate/outlier detection
- Drift detection with KS, Chi-square, and PSI tests
- Federated learning splits (local, proportional, stratified)
- Native .alimentar format with compression (zstd, lz4)
- Optional encryption and signing for datasets (feature-gated)
- Built-in datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Iris
- Arrow 53 and Parquet 53 support
- Zero-copy data access throughout
- Memory-mapped file support
- Feature flags for optional dependencies (s3, http, hf-hub, wasm)
- Comprehensive test suite with 571 tests

[Unreleased]: https://github.com/paiml/alimentar/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/paiml/alimentar/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/paiml/alimentar/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/paiml/alimentar/releases/tag/v0.1.0
