# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

alimentar ("to feed" in Spanish) is a pure Rust data loading, transformation, and distribution library for the paiml sovereign AI stack. It provides HuggingFace-compatible functionality with sovereignty-first design (local storage default, no mandatory cloud dependency).

## Design Principles

1. **Sovereign-first** - Local storage default, no mandatory cloud dependency
2. **Pure Rust** - No Python, no FFI (WASM-compatible)
3. **Zero-copy** - Arrow RecordBatch throughout
4. **Ecosystem aligned** - Arrow 53, Parquet 53 (matches trueno-db, trueno-graph)

## Build Commands

```bash
# Build
cargo build
cargo build --release

# Test
cargo test
cargo test --all-features

# Lint
cargo fmt --check
cargo clippy -- -D warnings

# Quality gates (when Makefile exists)
make check          # lint + test
make quality-gate   # lint + test + coverage (blocks if <90%)
make mutants        # mutation testing
make coverage       # coverage report
```

## Quality Standards (EXTREME TDD)

| Metric | Target |
|--------|--------|
| Test coverage | ≥85% (HTTP/HF Hub/S3 require network, not testable without mocking) |
| Mutation score | ≥85% |
| Cyclomatic complexity | ≤15 |
| SATD comments | 0 |
| unwrap() calls | 0 (use clippy disallowed-methods) |
| TDG grade | ≥B+ |
| WASM binary | <500KB |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        alimentar                            │
├─────────────────────────────────────────────────────────────┤
│  Importers          │  Core            │  Exporters         │
│  ─────────          │  ────            │  ─────────         │
│  • HuggingFace Hub  │  • Dataset       │  • Local FS        │
│  • Local files      │  • DataLoader    │  • S3-compatible   │
│  • S3-compatible    │  • Transforms    │  • Registry API    │
│  • HTTP/HTTPS       │  • Streaming     │                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   trueno-db             aprender              trueno-viz
   (storage)             (ML/DL)               (WASM/browser)
```

## Core Types

- **Dataset trait** - `len()`, `get()`, `schema()`, `iter()` returning Arrow RecordBatches
- **ArrowDataset** - In-memory or memory-mapped dataset backed by Arrow
- **StreamingDataset** - Lazy/streaming dataset with prefetch
- **DataLoader** - Batching iterator with shuffle, drop_last, num_workers (0 for WASM)
- **Transform trait** - `apply(batch) -> Result<RecordBatch>` for data transformations
- **StorageBackend trait** - Async trait for list/get/put/delete/exists operations

## Feature Flags

```toml
default = ["local", "tokio-runtime"]
local = []                           # Local filesystem
s3 = ["aws-sdk-s3"]                  # S3-compatible backends
http = ["reqwest"]                   # HTTP sources
hf-hub = ["http"]                    # HuggingFace Hub import
tokio-runtime = ["tokio"]            # Async runtime (non-WASM)
wasm = ["wasm-bindgen", "js-sys"]    # Browser/WASM target
```

## WASM Constraints

When targeting WASM:
- No filesystem access → use `MemoryBackend` or `HttpBackend`
- No multi-threading → `num_workers = 0`
- No tokio → use `wasm-bindgen-futures`
- Use `#[cfg(target_arch = "wasm32")]` for WASM-specific code

## Search Ownership

- **alimentar owns**: Registry metadata search (text/tag matching on index)
- **trueno-db owns**: SQL/filter queries, vector/semantic search (delegate to it)

## Configuration Files

| File | Purpose |
|------|---------|
| `.pmat-gates.toml` | Quality gate thresholds |
| `.cargo-mutants.toml` | Mutation testing config |
| `deny.toml` | Dependency policy |
| `renacer.toml` | Deep inspection config |

## CLI Commands (when implemented)

```bash
alimentar import hf squad --output ./data/squad
alimentar convert data.csv data.parquet
alimentar registry list|push|pull
alimentar info|head|schema ./data/train.parquet
```
