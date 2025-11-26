# PMAT Development Roadmap

## Completed: v0.1.0 Core
- **Status**: COMPLETED
- **Features**: ArrowDataset, DataLoader, Transforms, LocalBackend, MemoryBackend, CLI

## Current Sprint: v0.2.0 Storage & Registry
- **Duration**: 2025-11-25 to 2025-12-09
- **Priority**: P0
- **Quality Gates**: Complexity ≤ 15, SATD = 0, Coverage ≥ 90%

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| S3-01 | Implement S3Backend with aws-sdk-s3 | ✅ done | high | P0 |
| HTTP-01 | Implement HttpBackend with reqwest | ✅ done | medium | P0 |
| REG-01 | Registry index format and DatasetMetadata | ✅ done | medium | P0 |
| REG-02 | Registry publish operation | ✅ done | medium | P0 |
| REG-03 | Registry pull operation | ✅ done | medium | P0 |
| REG-04 | Registry list and search | ✅ done | low | P1 |
| STREAM-01 | StreamingDataset implementation | pending | high | P0 |
| CLI-01 | Add registry CLI commands | pending | medium | P1 |
| TEST-01 | S3 provider integration tests (MinIO) | pending | medium | P0 |

### Definition of Done
- [ ] All P0 tasks completed
- [ ] Quality gates passed (90% coverage, 0 SATD)
- [ ] Documentation updated
- [ ] All tests passing
- [ ] Changelog updated
- [ ] Zero unwrap() in library code

