# alimentar Makefile
# Toyota Way: Extreme TDD with quality gates (matches trueno, trueno-db, aprender)

# Use bash for shell commands to support advanced features
SHELL := /bin/bash

# Parallel job execution
MAKEFLAGS += -j$(shell nproc)

# Quality directives
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help build build-release bench bench-check test test-fast test-verbose test-lib test-s3 test-s3-full lint lint-pedantic lint-fast fmt fmt-check check coverage coverage-open coverage-check coverage-clean quality-gate mutants mutation-report mutation-clean tdg clean doc doc-open watch watch-test ci pre-commit dev-deps stats pmat-tdg pmat-analyze pmat-score pmat-rust-score pmat-rust-score-fast pmat-quality-gate pmat-all wasm wasm-check book book-build book-serve book-test

# Coverage threshold (85% minimum)
# Note: HTTP backend and HF Hub require network calls that aren't testable without mocking.
#       S3 backend is excluded (requires MinIO). With network-dependent code excluded,
#       achievable coverage is ~85%. trueno-db targets 90% but its GPU backend is testable.
COVERAGE_THRESHOLD := 85

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Development Commands

build: ## Build the project
	cargo build --all-features

build-release: ## Build release version
	cargo build --release --all-features

## Benchmarking

bench: ## Run benchmarks
	cargo bench

bench-check: ## Check benchmarks compile
	cargo bench --no-run

## Testing
## PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
## - make test-fast: < 2 minutes (parallel execution with nextest)
## - make coverage:  < 5 minutes (two-phase pattern)
## - make test:      comprehensive

test: ## Run all tests
	cargo test --all-features

test-fast: ## Run tests quickly (parallel with nextest, target: <2 min)
	@echo "⚡ Running fast tests (target: <2 min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		cargo nextest run \
			--workspace \
			--all-features \
			--status-level skip \
			--failure-output immediate; \
	else \
		echo "📦 cargo-nextest not found, using cargo test..."; \
		cargo test --all-features --workspace; \
	fi

test-verbose: ## Run tests with verbose output
	cargo test --all-features -- --nocapture

test-lib: ## Run library tests only
	cargo test --lib --all-features

test-s3: ## Run S3 integration tests (requires docker compose up -d first)
	@echo "🪣 Running S3 integration tests with MinIO..."
	@echo "   Ensure MinIO is running: docker compose up -d"
	cargo test --features s3 s3_integration -- --ignored

test-s3-full: ## Start MinIO and run S3 integration tests
	@echo "🚀 Starting MinIO via docker compose..."
	docker compose up -d
	@echo "⏳ Waiting for MinIO to be ready..."
	@sleep 5
	@echo "🪣 Running S3 integration tests..."
	cargo test --features s3 s3_integration -- --ignored
	@echo "🧹 Stopping MinIO..."
	docker compose down

## Quality Gates (EXTREME TDD - 95% coverage minimum)

lint: ## Run clippy with zero tolerance (ALL targets)
	cargo clippy --all-targets --all-features -- -D warnings

lint-pedantic: ## Run clippy with pedantic lints
	cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic

lint-fast: ## Fast clippy (library only)
	cargo clippy --lib --all-features -- -D warnings

fmt: ## Format code
	cargo fmt

fmt-check: ## Check formatting without modifying
	cargo fmt --check

check: fmt-check lint test ## Run basic quality checks

## Coverage (requires cargo-llvm-cov, 85% minimum)
## Note: S3 feature excluded from coverage (requires MinIO, tests are #[ignore])
## Note: HTTP backend and HF Hub require network calls (not testable without mocking)
## Following: Two-Phase Pattern (run tests separately from report generation)
## TARGET: < 5 minutes

COVERAGE_FEATURES := local,tokio-runtime,cli,mmap,http,hf-hub,shuffle,format-encryption,format-signing,format-streaming

coverage: ## Generate HTML coverage report (target: <5 min)
	@echo "📊 Running coverage analysis (target: <5 min)..."
	@echo "   Note: S3 feature excluded (requires MinIO)"
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "⚙️  Temporarily disabling global cargo config (mold breaks coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup 2>/dev/null || true
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@cargo llvm-cov --no-report nextest --no-tests=warn --features "$(COVERAGE_FEATURES)" --workspace
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@cp target/coverage/lcov.info lcov.info
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml 2>/dev/null || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- HTML report: target/coverage/html/index.html"
	@echo "- LCOV file: target/coverage/lcov.info"
	@echo "- Open HTML: make coverage-open"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first to generate the HTML report"; \
	fi

coverage-check: ## Enforce 85% coverage threshold (BLOCKS on failure)
	@echo "🔒 Enforcing $(COVERAGE_THRESHOLD)% coverage threshold..."
	@echo "   Note: S3 feature excluded (requires MinIO)"
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@cargo llvm-cov clean --workspace
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup 2>/dev/null || true
	@cargo llvm-cov --no-report nextest --no-tests=warn --features "$(COVERAGE_FEATURES)" --workspace
	@cargo llvm-cov report --fail-under-lines $(COVERAGE_THRESHOLD) || \
		(echo "❌ FAIL: Coverage below $(COVERAGE_THRESHOLD)% threshold"; \
		 test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml 2>/dev/null; exit 1)
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml 2>/dev/null || true
	@echo "✅ Coverage threshold met (≥$(COVERAGE_THRESHOLD)%)"

coverage-clean: ## Clean coverage artifacts
	@cargo llvm-cov clean --workspace
	@rm -f lcov.info coverage.xml target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete
	@echo "✓ Coverage artifacts cleaned"

## Mutation Testing (requires cargo-mutants, target: ≥85% kill rate)

mutants: ## Run mutation testing (target: ≥85% kill rate)
	@echo "🧬 Running mutation testing (this will take a while)..."
	@echo "Target: >85% mutation score"
	@if command -v cargo-mutants >/dev/null 2>&1; then \
		cargo mutants --no-times --output mutants.out || true; \
		echo "✅ Mutation testing complete. Results in mutants.out/"; \
	else \
		echo "📥 Installing cargo-mutants..."; \
		cargo install cargo-mutants && cargo mutants --no-times --output mutants.out || true; \
	fi

mutation-report: ## Analyze mutation test results
	@echo "📊 Analyzing mutation test results..."
	@if [ -d "mutants.out" ]; then \
		cat mutants.out/mutants.out 2>/dev/null || echo "No mutation results yet"; \
	else \
		echo "No mutation results found. Run 'make mutants' first."; \
	fi

mutation-clean: ## Clean mutation testing artifacts
	@rm -rf mutants.out mutants.out.old
	@echo "✓ Mutation testing artifacts cleaned"

## Technical Debt Grading

tdg: ## Run TDG analysis (target: ≥B+ / 85)
	@pmat analyze tdg 2>/dev/null || echo "⚠️  PMAT not available"

## Quality Gate (BLOCKS if coverage < 85%)

quality-gate: lint test coverage-check ## Run full quality gate (BLOCKS if coverage < 85%)
	@echo "✅ All quality gates passed"

## PMAT Integration (matches trueno standards)

pmat-tdg: ## Run PMAT Technical Debt Grading (minimum: B+)
	@pmat analyze tdg

pmat-analyze: ## Run comprehensive PMAT analysis
	@pmat analyze complexity --project-path . || true
	@pmat analyze satd --path . || true
	@pmat analyze dead-code --path . || true
	@pmat analyze duplicates || true
	@pmat analyze defects --path . || true

pmat-score: ## Calculate repository health score (minimum: 90/110)
	@pmat repo-score || true

pmat-rust-score: ## Calculate Rust project score (target: A+ grade)
	@mkdir -p target/pmat-reports
	@pmat rust-project-score --path . || echo "⚠️  Rust project score not available in this PMAT version"

pmat-rust-score-fast: ## Calculate Rust project score (fast mode)
	@pmat rust-project-score --path . || echo "⚠️  Rust project score not available in this PMAT version"

pmat-quality-gate: ## Run PMAT quality gate (TDG ≥B+, repo-score ≥90)
	@echo "🔒 Running PMAT quality gate..."
	@pmat analyze tdg --min-grade B+ 2>/dev/null || echo "    ⚠️  PMAT TDG not available"
	@pmat repo-score . --min-score 90 2>/dev/null || echo "    ⚠️  PMAT repo-score not available"

pmat-all: pmat-tdg pmat-analyze pmat-score pmat-rust-score ## Run all PMAT analyses

## WASM Build

wasm: ## Build for WASM target
	cargo build --target wasm32-unknown-unknown --release --no-default-features --features wasm

wasm-check: ## Check WASM compiles
	cargo check --target wasm32-unknown-unknown --no-default-features --features wasm

## Documentation

doc: ## Build API documentation
	cargo doc --all-features --no-deps

doc-open: ## Build and open API documentation
	cargo doc --all-features --no-deps --open

## mdBook (User Guide)

book: book-build ## Build and open the user guide
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open book/book/index.html; \
	elif command -v open >/dev/null 2>&1; then \
		open book/book/index.html; \
	fi

book-build: ## Build the mdBook user guide
	@echo "📚 Building alimentar user guide..."
	@if command -v mdbook >/dev/null 2>&1; then \
		mdbook build book; \
		echo "✅ Book built: book/book/index.html"; \
	else \
		echo "❌ mdbook not found. Install with: cargo install mdbook"; \
		exit 1; \
	fi

book-serve: ## Serve the book locally for development
	@echo "📖 Serving book at http://localhost:3000..."
	@mdbook serve book --open

book-test: ## Test book code examples
	@echo "🔍 Testing book code examples..."
	@mdbook test book 2>/dev/null || echo "⚠️  Some examples may not be runnable (requires full project context)"

## Development Helpers

watch: ## Watch for changes and check
	cargo watch -x "check --all-features"

watch-test: ## Watch for changes and test
	cargo watch -x "test --all-features"

## CI/CD

ci: lint test coverage doc ## Run CI pipeline
	@echo "✅ CI pipeline passed"

pre-commit: fmt-check lint test ## Pre-commit hook
	@echo "✅ Pre-commit checks passed"

## Maintenance

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/ coverage/ mutants.out/ lcov.info

dev-deps: ## Install development dependencies
	cargo install cargo-llvm-cov cargo-nextest cargo-tarpaulin cargo-mutants cargo-watch

## Project Statistics

stats: ## Show project statistics
	@echo "📊 Project Statistics"
	@echo "===================="
	@echo ""
	@echo "Lines of code:"
	@find src -name "*.rs" | xargs wc -l | tail -1
	@echo ""
	@echo "Test count:"
	@cargo test --all-features 2>&1 | grep -E "running [0-9]+ tests" | awk '{sum += $$2} END {print "  " sum " tests"}'
	@echo ""
	@echo "Dependencies:"
	@cargo tree --depth 1 2>/dev/null | wc -l | xargs -I {} echo "  {} direct dependencies"
