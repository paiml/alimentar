# Data Quality Tooling Improvement Specification

**Document:** ALIM-SPEC-005
**Status:** Draft - In Review
**Author:** PAIML Engineering
**Date:** 2025-11-29
**Toyota Way Principle:** Jidoka (Build Quality In) & Genchi Genbutsu (Go and See)

---

## 1. Executive Summary

This specification defines requirements for comprehensive, self-contained data quality tooling in `alimentar`. Adhering to the **Toyota Way**, specifically the principles of **Jidoka** (built-in quality) and **Respect for People** (eliminating frustrating waste), we establish that quality must be verified at the source.

**Core Philosophy:** "Defects are not inevitable; they are failures of the process."
The tooling aims to transform the developer's workflow from "inspecting quality out" (finding bugs later) to "building quality in" (preventing them immediately) [1].

---

## 2. Current State Analysis (Genchi Genbutsu)

### 2.1 Observed Defects (The "Gemba")

Go-and-see (Genchi Genbutsu) inspection of the `CPython` stdlib doctest extraction pipeline revealed specific defects hidden by current batch processing:

| Issue | Count | Impact | Root Cause Category |
|-------|-------|--------|---------------------|
| Contaminated `expected` field | ~5% | Training data poisoning | **Process** (Parser heuristic failure) |
| Duplicate rows | 42 (2.5%) | Model overfitting risk | **Process** (Lack of standard check) |
| Empty/malformed inputs | ~1% | Pipeline failures | **Input** (Missing Poka-Yoke) |
| Null signature fields | 100% | Missing metadata | **Implementation** (Incomplete feature) |

### 2.2 Tool Failures (Muda of Defect)

Current tooling creates "Waste of Correction" and "Waste of Waiting":
- **False Positives:** 0% Quality Score is meaningless, causing alarm fatigue.
- **Disconnected Tools:** Developers must switch contexts (Muda of Motion) to use DuckDB/Pandas.
- **Late Detection:** Defects are found during training, not extraction.

### 2.3 External Dependency Anti-Pattern

Current reliance on external tools (DuckDB, PyArrow) violates **Jikotei Kanketsu** (Self-Reliancy). A developer cannot "stop and fix" if they lack the tools to diagnose the problem immediately within the environment [2].

---

## 3. Root Cause Analysis (5 Whys)

**Problem:** Doctest extraction produces contaminated training data.

1.  **Why?** The `expected` field contains documentation prose mixed with output.
2.  **Why?** The parser doesn't detect prose after expected output ends.
3.  **Why?** The termination condition only checks for `>>>` or EOF.
4.  **Why?** No heuristic exists to identify natural language vs. code output.
5.  **Why?** The parser was designed for simple doctests, ignoring the complexity of CPython's documentation style.

**Root Cause:** The process lacks a "Standard Work" definition for complex doctest parsing.

---

## 4. Design Principles

### 4.1 Jidoka (Autonomation)

> "Build in quality at every step. When an abnormality occurs, stop and fix it immediately." — Taiichi Ohno [3]

**Application:**
- **Stop the Line:** The `alimentar` pipeline must default to failing on data quality violations unless explicitly overridden.
- **Human-Machine Teaming:** The tool detects the anomaly (machine), the human provides the judgment (fix/override).

### 4.2 Standard Work (Hyojun)

> "Where there is no standard there can be no Kaizen." — Masaaki Imai [4]

**Application:**
- **Quality Profiles:** Define "what good looks like" (Standard Work) for each dataset type (Doctest, CSV, JSON).
- **Versioned Standards:** Profiles are checked into version control, serving as the baseline for improvement.

### 4.3 Visual Management (Andon)

> "Make problems visible so no one can hide them." — Toyota Way Principle 7 [5]

**Application:**
- **Red/Green Reporting:** CLI output must use color and clear symbols to indicate status instantly.
- **No Hidden Defects:** "Unknown" states are treated as failures.

### 4.4 Elimination of Waste (Muda)

> "Eliminate all non-value-adding activities." — Shigeo Shingo [6]

**Application:**
- **Zero Setup:** `alimentar` must run quality checks with zero external configuration or installation (Removing *Muda of Waiting/Processing*).
- **Fast Feedback:** Checks must run in sub-second time for interactive loops.

---

## 5. Functional Requirements

### 5.1 Doctest Parser Fix (Critical)

**Requirement:** ALIM-R001
**Priority:** P0 (Immediate)

The parser must implement a "Poka-Yoke" (mistake-proofing) mechanism to reject prose contamination [6].

```rust
/// Heuristics for detecting prose vs. code output (Poka-Yoke):
fn is_prose_continuation(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() { return false; }

    // 1. Sentence pattern: Capital followed by lowercase
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() >= 2 && chars[0].is_uppercase() && chars[1].is_lowercase() {
        // Exclusion logic for Python literals (True/False/None)
        let first_word = trimmed.split_whitespace().next().unwrap_or("");
        if !["True", "False", "None"].contains(&first_word) 
           && !first_word.chars().all(|c| c.is_alphanumeric()) {
            return true;
        }
    }

    // 2. Docstring markers (Explicit signals)
    if [":param", ":return", ":raises", ":type"].iter().any(|m| trimmed.starts_with(m)) {
        return true;
    }

    // 3. Common prose starters
    ["The ", "This ", "Note:", "Warning:", "Example:"].iter().any(|s| trimmed.starts_with(s))
}
```

### 5.2 Self-Contained Quality Tools (Respect for People)

**Requirement:** ALIM-R002
**Priority:** P0 (Immediate)

To respect the developer's time and focus, tools must be available "Just-in-Time" without context switching [7].

| Command | Status | Requirement |
|---------|--------|-------------|
| `alimentar quality check <file>` | **Refactor** | Implement profile-based scoring (Standard Work). |
| `alimentar quality report <file>` | **Enhance** | Add "Andon" style visual indicators (Colors/Symbols). |
| `alimentar stats <file>` | **New** | Compute distribution metrics in-process (No Pandas). |
| `alimentar histogram <file>` | **New** | ASCII visual control for distribution shape. |

### 5.3 Statistical Analysis (Fact-Based Decisions)

**Requirement:** ALIM-R003
**Priority:** P1

Decisions must be based on facts (data), not intuition. Implement rigorous statistical profiling [8].

```rust
pub struct ColumnStats {
    pub count: usize,
    pub null_count: usize,
    pub unique_count: usize,
    // Entropy measures "information content" - crucial for detecting low-quality features
    pub entropy: f64, 
    pub percentiles: [f64; 5], 
    pub is_constant: bool,
}
```

### 5.4 Quality Profiles as Standard Work

**Requirement:** ALIM-R004
**Priority:** P1

Define explicit standards for data quality. Without this, we cannot distinguish "abnormality" from "noise" [4].

```yaml
# profiles/doctest_standard.yaml
name: doctest
description: "Standard Work definition for Python Doctest Corpus"
columns:
  input:
    type: string
    pattern: "^>>>"      # Poka-yoke: Must start with prompt
    required: true
  expected:
    type: string
    max_length: 500      # Heuristic: Long output is usually noise
    prose_contamination_check: true
thresholds:
  max_duplicate_ratio: 0.00  # Goal: Zero Defects
```

### 5.5 Pre-Load Validation (Stop the Line)

**Requirement:** ALIM-R005
**Priority:** P1

Prevent defective parts (data) from entering the assembly line (corpus) [3].

```rust
impl DocTestCorpus {
    pub fn push_validated(&mut self, doctest: DocTest) -> Result<(), ValidationError> {
        // Jidoka: Stop immediately if data violates the Standard
        if self.is_likely_contaminated(&doctest.expected) {
            return Err(ValidationError::ProseContamination);
        }
        self.push(doctest);
        Ok(())
    }
}
```

### 5.6 Remediation (Countermeasures)

**Requirement:** ALIM-R006
**Priority:** P2

Tools must allow rapid application of countermeasures (fixes) to return to standard [9].

```bash
# Apply standard countermeasures
alimentar fix corpus.parquet --profile doctest --output corpus_clean.parquet
```

### 5.7 Visual Control (Andon)

**Requirement:** ALIM-R007
**Priority:** P2

Use `trueno` to generate immediate visual feedback.

```rust
// ASCII Histogram for immediate "at-a-glance" understanding
pub fn print_histogram(values: &[f64]) {
    // ... implementation showing distribution shape ...
}
```

---

## 6. Integration Architecture

The architecture reflects the **Pull System**: downstream consumers (`depyler`, `aprender`) pull valid data from the `alimentar` core.

```
┌─────────────────────────────────────────────────────────────────┐
│                         alimentar                                │
│               (Single Source of Truth)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  trueno  │        │ aprender │        │  depyler │
    │ (Metrics)│        │ (Format) │        │ (Train)  │
    └──────────┘        └──────────┘        └──────────┘
```

---

## 7. References & Annotations

### Peer-Reviewed & Foundational Sources

[1] **Poppendieck, M., & Poppendieck, T. (2003).** *Lean Software Development: An Agile Toolkit*. Addison-Wesley.
> Establishes the mapping of "Build Integrity In" (Jidoka) to software engineering, arguing that testing should be part of the development process, not a separate phase.

[2] **Liker, J. K. (2004).** *The Toyota Way: 14 Management Principles*. McGraw-Hill.
> Principle 12: "Go and see for yourself to thoroughly understand the situation (Genchi Genbutsu)." Supports the requirement for self-contained inspection tools.

[3] **Ohno, T. (1988).** *Toyota Production System: Beyond Large-Scale Production*. Productivity Press.
> The primary source for **Jidoka** (Autonomation). Machines must detect abnormalities and stop, preventing the generation of defective products (data).

[4] **Imai, M. (1986).** *Kaizen: The Key to Japan's Competitive Success*. McGraw-Hill.
> Defines **Standard Work** as the foundation for improvement. Our "Quality Profiles" act as this standard; without them, improvement is random, not systematic.

[5] **Liker, J. K., & Meier, D. (2006).** *The Toyota Way Fieldbook*. McGraw-Hill.
> Details **Visual Management** (Andon). Supports the requirement for `alimentar quality report` to provide immediate, clear signals (Red/Green) of data health.

[6] **Shingo, S. (1986).** *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press.
> The concept of **Poka-Yoke** (mistake-proofing). The doctest parser's heuristic checks are literal poka-yokes preventing "prose contamination" defects from being created.

[7] **Womack, J. P., & Jones, D. T. (1996).** *Lean Thinking*. Simon & Schuster.
> Identifies "Waiting" and "Motion" as major forms of **Waste (Muda)**. Self-contained tools eliminate the waste of context-switching and setup time.

[8] **Deming, W. E. (1986).** *Out of the Crisis*. MIT Press.
> "In God we trust; all others must bring data." Supports ALIM-R003 (Statistical Analysis) – quality must be managed via statistical process control, not guesswork.

[9] **Ishikawa, K. (1985).** *What Is Total Quality Control? The Japanese Way*. Prentice-Hall.
> Emphasizes **Root Cause Analysis**. The "Remediation" tools (ALIM-R006) allow developers to address the root causes found during inspection.

[10] **Wang, R. Y., & Strong, D. M. (1996).** "Beyond Accuracy: What Data Quality Means to Data Consumers". *Journal of Management Information Systems*.
> Defines data quality as "fitness for use". This supports our "Profile" approach—data quality is relative to the task (Doctest vs. CSV), not an absolute metric.

---

## 8. Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude | 2025-11-29 | Draft |
| Reviewer | Noah | 2025-11-29 | Approved |