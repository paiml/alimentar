//! Instruction/response pair extraction from Python source.
//!
//! Converts Python function definitions (signature + docstring + body)
//! into instruction/response JSONL pairs suitable for fine-tuning.
//!
//! # Pipeline
//!
//! 1. Walk source directories for `.py` files
//! 2. Parse each file to extract functions with docstrings
//! 3. Generate instruction (signature + docstring) / response (body) pairs
//! 4. Filter by response length (min/max lines)
//! 5. Deduplicate by instruction text
//!
//! # References
//!
//! - Spec §12.0: Ground Truth Corpora (Tier 1)
//! - GH-7: `apr data prep` native instruct pair extraction

use std::collections::HashSet;
use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::Result;

/// A single instruction/response pair for fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructPair {
    /// Natural language instruction (from signature + docstring)
    pub instruction: String,
    /// Code response (function body)
    pub response: String,
    /// Source corpus identifier
    pub source: String,
    /// Module path
    pub module: String,
    /// Function name
    pub function: String,
}

/// Configuration for instruct pair extraction.
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    /// Minimum response lines (default: 3)
    pub min_lines: usize,
    /// Maximum response lines (default: 200)
    pub max_lines: usize,
    /// Deduplicate by instruction text
    pub deduplicate: bool,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            min_lines: 3,
            max_lines: 200,
            deduplicate: true,
        }
    }
}

/// Extractor that produces instruction/response pairs from Python source.
pub struct InstructExtractor {
    config: ExtractConfig,
    /// Regex for function/class definitions
    def_re: Regex,
    /// Regex for triple-quoted docstrings
    docstring_re: Regex,
}

impl InstructExtractor {
    /// Create a new extractor with the given config.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new(config: ExtractConfig) -> Self {
        Self {
            config,
            def_re: Regex::new(
                r"(?m)^([^\S\n]*)(def|class)\s+(\w+)(\([^)]*\)(?:\s*->\s*[^:\n]+)?)\s*:",
            )
            .expect("valid regex"),
            docstring_re: Regex::new(r#"(?s)^(\s*)"""(.*?)""""#)
                .expect("valid regex"),
        }
    }

    /// Extract instruct pairs from a Python source string.
    pub fn extract_from_source(
        &self,
        source: &str,
        corpus: &str,
        module: &str,
    ) -> Vec<InstructPair> {
        let lines: Vec<&str> = source.lines().collect();
        let mut pairs = Vec::new();

        for caps in self.def_re.captures_iter(source) {
            let indent_str = caps.get(1).map_or("", |m| m.as_str());
            let kind = caps.get(2).map_or("", |m| m.as_str());
            if kind == "class" {
                continue; // Only extract functions
            }

            let name = caps.get(3).map_or("", |m| m.as_str());
            let params = caps.get(4).map_or("", |m| m.as_str());
            let signature = format!("def {name}{params}");

            // Find line number of this def
            let def_start = caps.get(0).map_or(0, |m| m.start());
            let def_line = source[..def_start].matches('\n').count();
            let base_indent = indent_str.len();

            // Find docstring (next non-empty line after def should be """)
            let (docstring, body_start_line) =
                self.find_docstring(&lines, def_line + 1, base_indent);

            if docstring.is_empty() {
                continue; // Skip functions without docstrings
            }

            // Extract function body (after docstring)
            let body = self.extract_body(&lines, body_start_line, base_indent);

            let body_lines = body.lines().count();
            if body_lines < self.config.min_lines
                || body_lines > self.config.max_lines
            {
                continue;
            }

            // Build instruction from signature + docstring
            let instruction = format!(
                "Complete the following Python function:\n\n{signature}:\n    \"\"\"{docstring}\"\"\""
            );

            pairs.push(InstructPair {
                instruction,
                response: body,
                source: corpus.to_string(),
                module: module.to_string(),
                function: name.to_string(),
            });
        }

        pairs
    }

    /// Find the docstring following a def line.
    /// Returns (docstring_content, line_after_docstring).
    fn find_docstring(
        &self,
        lines: &[&str],
        start_line: usize,
        base_indent: usize,
    ) -> (String, usize) {
        // Look for """ in the lines after the def
        for i in start_line..lines.len().min(start_line + 3) {
            let trimmed = lines[i].trim();
            if trimmed.is_empty() {
                continue;
            }

            // Check for single-line docstring: """text"""
            if trimmed.starts_with("\"\"\"") && trimmed.ends_with("\"\"\"")
                && trimmed.len() > 6
            {
                let content = &trimmed[3..trimmed.len() - 3];
                return (content.trim().to_string(), i + 1);
            }

            // Check for multi-line docstring start
            if trimmed.starts_with("\"\"\"") {
                let mut doc_lines = Vec::new();
                let first_content = &trimmed[3..];
                if !first_content.trim().is_empty() {
                    doc_lines.push(first_content.trim().to_string());
                }

                for j in (i + 1)..lines.len() {
                    let line = lines[j];
                    if line.trim().ends_with("\"\"\"") {
                        let last = line.trim().strip_suffix("\"\"\"").unwrap_or("");
                        if !last.is_empty() {
                            doc_lines.push(last.to_string());
                        }
                        return (doc_lines.join("\n"), j + 1);
                    }
                    // Strip base indent + 4 (docstring indent)
                    let stripped = if line.len() > base_indent + 4 {
                        &line[base_indent + 4..]
                    } else {
                        line.trim()
                    };
                    doc_lines.push(stripped.to_string());
                }
            }

            // Not a docstring line — no docstring for this function
            break;
        }

        (String::new(), start_line)
    }

    /// Extract function body (lines after docstring, at proper indent).
    fn extract_body(
        &self,
        lines: &[&str],
        start_line: usize,
        base_indent: usize,
    ) -> String {
        let mut body_lines = Vec::new();
        let body_indent = base_indent + 4;

        for i in start_line..lines.len() {
            let line = lines[i];

            // Empty lines are included
            if line.trim().is_empty() {
                body_lines.push(String::new());
                continue;
            }

            // Check if we've dedented back to base level or less
            let line_indent = line.len() - line.trim_start().len();
            if line_indent <= base_indent && !line.trim().is_empty() {
                break;
            }

            // Strip base body indent
            let stripped = if line.len() > body_indent {
                &line[body_indent..]
            } else {
                line.trim()
            };
            body_lines.push(stripped.to_string());
        }

        // Trim trailing empty lines
        while body_lines.last().is_some_and(|l| l.is_empty()) {
            body_lines.pop();
        }

        body_lines.join("\n")
    }

    /// Extract instruct pairs from a directory of Python files.
    pub fn extract_from_directory(
        &self,
        dir: &Path,
        corpus: &str,
    ) -> Result<Vec<InstructPair>> {
        let mut all_pairs = Vec::new();
        let mut seen = HashSet::new();

        for entry in WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if !path.extension().is_some_and(|ext| ext == "py") {
                continue;
            }

            let source = std::fs::read_to_string(path).map_err(|e| {
                crate::Error::Io {
                    path: Some(path.to_path_buf()),
                    source: e,
                }
            })?;

            let module = path_to_module(dir, path);
            let pairs = self.extract_from_source(&source, corpus, &module);

            for pair in pairs {
                if self.config.deduplicate {
                    if seen.contains(&pair.instruction) {
                        continue;
                    }
                    seen.insert(pair.instruction.clone());
                }
                all_pairs.push(pair);
            }
        }

        Ok(all_pairs)
    }

    /// Write instruct pairs as JSONL to a file.
    pub fn write_jsonl(pairs: &[InstructPair], path: &Path) -> Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(path).map_err(|e| {
            crate::Error::Io {
                path: Some(path.to_path_buf()),
                source: e,
            }
        })?;
        let mut writer = std::io::BufWriter::new(file);
        for pair in pairs {
            let json = serde_json::to_string(pair).map_err(|e| {
                crate::Error::InvalidConfig {
                message: format!("JSON serialize error: {e}"),
            }
            })?;
            writeln!(writer, "{json}").map_err(|e| crate::Error::Io {
                path: Some(path.to_path_buf()),
                source: e,
            })?;
        }
        Ok(())
    }
}

/// Convert a file path to a Python module name.
fn path_to_module(base: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(base).unwrap_or(path);
    let stem = relative.with_extension("");
    stem.to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, ".")
        .trim_end_matches(".__init__")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_function() {
        let source = r#"
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number.

    Computes using iterative approach for O(n) time.
    """
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"#;
        let ext = InstructExtractor::new(ExtractConfig {
            min_lines: 1,
            ..ExtractConfig::default()
        });
        let pairs = ext.extract_from_source(source, "test", "math");

        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].instruction.contains("def fibonacci"));
        assert!(pairs[0].instruction.contains("Fibonacci number"));
        assert!(pairs[0].response.contains("a, b = b, a + b"));
        assert_eq!(pairs[0].function, "fibonacci");
    }

    #[test]
    fn test_skip_no_docstring() {
        let source = r#"
def no_doc(x):
    return x + 1
"#;
        let ext = InstructExtractor::new(ExtractConfig {
            min_lines: 1,
            ..ExtractConfig::default()
        });
        let pairs = ext.extract_from_source(source, "test", "mod");
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_skip_short_body() {
        let source = r#"
def tiny(x: int) -> int:
    """Return x."""
    return x
"#;
        let ext = InstructExtractor::new(ExtractConfig::default());
        let pairs = ext.extract_from_source(source, "test", "mod");
        // Body is 1 line, min_lines is 3 by default
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_skip_class() {
        let source = r#"
class MyClass(Base):
    """A class."""
    def __init__(self):
        pass
"#;
        let ext = InstructExtractor::new(ExtractConfig {
            min_lines: 1,
            ..ExtractConfig::default()
        });
        let pairs = ext.extract_from_source(source, "test", "mod");
        // Should extract __init__ but not class itself
        // __init__ has no docstring + 1-line body, so skipped
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_multiline_docstring() {
        let source = r#"
def merge_sort(arr: list) -> list:
    """Sort a list using merge sort algorithm.

    Time complexity: O(n log n)
    Space complexity: O(n)

    Args:
        arr: Input list to sort
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
"#;
        let ext = InstructExtractor::new(ExtractConfig {
            min_lines: 1,
            ..ExtractConfig::default()
        });
        let pairs = ext.extract_from_source(source, "depyler", "sorting");

        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].instruction.contains("merge sort"));
        assert!(pairs[0].response.contains("merge_sort(arr[:mid])"));
        assert_eq!(pairs[0].source, "depyler");
    }

    #[test]
    fn test_deduplication() {
        let source = r#"
def foo(x: int) -> int:
    """Do something."""
    a = x + 1
    b = a * 2
    return b
"#;
        let ext = InstructExtractor::new(ExtractConfig {
            min_lines: 1,
            deduplicate: true,
            ..ExtractConfig::default()
        });
        // Extract twice from same source
        let mut pairs = ext.extract_from_source(source, "test", "mod");
        let pairs2 = ext.extract_from_source(source, "test", "mod");

        let mut seen = HashSet::new();
        pairs.extend(pairs2);
        pairs.retain(|p| seen.insert(p.instruction.clone()));

        // Should deduplicate to 1
        assert_eq!(pairs.len(), 1);
    }
}
