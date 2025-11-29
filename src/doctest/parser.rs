//! Python doctest parser.
//!
//! Extracts doctests from Python source files using regex-based parsing.

use std::path::Path;

use regex::Regex;
use walkdir::WalkDir;

use super::{DocTest, DocTestCorpus};
use crate::Result;

/// Parser for extracting Python doctests from source files.
#[derive(Debug)]
pub struct DocTestParser {
    /// Regex for finding docstrings
    docstring_re: Regex,
    /// Regex for finding function/class definitions
    def_re: Regex,
}

impl Default for DocTestParser {
    fn default() -> Self {
        Self::new()
    }
}

impl DocTestParser {
    /// Create a new parser.
    ///
    /// # Panics
    /// Panics if regex compilation fails (should never happen with valid
    /// patterns).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new() -> Self {
        Self {
            // Match triple-quoted strings (""" only - most common for docstrings)
            // Can't use backreferences in Rust regex, so we match """ ... """ directly
            docstring_re: Regex::new(r#"(?s)"""(.*?)""""#).expect("valid regex"),
            // Match def/class definitions to get function names
            def_re: Regex::new(r"(?m)^(\s*)(def|class)\s+(\w+)").expect("valid regex"),
        }
    }

    /// Extract doctests from a Python source string.
    ///
    /// # Arguments
    /// * `source` - Python source code
    /// * `module` - Module name for the extracted doctests
    #[must_use]
    pub fn parse_source(&self, source: &str, module: &str) -> Vec<DocTest> {
        let mut results = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        // Build a map of line number -> (indent, name, is_class)
        let mut context_map: Vec<Option<(usize, String, bool)>> = vec![None; lines.len()];
        let mut context_stack: Vec<(usize, String, bool)> = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            if let Some(caps) = self.def_re.captures(line) {
                let indent = caps.get(1).map_or(0, |m| m.as_str().len());
                let kind = caps.get(2).map_or("", |m| m.as_str());
                let name = caps.get(3).map_or("", |m| m.as_str()).to_string();
                let is_class = kind == "class";

                // Pop contexts with same or greater indent
                while context_stack
                    .last()
                    .is_some_and(|(ctx_indent, _, _)| *ctx_indent >= indent)
                {
                    context_stack.pop();
                }

                context_stack.push((indent, name, is_class));
            }

            // Store current context for this line
            context_map[i] = context_stack.last().cloned();
        }

        // Find all docstrings and extract doctests
        for caps in self.docstring_re.captures_iter(source) {
            let Some(docstring_match) = caps.get(0) else {
                continue;
            };
            let content = caps.get(1).map_or("", |m| m.as_str());

            // Find which line this docstring starts on
            let start_byte = docstring_match.start();
            let line_num = source[..start_byte].matches('\n').count();

            // Determine the function name from context
            let function_name = Self::get_function_name(line_num, &context_map, &lines);

            // Extract doctests from this docstring
            let doctests = Self::extract_from_docstring(content, module, &function_name);
            results.extend(doctests);
        }

        results
    }

    /// Get the function name for a docstring at the given line.
    fn get_function_name(
        line_num: usize,
        context_map: &[Option<(usize, String, bool)>],
        lines: &[&str],
    ) -> String {
        // Check if there's a def/class on this line or the previous line
        if line_num < lines.len() {
            // Look at current context
            if let Some((_, ref name, is_class)) = context_map.get(line_num).and_then(|c| c.clone())
            {
                // Check if we're inside a class method
                if !is_class {
                    // It's a function - check if it's inside a class
                    for i in (0..line_num).rev() {
                        if let Some((_, ref class_name, true)) =
                            context_map.get(i).and_then(|c| c.clone())
                        {
                            return format!("{class_name}.{name}");
                        }
                    }
                }
                return name.clone();
            }
        }

        // Module-level docstring
        "__module__".to_string()
    }

    /// Extract doctests from a docstring's content.
    fn extract_from_docstring(content: &str, module: &str, function: &str) -> Vec<DocTest> {
        let mut results = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Look for >>> prompt
            if let Some(input_start) = line.strip_prefix(">>>") {
                let mut input_lines = vec![format!(">>>{}", input_start)];
                i += 1;

                // Collect continuation lines (...)
                while i < lines.len() {
                    let next_line = lines[i].trim();
                    if let Some(cont) = next_line.strip_prefix("...") {
                        input_lines.push(format!("...{}", cont));
                        i += 1;
                    } else {
                        break;
                    }
                }

                // Collect expected output (until next >>> or blank line pattern)
                let mut expected_lines: Vec<&str> = Vec::new();
                // Find the base indentation of this doctest block
                let base_indent = lines
                    .get(i.saturating_sub(1))
                    .map(|l| l.len() - l.trim_start().len())
                    .unwrap_or(0);

                while i < lines.len() {
                    let next_line = lines[i];
                    let trimmed = next_line.trim();

                    // Stop at next prompt or end of doctest section
                    if trimmed.starts_with(">>>") {
                        break;
                    }

                    // Poka-Yoke (ALIM-R001): Stop at prose continuation to prevent contamination
                    if is_prose_continuation(trimmed) {
                        break;
                    }

                    // Empty line might end the expected output, but tracebacks can have blank lines
                    if trimmed.is_empty() && !expected_lines.is_empty() {
                        // Peek ahead - if next non-empty line is >>> or end, we're done
                        let mut j = i + 1;
                        while j < lines.len() && lines[j].trim().is_empty() {
                            j += 1;
                        }
                        if j >= lines.len() || lines[j].trim().starts_with(">>>") {
                            break;
                        }
                        // Otherwise include the blank line (might be part of
                        // multiline output)
                    }

                    if !trimmed.is_empty() || !expected_lines.is_empty() {
                        // Strip base indentation but preserve relative indentation
                        let stripped = if next_line.len() > base_indent {
                            &next_line
                                [base_indent.min(next_line.len() - next_line.trim_start().len())..]
                        } else {
                            trimmed
                        };
                        expected_lines.push(stripped.trim_end());
                    }
                    i += 1;
                }

                // Trim trailing empty lines from expected
                while expected_lines.last().is_some_and(|l| l.is_empty()) {
                    expected_lines.pop();
                }

                let input = input_lines.join("\n");
                let expected = expected_lines.join("\n");

                results.push(DocTest::new(module, function, input, expected));
            } else {
                i += 1;
            }
        }

        results
    }

    /// Extract doctests from a Python file.
    pub fn parse_file(&self, path: &Path, module: &str) -> Result<Vec<DocTest>> {
        let source = std::fs::read_to_string(path).map_err(|e| crate::Error::Io {
            path: Some(path.to_path_buf()),
            source: e,
        })?;
        Ok(self.parse_source(&source, module))
    }

    /// Extract doctests from a directory of Python files.
    ///
    /// # Arguments
    /// * `dir` - Directory to scan for .py files
    /// * `source` - Source identifier (e.g., "cpython")
    /// * `version` - Version string or git SHA
    pub fn parse_directory(
        &self,
        dir: &Path,
        source: &str,
        version: &str,
    ) -> Result<DocTestCorpus> {
        let mut corpus = DocTestCorpus::new(source, version);

        for entry in WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "py") {
                let module = path_to_module(dir, path);
                let doctests = self.parse_file(path, &module)?;
                for dt in doctests {
                    corpus.push(dt);
                }
            }
        }

        Ok(corpus)
    }
}

/// Convert a file path to a Python module name.
///
/// Example: `/lib/python/os/path.py` with base `/lib/python` -> `os.path`
fn path_to_module(base: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(base).unwrap_or(path);
    let stem = relative.with_extension("");
    stem.to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, ".")
        .trim_end_matches(".__init__")
        .to_string()
}

/// Detect if a line is prose continuation (documentation text) rather than code output.
/// This is a Poka-Yoke to prevent prose contamination in expected output.
///
/// # Arguments
/// * `line` - The line to check
///
/// # Returns
/// `true` if the line appears to be prose/documentation, `false` if it looks like code output
#[must_use]
pub fn is_prose_continuation(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Python exception lines: ErrorName: message (valid output, not prose)
    // e.g., "ZeroDivisionError: division by zero", "UserWarning: deprecation"
    // Must be PascalCase (not just "Warning" which is a prose indicator)
    let first_word: &str = trimmed.split(|c: char| c == ':' || c.is_whitespace())
        .next()
        .unwrap_or("");
    let is_python_exception = (first_word.ends_with("Error")
        || first_word.ends_with("Exception")
        || first_word.ends_with("Warning"))
        && first_word.len() > 7  // Longer than just "Warning" or "Error"
        && first_word.chars().filter(|c| c.is_uppercase()).count() >= 2;
    if is_python_exception {
        return false;
    }

    // 1. Sentence pattern: Capital letter followed by lowercase (prose indicator)
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() >= 2 && chars[0].is_uppercase() && chars[1].is_lowercase() {
        // Python literals and error keywords that start with capital
        if ["True", "False", "None", "Traceback"].contains(&first_word) {
            return false;
        }
        // Single capitalized word might be a class name in output
        if first_word.chars().all(|c| c.is_alphanumeric() || c == '_')
           && trimmed.split_whitespace().count() == 1 {
            return false;
        }
        // Multi-word sentence starting with capital = prose
        // But only if it doesn't look like Python output
        if trimmed.split_whitespace().count() > 2 {
            // Check if it looks like code output (contains Python-like tokens)
            // Note: "..." at end of sentence is ellipsis (prose), not continuation marker
            if trimmed.contains(">>>") || trimmed.starts_with("...")
               || trimmed.starts_with('<') || trimmed.starts_with('[')
               || trimmed.starts_with('{') || trimmed.starts_with('(') {
                return false;
            }
            return true;
        }
    }

    // 2. Docstring markers (explicit signals)
    let docstring_markers = [":param", ":return", ":raises", ":type", ":rtype",
                            ":arg", ":args:", ":keyword", ":ivar", ":cvar"];
    if docstring_markers.iter().any(|m| trimmed.starts_with(m)) {
        return true;
    }

    // 3. Common prose starters (must not contain code-like content)
    let prose_starters = [
        "The ", "This ", "Note:", "Warning:", "Example:", "Examples:",
        "See ", "If ", "When ", "For ", "An ", "A ", "It ",
        "Returns ", "Raises ", "Args:", "Arguments:", "Parameters:",
        "By ", "Use ", "Set ", "Get ", "You ", "We ", "They ",
    ];
    if prose_starters.iter().any(|s| trimmed.starts_with(s)) {
        // Don't filter if it contains code-like content (e.g., "Type >>> to continue")
        if trimmed.contains(">>>") || trimmed.contains("...") {
            return false;
        }
        return true;
    }

    // 4. Sphinx/reStructuredText markers
    if trimmed.starts_with(".. ") || trimmed.starts_with(">>>") {
        return false; // These are code-related
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== ALIM-R001: Prose Detection Tests (Poka-Yoke) ==========

    #[test]
    fn test_prose_detection_sentence() {
        // Prose sentences should be detected
        assert!(is_prose_continuation("The stdout argument is not allowed."));
        assert!(is_prose_continuation("This function returns a value."));
        assert!(is_prose_continuation("Note: This is important."));
        assert!(is_prose_continuation("Warning: Use with caution."));
    }

    #[test]
    fn test_prose_detection_docstring_markers() {
        // Docstring markers should be detected as prose
        assert!(is_prose_continuation(":param x: the input value"));
        assert!(is_prose_continuation(":return: the computed result"));
        assert!(is_prose_continuation(":raises ValueError: if invalid"));
        assert!(is_prose_continuation(":type x: int"));
    }

    #[test]
    fn test_prose_detection_common_starters() {
        // Common documentation starters
        assert!(is_prose_continuation("If you use this argument..."));
        assert!(is_prose_continuation("When the value is negative..."));
        assert!(is_prose_continuation("For more information..."));
        assert!(is_prose_continuation("Returns the computed value."));
    }

    #[test]
    fn test_prose_detection_false_negatives() {
        // Code output should NOT be detected as prose
        assert!(!is_prose_continuation("True"));
        assert!(!is_prose_continuation("False"));
        assert!(!is_prose_continuation("None"));
        assert!(!is_prose_continuation("123"));
        assert!(!is_prose_continuation("'hello world'"));
        assert!(!is_prose_continuation("b'bytes'"));
        assert!(!is_prose_continuation("[1, 2, 3]"));
        assert!(!is_prose_continuation("{'key': 'value'}"));
        assert!(!is_prose_continuation("(0, '/bin/ls')"));
        assert!(!is_prose_continuation("Point(x=11, y=22)"));
        assert!(!is_prose_continuation(""));
    }

    #[test]
    fn test_prose_detection_edge_cases() {
        // Class names (single capitalized word)
        assert!(!is_prose_continuation("ValueError"));
        assert!(!is_prose_continuation("MyClass"));
        // Traceback lines
        assert!(!is_prose_continuation("Traceback (most recent call last):"));
        // Empty and whitespace
        assert!(!is_prose_continuation(""));
        assert!(!is_prose_continuation("   "));
    }

    #[test]
    fn test_extract_with_prose_contamination() {
        // This is the critical test - prose after expected should be excluded
        let parser = DocTestParser::new();
        let source = r#"
def check_output():
    """
    >>> check_output(["ls", "-l"])
    b'output\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error, use stderr=STDOUT.

    >>> check_output(["echo", "hi"])
    b'hi\n'
    """
    pass
"#;
        let doctests = parser.parse_source(source, "test");
        assert_eq!(doctests.len(), 2);
        // First doctest should NOT include the prose
        assert_eq!(doctests[0].expected, "b'output\\n'");
        assert!(!doctests[0].expected.contains("stdout argument"));
        // Second doctest should be clean
        assert_eq!(doctests[1].expected, "b'hi\\n'");
    }

    // ========== Original Tests ==========

    #[test]
    fn test_path_to_module_simple() {
        let base = Path::new("/lib");
        let path = Path::new("/lib/os.py");
        assert_eq!(path_to_module(base, path), "os");
    }

    #[test]
    fn test_path_to_module_nested() {
        let base = Path::new("/lib");
        let path = Path::new("/lib/os/path.py");
        assert_eq!(path_to_module(base, path), "os.path");
    }

    #[test]
    fn test_path_to_module_init() {
        let base = Path::new("/lib");
        let path = Path::new("/lib/collections/__init__.py");
        assert_eq!(path_to_module(base, path), "collections");
    }

    #[test]
    fn test_extract_simple() {
        let parser = DocTestParser::new();
        let source = r#"
def foo():
    """
    >>> 1 + 1
    2
    """
    pass
"#;
        let doctests = parser.parse_source(source, "test");
        assert_eq!(doctests.len(), 1);
        assert_eq!(doctests[0].input, ">>> 1 + 1");
        assert_eq!(doctests[0].expected, "2");
    }

    #[test]
    fn test_extract_multiline_input() {
        let parser = DocTestParser::new();
        let source = r#"
def foo():
    """
    >>> x = (
    ...     1 + 2
    ... )
    >>> x
    3
    """
    pass
"#;
        let doctests = parser.parse_source(source, "test");
        assert_eq!(doctests.len(), 2);
        assert_eq!(doctests[0].input, ">>> x = (\n...     1 + 2\n... )");
        assert_eq!(doctests[0].expected, "");
    }

    // ========== Property Tests (50 cases for fast CI) ==========

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_empty_never_prose(s in "\\s*") {
            // Empty/whitespace-only is never prose
            assert!(!is_prose_continuation(&s));
        }

        #[test]
        fn prop_python_literals_never_prose(literal in prop_oneof![
            Just("True"),
            Just("False"),
            Just("None"),
        ]) {
            assert!(!is_prose_continuation(literal));
        }

        #[test]
        fn prop_exception_lines_never_prose(exc in prop_oneof![
            Just("ValueError: invalid input"),
            Just("TypeError: expected str"),
            Just("ZeroDivisionError: division by zero"),
            Just("KeyError: 'missing'"),
            Just("IndexError: out of range"),
            Just("RuntimeError: something went wrong"),
        ]) {
            assert!(!is_prose_continuation(exc), "Exception detected as prose: {}", exc);
        }

        #[test]
        fn prop_docstring_markers_are_prose(marker in prop_oneof![
            Just(":param x: value"),
            Just(":return: result"),
            Just(":raises ValueError: msg"),
            Just(":type x: int"),
        ]) {
            assert!(is_prose_continuation(marker));
        }

        #[test]
        fn prop_code_output_preserved(output in prop_oneof![
            Just("[1, 2, 3]"),
            Just("{'a': 1}"),
            Just("(1, 2)"),
            Just("<object at 0x...>"),
            Just("123"),
            Just("'string'"),
        ]) {
            assert!(!is_prose_continuation(output));
        }

        #[test]
        fn prop_deterministic(s in ".*") {
            // Same input always gives same output (determinism)
            let r1 = is_prose_continuation(&s);
            let r2 = is_prose_continuation(&s);
            assert_eq!(r1, r2);
        }
    }
}
