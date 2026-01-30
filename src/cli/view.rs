//! TUI view commands for interactive dataset viewing.

use std::io::Write;
use std::path::PathBuf;

use crate::tui::{DatasetAdapter, DatasetViewer};

use super::basic::load_dataset;

/// Interactive TUI viewer for datasets.
pub(crate) fn cmd_view(path: &PathBuf, initial_search: Option<&str>) -> crate::Result<()> {
    use crossterm::{cursor, execute, terminal};
    use std::io::stdout;

    let dataset = load_dataset(path)?;
    let adapter = DatasetAdapter::from_dataset(&dataset)
        .map_err(|e| crate::Error::storage(format!("TUI adapter error: {e}")))?;

    // Get terminal size
    let (width, height) = terminal::size().unwrap_or((80, 24));
    let mut viewer = DatasetViewer::with_dimensions(adapter, width, height.saturating_sub(2));

    // Apply initial search if provided
    if let Some(query) = initial_search {
        viewer.search(query);
    }

    // Enter raw mode for keyboard input
    terminal::enable_raw_mode()
        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

    let mut stdout = stdout();

    // Hide cursor
    execute!(stdout, cursor::Hide)
        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

    let result = run_tui_loop(&mut viewer, &mut stdout, path);

    // Cleanup: restore terminal
    let _ = execute!(stdout, cursor::Show);
    let _ = terminal::disable_raw_mode();

    result
}

/// Run the TUI event loop.
pub(crate) fn run_tui_loop<W: Write>(
    viewer: &mut DatasetViewer,
    stdout: &mut W,
    path: &std::path::Path,
) -> crate::Result<()> {
    use crossterm::{
        cursor,
        event::{self, Event, KeyCode, KeyModifiers},
        execute,
        style::{Attribute, Print, SetAttribute},
        terminal::{self, Clear, ClearType},
    };

    loop {
        // Clear screen and move to top
        execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))
            .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

        // Render title bar
        let (width, _) = terminal::size().unwrap_or((80, 24));
        let title = format!(
            " {} | {} rows | {}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            viewer.row_count(),
            if viewer.adapter().is_streaming() {
                "Streaming"
            } else {
                "InMemory"
            }
        );
        execute!(
            stdout,
            SetAttribute(Attribute::Reverse),
            Print(format!("{:width$}", title, width = width as usize)),
            SetAttribute(Attribute::Reset),
            Print("\r\n")
        )
        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

        // Render the viewer
        for line in viewer.render_lines() {
            execute!(stdout, Print(&line), Print("\r\n"))
                .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;
        }

        // Render status bar
        let status = format!(
            " Row {}-{} of {} | {} scroll | PgUp/PgDn page | Home/End | /search | q quit ",
            viewer.scroll_offset() + 1,
            (viewer.scroll_offset() + viewer.visible_row_count() as usize).min(viewer.row_count()),
            viewer.row_count(),
            "\u{2191}\u{2193}" // up/down arrows
        );
        execute!(
            stdout,
            SetAttribute(Attribute::Reverse),
            Print(format!("{:width$}", status, width = width as usize)),
            SetAttribute(Attribute::Reset)
        )
        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

        stdout
            .flush()
            .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

        // Handle input
        if event::poll(std::time::Duration::from_millis(100))
            .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?
        {
            if let Event::Key(key) =
                event::read().map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?
            {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Down | KeyCode::Char('j') => viewer.scroll_down(),
                    KeyCode::Up | KeyCode::Char('k') => viewer.scroll_up(),
                    KeyCode::PageDown | KeyCode::Char(' ') => viewer.page_down(),
                    KeyCode::PageUp => viewer.page_up(),
                    KeyCode::Home | KeyCode::Char('g') => viewer.home(),
                    KeyCode::End | KeyCode::Char('G') => viewer.end(),
                    KeyCode::Char('/') => {
                        // Simple search prompt
                        if let Some(query) = prompt_search(stdout)? {
                            viewer.search(&query);
                        }
                    }
                    _ => {}
                }
            }

            // Handle resize
            if let Event::Resize(w, h) =
                event::read().map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?
            {
                viewer.set_dimensions(w, h.saturating_sub(2));
            }
        }
    }

    Ok(())
}

/// Prompt for search query.
pub(crate) fn prompt_search<W: Write>(stdout: &mut W) -> crate::Result<Option<String>> {
    use crossterm::{
        cursor,
        event::{self, Event, KeyCode},
        execute,
        style::Print,
        terminal::{self, Clear, ClearType},
    };

    let (_, height) = terminal::size().unwrap_or((80, 24));

    // Move to bottom and show prompt
    execute!(
        stdout,
        cursor::MoveTo(0, height - 1),
        Clear(ClearType::CurrentLine),
        cursor::Show,
        Print("Search: ")
    )
    .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;
    stdout
        .flush()
        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

    let mut query = String::new();

    loop {
        if let Event::Key(key) =
            event::read().map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?
        {
            match key.code {
                KeyCode::Enter => {
                    execute!(stdout, cursor::Hide)
                        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;
                    return Ok(if query.is_empty() { None } else { Some(query) });
                }
                KeyCode::Esc => {
                    execute!(stdout, cursor::Hide)
                        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;
                    return Ok(None);
                }
                KeyCode::Backspace => {
                    query.pop();
                    execute!(
                        stdout,
                        cursor::MoveTo(8, height - 1),
                        Clear(ClearType::UntilNewLine),
                        Print(&query)
                    )
                    .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;
                }
                KeyCode::Char(c) => {
                    query.push(c);
                    execute!(stdout, Print(c))
                        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;
                }
                _ => {}
            }
            stdout
                .flush()
                .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;
        }
    }
}
