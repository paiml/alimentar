//! alimentar CLI entry point

use std::process::ExitCode;

fn main() -> ExitCode {
    alimentar::cli::run()
}
