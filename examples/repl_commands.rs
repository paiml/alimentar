//! REPL Command Parser Demo
//!
//! Demonstrates programmatic parsing of REPL commands.
//! Run: cargo run --example repl_commands --features repl

#[cfg(feature = "repl")]
fn main() {
    use alimentar::repl::{CommandParser, ReplCommand};

    println!("=== REPL Command Parser Demo ===\n");

    // Data loading commands
    let commands = [
        "load data/iris.csv",
        "datasets",
        "use iris",
        "info",
        "head 5",
        "head",
        "schema",
    ];

    println!("--- Data Commands ---");
    for cmd in &commands {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {:<20} => {:?}", cmd, parsed),
            Err(e) => println!("  {:<20} => Error: {}", cmd, e),
        }
    }

    // Quality commands
    println!("\n--- Quality Commands ---");
    let quality_cmds = [
        "quality check",
        "quality score",
        "quality score --suggest",
        "quality score --json --badge",
    ];
    for cmd in &quality_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {:<30} => {:?}", cmd, parsed),
            Err(e) => println!("  {:<30} => Error: {}", cmd, e),
        }
    }

    // Export and conversion
    println!("\n--- Export/Convert Commands ---");
    let export_cmds = [
        "convert csv",
        "convert parquet",
        "convert json",
        "export quality",
        "export quality --json",
    ];
    for cmd in &export_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {:<25} => {:?}", cmd, parsed),
            Err(e) => println!("  {:<25} => Error: {}", cmd, e),
        }
    }

    // Session commands
    println!("\n--- Session Commands ---");
    let session_cmds = [
        "history",
        "history --export",
        "help",
        "help quality",
        "?",
        "quit",
        "exit",
        "q",
    ];
    for cmd in &session_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {:<20} => {:?}", cmd, parsed),
            Err(e) => println!("  {:<20} => Error: {}", cmd, e),
        }
    }

    // Error cases
    println!("\n--- Error Handling ---");
    let error_cmds = ["load", "quality", "drift", "unknown", ""];
    for cmd in &error_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {:<15} => {:?}", cmd, parsed),
            Err(e) => println!("  {:<15} => Error: {}", cmd, e),
        }
    }

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "repl"))]
fn main() {
    eprintln!("Run with: cargo run --example repl_commands --features repl");
}
