//! alimentar CLI - Data Loading, Distribution and Tooling
//!
//! Command-line interface for alimentar operations.


use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use crate::{
    backend::LocalBackend,
    drift::{DriftDetector, DriftSeverity, DriftTest},
    federated::{
        FederatedSplitCoordinator, FederatedSplitStrategy, NodeSplitInstruction, NodeSplitManifest,
    },
    quality::QualityChecker,
    registry::{DatasetMetadata, Registry},
    sketch::{DataSketch, DistributedDriftDetector, SketchType},
    split::DatasetSplit,
    ArrowDataset, Dataset,
};
use arrow::util::pretty::print_batches;
use clap::{Parser, Subcommand};

/// alimentar - Data Loading, Distribution and Tooling in Pure Rust
#[derive(Parser)]
#[command(name = "alimentar")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert between data formats
    Convert {
        /// Input file path
        input: PathBuf,
        /// Output file path
        output: PathBuf,
    },
    /// Display dataset information
    Info {
        /// Path to dataset file
        path: PathBuf,
    },
    /// Display first N rows of a dataset
    Head {
        /// Path to dataset file
        path: PathBuf,
        /// Number of rows to display
        #[arg(short = 'n', long, default_value = "10")]
        rows: usize,
    },
    /// Display dataset schema
    Schema {
        /// Path to dataset file
        path: PathBuf,
    },
    /// Import dataset from HuggingFace Hub.
    #[allow(clippy::doc_markdown)]
    #[cfg(feature = "hf-hub")]
    Import {
        #[command(subcommand)]
        source: ImportSource,
    },
    /// HuggingFace Hub commands (push/upload datasets)
    #[allow(clippy::doc_markdown)]
    #[cfg(feature = "hf-hub")]
    #[command(subcommand)]
    Hub(HubCommands),
    /// Registry commands for dataset sharing and discovery
    #[command(subcommand)]
    Registry(RegistryCommands),
    /// Data drift detection commands
    #[command(subcommand)]
    Drift(DriftCommands),
    /// Data quality checking commands
    #[command(subcommand)]
    Quality(QualityCommands),
    /// Federated split coordination commands
    #[command(subcommand)]
    Fed(FedCommands),
    /// Python doctest extraction commands
    #[cfg(feature = "doctest")]
    #[command(subcommand)]
    Doctest(DoctestCommands),
    /// Interactive REPL for data exploration
    #[cfg(feature = "repl")]
    Repl,
}

/// Import source options
#[cfg(feature = "hf-hub")]
#[derive(Subcommand)]
enum ImportSource {
    /// Import from HuggingFace Hub.
    #[allow(clippy::doc_markdown)]
    Hf {
        /// Dataset repository ID (e.g., "squad", "openai/gsm8k")
        repo_id: String,
        /// Output path for the downloaded dataset
        #[arg(short, long)]
        output: PathBuf,
        /// Git revision (branch, tag, or commit)
        #[arg(short, long, default_value = "main")]
        revision: String,
        /// Dataset subset/configuration
        #[arg(short, long)]
        subset: Option<String>,
        /// Data split (train, validation, test)
        #[arg(long, default_value = "train")]
        split: String,
    },
}

/// HuggingFace Hub commands
#[cfg(feature = "hf-hub")]
#[derive(Subcommand)]
enum HubCommands {
    /// Push (upload) a dataset to HuggingFace Hub
    #[allow(clippy::doc_markdown)]
    Push {
        /// Path to the parquet file to upload
        input: PathBuf,
        /// HuggingFace repository ID (e.g., "paiml/my-dataset")
        repo_id: String,
        /// Path in the repository (e.g., "data/train.parquet")
        #[arg(short, long)]
        path_in_repo: Option<String>,
        /// Commit message for the upload
        #[arg(short, long, default_value = "Upload via alimentar")]
        message: String,
        /// Path to README.md to upload as dataset card
        #[arg(long)]
        readme: Option<PathBuf>,
        /// Make the dataset private
        #[arg(long)]
        private: bool,
    },
}

#[derive(Subcommand)]
enum RegistryCommands {
    /// Initialize a new registry
    Init {
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// List all datasets in a registry
    List {
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// Push (publish) a dataset to the registry
    Push {
        /// Path to the dataset file (parquet)
        input: PathBuf,
        /// Dataset name in the registry
        #[arg(short, long)]
        name: String,
        /// Dataset version (semver)
        #[arg(short, long, default_value = "1.0.0")]
        version: String,
        /// Description of the dataset
        #[arg(short, long, default_value = "")]
        description: String,
        /// License identifier (e.g., MIT, Apache-2.0)
        #[arg(short, long, default_value = "")]
        license: String,
        /// Tags for the dataset (comma-separated)
        #[arg(short, long, default_value = "")]
        tags: String,
        /// Path to registry directory
        #[arg(long, default_value = ".alimentar")]
        registry: PathBuf,
    },
    /// Pull (download) a dataset from the registry
    Pull {
        /// Dataset name
        name: String,
        /// Output path for the dataset
        #[arg(short, long)]
        output: PathBuf,
        /// Specific version to pull (defaults to latest)
        #[arg(short, long)]
        version: Option<String>,
        /// Path to registry directory
        #[arg(long, default_value = ".alimentar")]
        registry: PathBuf,
    },
    /// Search datasets by name or description
    Search {
        /// Search query
        query: String,
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// Show detailed info about a specific dataset
    ShowInfo {
        /// Dataset name
        name: String,
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// Delete a dataset version from the registry
    Delete {
        /// Dataset name
        name: String,
        /// Version to delete
        #[arg(short, long)]
        version: String,
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
}

/// Drift detection commands
#[derive(Subcommand)]
enum DriftCommands {
    /// Detect drift between reference and current datasets
    Detect {
        /// Reference (baseline) dataset
        #[arg(short, long)]
        reference: PathBuf,
        /// Current dataset to compare
        #[arg(short, long)]
        current: PathBuf,
        /// Statistical tests to use (ks, chi2, psi, js)
        #[arg(short, long, default_value = "ks,psi")]
        tests: String,
        /// Significance threshold (alpha)
        #[arg(short, long, default_value = "0.05")]
        alpha: f64,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Generate a drift report summary
    Report {
        /// Reference (baseline) dataset
        #[arg(short, long)]
        reference: PathBuf,
        /// Current dataset to compare
        #[arg(short, long)]
        current: PathBuf,
        /// Output file for the report (JSON format)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Create a sketch from a dataset for distributed drift detection
    Sketch {
        /// Input dataset file
        input: PathBuf,
        /// Output sketch file
        #[arg(short, long)]
        output: PathBuf,
        /// Sketch algorithm type (tdigest, ddsketch)
        #[arg(short = 't', long, default_value = "tdigest")]
        sketch_type: String,
        /// Source identifier (e.g., node name)
        #[arg(short, long)]
        source: Option<String>,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Merge multiple sketches into one
    Merge {
        /// Sketch files to merge
        #[arg(required = true)]
        sketches: Vec<PathBuf>,
        /// Output merged sketch file
        #[arg(short, long)]
        output: PathBuf,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Compare two sketches for drift
    Compare {
        /// Reference (baseline) sketch
        #[arg(short, long)]
        reference: PathBuf,
        /// Current sketch to compare
        #[arg(short, long)]
        current: PathBuf,
        /// Drift detection threshold
        #[arg(short = 't', long, default_value = "0.1")]
        threshold: f64,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
}

/// Quality checking commands
#[derive(Subcommand)]
enum QualityCommands {
    /// Check data quality of a dataset
    Check {
        /// Path to dataset file
        path: PathBuf,
        /// Null ratio threshold (0.0 to 1.0)
        #[arg(long, default_value = "0.1")]
        null_threshold: f64,
        /// Duplicate ratio threshold (0.0 to 1.0)
        #[arg(long, default_value = "0.05")]
        duplicate_threshold: f64,
        /// Enable outlier detection
        #[arg(long, default_value = "true")]
        detect_outliers: bool,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Generate a quality report
    Report {
        /// Path to dataset file
        path: PathBuf,
        /// Output file for the report (JSON format)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Calculate 100-point quality score with letter grade (GH-6)
    Score {
        /// Path to dataset file
        path: PathBuf,
        /// Quality profile to use (default, doctest-corpus, ml-training, time-series)
        #[arg(short, long, default_value = "default")]
        profile: String,
        /// Show improvement suggestions for failed checks
        #[arg(long)]
        suggest: bool,
        /// Output as JSON for CI/CD integration
        #[arg(long)]
        json: bool,
        /// Output badge URL for shields.io
        #[arg(long)]
        badge: bool,
    },
    /// List available quality profiles
    Profiles,
}

/// Federated split coordination commands
#[derive(Subcommand)]
enum FedCommands {
    /// Generate a manifest from local dataset (runs on each node)
    Manifest {
        /// Input dataset file
        input: PathBuf,
        /// Output manifest file
        #[arg(short, long)]
        output: PathBuf,
        /// Unique node identifier
        #[arg(short, long)]
        node_id: String,
        /// Training set ratio
        #[arg(short = 'r', long, default_value = "0.8")]
        train_ratio: f64,
        /// Random seed for reproducibility
        #[arg(short, long, default_value = "42")]
        seed: u64,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Create a split plan from manifests (runs on coordinator)
    Plan {
        /// Manifest files from all nodes
        #[arg(required = true)]
        manifests: Vec<PathBuf>,
        /// Output plan file
        #[arg(short, long)]
        output: PathBuf,
        /// Split strategy (local, proportional, stratified)
        #[arg(short, long, default_value = "local")]
        strategy: String,
        /// Training set ratio
        #[arg(short = 'r', long, default_value = "0.8")]
        train_ratio: f64,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Column for stratification (required for stratified strategy)
        #[arg(long)]
        stratify_column: Option<String>,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Execute local split based on plan (runs on each node)
    Split {
        /// Input dataset file
        input: PathBuf,
        /// Split plan file
        #[arg(short, long)]
        plan: PathBuf,
        /// This node's ID
        #[arg(short, long)]
        node_id: String,
        /// Output training set file
        #[arg(long)]
        train_output: PathBuf,
        /// Output test set file
        #[arg(long)]
        test_output: PathBuf,
        /// Output validation set file (optional)
        #[arg(long)]
        validation_output: Option<PathBuf>,
    },
    /// Verify global split quality from manifests (runs on coordinator)
    Verify {
        /// Manifest files from all nodes
        #[arg(required = true)]
        manifests: Vec<PathBuf>,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
}

/// Python doctest extraction commands
#[cfg(feature = "doctest")]
#[derive(Subcommand)]
enum DoctestCommands {
    /// Extract doctests from Python source files
    Extract {
        /// Input directory containing Python source files
        input: PathBuf,
        /// Output parquet file
        #[arg(short, long)]
        output: PathBuf,
        /// Source identifier (e.g., "cpython", "numpy")
        #[arg(short, long, default_value = "unknown")]
        source: String,
        /// Version string or git SHA
        #[arg(short, long, default_value = "unknown")]
        version: String,
    },
    /// Merge multiple doctest corpora into one
    Merge {
        /// Input parquet files to merge
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output parquet file
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[allow(clippy::too_many_lines)]
/// Run the alimentar CLI.
///
pub fn run() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Convert { input, output } => cmd_convert(&input, &output),
        Commands::Info { path } => cmd_info(&path),
        Commands::Head { path, rows } => cmd_head(&path, rows),
        Commands::Schema { path } => cmd_schema(&path),
        #[cfg(feature = "hf-hub")]
        Commands::Import { source } => match source {
            ImportSource::Hf {
                repo_id,
                output,
                revision,
                subset,
                split,
            } => cmd_import_hf(&repo_id, &output, &revision, subset.as_deref(), &split),
        },
        #[cfg(feature = "hf-hub")]
        Commands::Hub(hub_cmd) => match hub_cmd {
            HubCommands::Push {
                input,
                repo_id,
                path_in_repo,
                message,
                readme,
                private,
            } => cmd_hub_push(&input, &repo_id, path_in_repo.as_deref(), &message, readme.as_ref(), private),
        },
        Commands::Registry(registry_cmd) => match registry_cmd {
            RegistryCommands::Init { path } => cmd_registry_init(&path),
            RegistryCommands::List { path } => cmd_registry_list(&path),
            RegistryCommands::Push {
                input,
                name,
                version,
                description,
                license,
                tags,
                registry,
            } => cmd_registry_push(
                &input,
                &name,
                &version,
                &description,
                &license,
                &tags,
                &registry,
            ),
            RegistryCommands::Pull {
                name,
                output,
                version,
                registry,
            } => cmd_registry_pull(&name, &output, version.as_deref(), &registry),
            RegistryCommands::Search { query, path } => cmd_registry_search(&query, &path),
            RegistryCommands::ShowInfo { name, path } => cmd_registry_show_info(&name, &path),
            RegistryCommands::Delete {
                name,
                version,
                path,
            } => cmd_registry_delete(&name, &version, &path),
        },
        Commands::Drift(drift_cmd) => match drift_cmd {
            DriftCommands::Detect {
                reference,
                current,
                tests,
                alpha,
                format,
            } => cmd_drift_detect(&reference, &current, &tests, alpha, &format),
            DriftCommands::Report {
                reference,
                current,
                output,
            } => cmd_drift_report(&reference, &current, output.as_ref()),
            DriftCommands::Sketch {
                input,
                output,
                sketch_type,
                source,
                format,
            } => cmd_drift_sketch(&input, &output, &sketch_type, source.as_deref(), &format),
            DriftCommands::Merge {
                sketches,
                output,
                format,
            } => cmd_drift_merge(&sketches, &output, &format),
            DriftCommands::Compare {
                reference,
                current,
                threshold,
                format,
            } => cmd_drift_compare(&reference, &current, threshold, &format),
        },
        Commands::Quality(quality_cmd) => match quality_cmd {
            QualityCommands::Check {
                path,
                null_threshold,
                duplicate_threshold,
                detect_outliers,
                format,
            } => cmd_quality_check(
                &path,
                null_threshold,
                duplicate_threshold,
                detect_outliers,
                &format,
            ),
            QualityCommands::Report { path, output } => cmd_quality_report(&path, output.as_ref()),
            QualityCommands::Score {
                path,
                profile,
                suggest,
                json,
                badge,
            } => cmd_quality_score(&path, &profile, suggest, json, badge),
            QualityCommands::Profiles => cmd_quality_profiles(),
        },
        Commands::Fed(fed_cmd) => match fed_cmd {
            FedCommands::Manifest {
                input,
                output,
                node_id,
                train_ratio,
                seed,
                format,
            } => cmd_fed_manifest(&input, &output, &node_id, train_ratio, seed, &format),
            FedCommands::Plan {
                manifests,
                output,
                strategy,
                train_ratio,
                seed,
                stratify_column,
                format,
            } => cmd_fed_plan(
                &manifests,
                &output,
                &strategy,
                train_ratio,
                seed,
                stratify_column.as_deref(),
                &format,
            ),
            FedCommands::Split {
                input,
                plan,
                node_id,
                train_output,
                test_output,
                validation_output,
            } => cmd_fed_split(
                &input,
                &plan,
                &node_id,
                &train_output,
                &test_output,
                validation_output.as_ref(),
            ),
            FedCommands::Verify { manifests, format } => cmd_fed_verify(&manifests, &format),
        },
        #[cfg(feature = "doctest")]
        Commands::Doctest(doctest_cmd) => match doctest_cmd {
            DoctestCommands::Extract {
                input,
                output,
                source,
                version,
            } => cmd_doctest_extract(&input, &output, &source, &version),
            DoctestCommands::Merge { inputs, output } => cmd_doctest_merge(&inputs, &output),
        },
        #[cfg(feature = "repl")]
        Commands::Repl => crate::repl::run(),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_convert(input: &PathBuf, output: &PathBuf) -> crate::Result<()> {
    // Load input (supports parquet, csv)
    let dataset = load_dataset(input)?;

    // Save output (supports parquet, csv)
    save_dataset(&dataset, output)?;

    println!(
        "Converted {} -> {} ({} rows)",
        input.display(),
        output.display(),
        dataset.len()
    );

    Ok(())
}

fn cmd_info(path: &PathBuf) -> crate::Result<()> {
    let dataset = load_dataset(path)?;

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    println!("File: {}", path.display());
    println!("Format: {}", get_format(path));
    println!("Rows: {}", dataset.len());
    println!("Batches: {}", dataset.num_batches());
    println!("Columns: {}", dataset.schema().fields().len());
    println!("Size: {} bytes", file_size);

    Ok(())
}

fn cmd_head(path: &PathBuf, rows: usize) -> crate::Result<()> {
    let dataset = load_dataset(path)?;

    if dataset.is_empty() {
        println!("Dataset is empty");
        return Ok(());
    }

    // Collect rows into batches
    let mut collected = Vec::new();
    let mut count = 0;

    for batch in dataset.iter() {
        let take = (rows - count).min(batch.num_rows());
        if take > 0 {
            collected.push(batch.slice(0, take));
            count += take;
        }
        if count >= rows {
            break;
        }
    }

    if collected.is_empty() {
        println!("No data to display");
        return Ok(());
    }

    // Print using Arrow's pretty printer
    print_batches(&collected).map_err(crate::Error::Arrow)?;

    if count < dataset.len() {
        println!("... showing {} of {} rows", count, dataset.len());
    }

    Ok(())
}

fn cmd_schema(path: &PathBuf) -> crate::Result<()> {
    let dataset = load_dataset(path)?;
    let schema = dataset.schema();

    println!("Schema for {}:", path.display());
    println!();

    for (i, field) in schema.fields().iter().enumerate() {
        let nullable = if field.is_nullable() {
            "nullable"
        } else {
            "not null"
        };
        println!(
            "  {}: {} ({}) [{}]",
            i,
            field.name(),
            field.data_type(),
            nullable
        );
    }

    println!();
    println!("Total columns: {}", schema.fields().len());

    Ok(())
}

fn load_dataset(path: &PathBuf) -> crate::Result<ArrowDataset> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "parquet" => ArrowDataset::from_parquet(path),
        "csv" => ArrowDataset::from_csv(path),
        "json" | "jsonl" => ArrowDataset::from_json(path),
        ext => Err(crate::Error::unsupported_format(ext)),
    }
}

fn save_dataset(dataset: &ArrowDataset, path: &PathBuf) -> crate::Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "parquet" => dataset.to_parquet(path),
        "csv" => dataset.to_csv(path),
        "json" | "jsonl" => dataset.to_json(path),
        ext => Err(crate::Error::unsupported_format(ext)),
    }
}

fn get_format(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("parquet") => "Parquet",
        Some("arrow" | "ipc") => "Arrow IPC",
        Some("csv") => "CSV",
        Some("json" | "jsonl") => "JSON",
        _ => "Unknown",
    }
}

// Import commands

#[cfg(feature = "hf-hub")]
fn cmd_import_hf(
    repo_id: &str,
    output: &PathBuf,
    revision: &str,
    subset: Option<&str>,
    split: &str,
) -> crate::Result<()> {
    use crate::hf_hub::HfDataset;

    println!("Importing {} from HuggingFace Hub...", repo_id);

    let mut builder = HfDataset::builder(repo_id).revision(revision).split(split);

    if let Some(s) = subset {
        builder = builder.subset(s);
    }

    let dataset = builder.build()?;

    println!("Downloading to {}...", output.display());
    let data = dataset.download_to(output)?;

    println!(
        "Successfully imported {} ({} rows) to {}",
        repo_id,
        data.len(),
        output.display()
    );

    Ok(())
}

/// Prints a quality warning for HuggingFace Hub uploads.
#[cfg(feature = "hf-hub")]
fn print_quality_warning() {
    eprintln!();
    eprintln!("WARNING: Data quality is CRITICAL for ML datasets!");
    eprintln!("Publishing low-quality data harms the ML community.");
    eprintln!();
    eprintln!("Before publishing, verify quality with:");
    eprintln!("  alimentar quality score <file.parquet>");
    eprintln!();
    eprintln!("Minimum recommended: Grade B (85%)");
    eprintln!();
    eprintln!("To improve quality, use:");
    eprintln!("  aprender clean <input> -o <output>     # Clean data");
    eprintln!("  entrenar augment <input> -o <output>   # Augment for training");
    eprintln!();
    eprintln!("See: https://paiml.github.io/alimentar/hf-hub/publishing.html");
    eprintln!();
}

#[cfg(feature = "hf-hub")]
fn cmd_hub_push(
    input: &PathBuf,
    repo_id: &str,
    path_in_repo: Option<&str>,
    message: &str,
    readme: Option<&PathBuf>,
    private: bool,
) -> crate::Result<()> {
    use crate::hf_hub::HfPublisher;

    // Display quality warning
    print_quality_warning();

    // Validate input file exists
    if !input.exists() {
        return Err(crate::Error::io(
            std::io::Error::new(std::io::ErrorKind::NotFound, "Input file not found"),
            input,
        ));
    }

    // Derive path_in_repo from filename if not specified
    let path_in_repo = path_in_repo.map(String::from).unwrap_or_else(|| {
        input
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| "data.parquet".to_string())
    });

    println!("Pushing {} to {}...", input.display(), repo_id);

    let publisher = HfPublisher::new(repo_id)
        .with_private(private)
        .with_commit_message(message);

    // Create repo (idempotent - succeeds if already exists)
    println!("Creating repository (if needed)...");
    publisher.create_repo_sync()?;

    // Upload parquet file
    println!("Uploading {}...", path_in_repo);
    publisher.upload_parquet_file_sync(input, &path_in_repo)?;

    // Upload README if provided
    if let Some(readme_path) = readme {
        println!("Uploading README.md...");
        let readme_content =
            std::fs::read_to_string(readme_path).map_err(|e| crate::Error::io(e, readme_path))?;
        publisher.upload_readme_validated_sync(&readme_content)?;
    }

    let visibility = if private { "private" } else { "public" };
    println!(
        "Successfully pushed to https://huggingface.co/datasets/{} ({})",
        repo_id, visibility
    );

    Ok(())
}

// Registry commands

fn create_registry(path: &PathBuf) -> crate::Result<Registry> {
    // Ensure directory exists
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|e| crate::Error::io(e, path))?;
    }
    let backend = LocalBackend::new(path)?;
    Ok(Registry::new(Box::new(backend)))
}

fn cmd_registry_init(path: &PathBuf) -> crate::Result<()> {
    let registry = create_registry(path)?;
    registry.init()?;
    println!("Initialized registry at: {}", path.display());
    Ok(())
}

fn cmd_registry_list(path: &PathBuf) -> crate::Result<()> {
    let registry = create_registry(path)?;
    let datasets = registry.list()?;

    if datasets.is_empty() {
        println!("No datasets in registry.");
        return Ok(());
    }

    println!("Datasets in registry:\n");
    println!(
        "{:<25} {:<12} {:<10} {:<15} DESCRIPTION",
        "NAME", "LATEST", "VERSIONS", "ROWS"
    );
    println!("{}", "-".repeat(80));

    for ds in datasets {
        let desc = if ds.metadata.description.len() > 30 {
            format!("{}...", &ds.metadata.description[..27])
        } else {
            ds.metadata.description.clone()
        };
        println!(
            "{:<25} {:<12} {:<10} {:<15} {}",
            ds.name,
            ds.latest,
            ds.versions.len(),
            ds.num_rows,
            desc
        );
    }

    Ok(())
}

fn cmd_registry_push(
    input: &PathBuf,
    name: &str,
    version: &str,
    description: &str,
    license: &str,
    tags: &str,
    registry_path: &PathBuf,
) -> crate::Result<()> {
    let registry = create_registry(registry_path)?;

    // Initialize if needed
    registry.init()?;

    // Load the dataset
    let dataset = load_dataset(input)?;

    // Parse tags
    let tag_list: Vec<String> = if tags.is_empty() {
        Vec::new()
    } else {
        tags.split(',').map(|s| s.trim().to_string()).collect()
    };

    // Create metadata
    let metadata = DatasetMetadata {
        description: description.to_string(),
        license: license.to_string(),
        tags: tag_list,
        source: Some(input.display().to_string()),
        citation: None,
        sha256: None, // Computed during save, not at publish time
    };

    // Publish
    registry.publish(name, version, &dataset, metadata)?;

    println!(
        "Published {}@{} ({} rows) to registry",
        name,
        version,
        dataset.len()
    );

    Ok(())
}

fn cmd_registry_pull(
    name: &str,
    output: &PathBuf,
    version: Option<&str>,
    registry_path: &PathBuf,
) -> crate::Result<()> {
    let registry = create_registry(registry_path)?;

    // Pull the dataset
    let dataset = registry.pull(name, version)?;

    // Save to output
    dataset.to_parquet(output)?;

    let ver = version.unwrap_or("latest");
    println!(
        "Pulled {}@{} ({} rows) to {}",
        name,
        ver,
        dataset.len(),
        output.display()
    );

    Ok(())
}

fn cmd_registry_search(query: &str, path: &PathBuf) -> crate::Result<()> {
    let registry = create_registry(path)?;
    let results = registry.search(query)?;

    if results.is_empty() {
        println!("No datasets found matching '{}'", query);
        return Ok(());
    }

    println!("Search results for '{}':\n", query);
    println!("{:<25} {:<12} {:<10} DESCRIPTION", "NAME", "LATEST", "ROWS");
    println!("{}", "-".repeat(70));

    for ds in results {
        let desc = if ds.metadata.description.len() > 30 {
            format!("{}...", &ds.metadata.description[..27])
        } else {
            ds.metadata.description.clone()
        };
        println!(
            "{:<25} {:<12} {:<10} {}",
            ds.name, ds.latest, ds.num_rows, desc
        );
    }

    Ok(())
}

fn cmd_registry_show_info(name: &str, path: &PathBuf) -> crate::Result<()> {
    let registry = create_registry(path)?;
    let info = registry.get_info(name)?;

    println!("Dataset: {}", info.name);
    println!("Latest: {}", info.latest);
    println!("Versions: {}", info.versions.join(", "));
    println!("Rows: {}", info.num_rows);
    println!("Size: {} bytes", info.size_bytes);
    println!();
    println!("Description: {}", info.metadata.description);
    println!("License: {}", info.metadata.license);
    println!("Tags: {}", info.metadata.tags.join(", "));

    if let Some(source) = &info.metadata.source {
        println!("Source: {}", source);
    }
    if let Some(citation) = &info.metadata.citation {
        println!("Citation: {}", citation);
    }

    println!();
    println!("Schema:");
    if let Some(fields) = info.schema.get("fields").and_then(|f| f.as_array()) {
        for field in fields {
            let name = field.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            let dtype = field
                .get("data_type")
                .and_then(|d| d.as_str())
                .unwrap_or("?");
            let nullable = field
                .get("nullable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true);
            let null_str = if nullable { "nullable" } else { "not null" };
            println!("  - {} ({}) [{}]", name, dtype, null_str);
        }
    }

    Ok(())
}

fn cmd_registry_delete(name: &str, version: &str, path: &PathBuf) -> crate::Result<()> {
    let registry = create_registry(path)?;
    registry.delete(name, version)?;
    println!("Deleted {}@{} from registry", name, version);
    Ok(())
}

// Drift detection commands

fn parse_drift_tests(tests_str: &str) -> Vec<DriftTest> {
    tests_str
        .split(',')
        .filter_map(|t| match t.trim().to_lowercase().as_str() {
            "ks" => Some(DriftTest::KolmogorovSmirnov),
            "chi2" | "chisquared" => Some(DriftTest::ChiSquared),
            "psi" => Some(DriftTest::PSI),
            "js" | "jensenshannon" => Some(DriftTest::JensenShannon),
            _ => None,
        })
        .collect()
}

fn severity_symbol(severity: DriftSeverity) -> &'static str {
    match severity {
        DriftSeverity::None => "✓",
        DriftSeverity::Low => "○",
        DriftSeverity::Medium => "●",
        DriftSeverity::High => "▲",
        DriftSeverity::Critical => "✖",
    }
}

fn cmd_drift_detect(
    reference: &PathBuf,
    current: &PathBuf,
    tests_str: &str,
    alpha: f64,
    format: &str,
) -> crate::Result<()> {
    let ref_dataset = load_dataset(reference)?;
    let cur_dataset = load_dataset(current)?;

    let tests = parse_drift_tests(tests_str);
    if tests.is_empty() {
        return Err(crate::Error::invalid_config(
            "No valid tests specified. Use: ks, chi2, psi, js",
        ));
    }

    let mut detector = DriftDetector::new(ref_dataset).with_alpha(alpha);
    for test in tests {
        detector = detector.with_test(test);
    }

    let report = detector.detect(&cur_dataset)?;

    if format == "json" {
        // JSON output
        let json = serde_json::json!({
            "drift_detected": report.drift_detected,
            "columns": report.column_scores.values().map(|d| {
                serde_json::json!({
                    "column": d.column,
                    "test": format!("{:?}", d.test),
                    "statistic": d.statistic,
                    "p_value": d.p_value,
                    "drift_detected": d.drift_detected,
                    "severity": format!("{:?}", d.severity),
                })
            }).collect::<Vec<_>>()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json)
                .map_err(|e| crate::Error::Format(e.to_string()))?
        );
    } else {
        // Text output
        println!("Drift Detection Report");
        println!("======================");
        println!("Reference: {}", reference.display());
        println!("Current:   {}", current.display());
        println!("Alpha:     {}", alpha);
        println!();

        if report.drift_detected {
            println!("⚠️  DRIFT DETECTED\n");
        } else {
            println!("✓ No significant drift detected\n");
        }

        println!(
            "{:<20} {:<15} {:<12} {:<12} {:<10} DRIFT",
            "COLUMN", "TEST", "STATISTIC", "P-VALUE", "SEVERITY"
        );
        println!("{}", "-".repeat(80));

        for drift in report.column_scores.values() {
            let drift_str = if drift.drift_detected { "YES" } else { "no" };
            let p_value_str = drift
                .p_value
                .map_or_else(|| "N/A".to_string(), |p| format!("{:.4}", p));
            println!(
                "{:<20} {:<15} {:<12.4} {:<12} {:<10} {} {}",
                drift.column,
                format!("{:?}", drift.test),
                drift.statistic,
                p_value_str,
                format!("{:?}", drift.severity),
                severity_symbol(drift.severity),
                drift_str
            );
        }

        println!();
        let drifted: Vec<_> = report
            .column_scores
            .values()
            .filter(|d| d.drift_detected)
            .map(|d| d.column.clone())
            .collect();
        if !drifted.is_empty() {
            println!("Columns with drift: {}", drifted.join(", "));
        }
    }

    Ok(())
}

fn cmd_drift_report(
    reference: &PathBuf,
    current: &PathBuf,
    output: Option<&PathBuf>,
) -> crate::Result<()> {
    let ref_dataset = load_dataset(reference)?;
    let cur_dataset = load_dataset(current)?;

    let detector = DriftDetector::new(ref_dataset)
        .with_test(DriftTest::KolmogorovSmirnov)
        .with_test(DriftTest::PSI)
        .with_test(DriftTest::JensenShannon);

    let report = detector.detect(&cur_dataset)?;

    let drifted_count = report
        .column_scores
        .values()
        .filter(|d| d.drift_detected)
        .count();
    let json = serde_json::json!({
        "reference": reference.display().to_string(),
        "current": current.display().to_string(),
        "drift_detected": report.drift_detected,
        "summary": {
            "total_columns": report.column_scores.len(),
            "drifted_columns": drifted_count,
        },
        "columns": report.column_scores.values().map(|d| {
            serde_json::json!({
                "column": d.column,
                "test": format!("{:?}", d.test),
                "statistic": d.statistic,
                "p_value": d.p_value,
                "drift_detected": d.drift_detected,
                "severity": format!("{:?}", d.severity),
            })
        }).collect::<Vec<_>>()
    });

    let json_str =
        serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?;

    if let Some(output_path) = output {
        std::fs::write(output_path, &json_str).map_err(|e| crate::Error::io(e, output_path))?;
        println!("Drift report written to: {}", output_path.display());
    } else {
        println!("{}", json_str);
    }

    Ok(())
}

// Sketch-based distributed drift detection commands

fn parse_sketch_type(s: &str) -> Option<SketchType> {
    match s.to_lowercase().as_str() {
        "tdigest" | "t-digest" => Some(SketchType::TDigest),
        "ddsketch" | "dd-sketch" => Some(SketchType::DDSketch),
        _ => None,
    }
}

fn cmd_drift_sketch(
    input: &PathBuf,
    output: &PathBuf,
    sketch_type: &str,
    source: Option<&str>,
    format: &str,
) -> crate::Result<()> {
    let sketch_type = parse_sketch_type(sketch_type).ok_or_else(|| {
        crate::Error::invalid_config(format!(
            "Unknown sketch type: {}. Use 'tdigest' or 'ddsketch'",
            sketch_type
        ))
    })?;

    let dataset = load_dataset(input)?;
    let mut sketch = DataSketch::from_dataset(&dataset, sketch_type)?;

    if let Some(src) = source {
        sketch = sketch.with_source(src);
    }

    match format {
        "binary" | "bin" => {
            let bytes = sketch.to_bytes()?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            // Default to JSON
            let json = serde_json::to_string_pretty(&sketch)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Created {} sketch from {} ({} rows) -> {}",
        sketch_type.name(),
        input.display(),
        dataset.len(),
        output.display()
    );

    Ok(())
}

fn load_sketch(path: &PathBuf) -> crate::Result<DataSketch> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "bin" | "binary" => {
            let bytes = std::fs::read(path).map_err(|e| crate::Error::io(e, path))?;
            DataSketch::from_bytes(&bytes)
        }
        _ => {
            // Default to JSON
            let json = std::fs::read_to_string(path).map_err(|e| crate::Error::io(e, path))?;
            serde_json::from_str(&json)
                .map_err(|e| crate::Error::Format(format!("Invalid sketch JSON: {}", e)))
        }
    }
}

fn cmd_drift_merge(sketches: &[PathBuf], output: &PathBuf, format: &str) -> crate::Result<()> {
    if sketches.is_empty() {
        return Err(crate::Error::invalid_config(
            "No sketches provided to merge",
        ));
    }

    let loaded: Vec<DataSketch> = sketches
        .iter()
        .map(load_sketch)
        .collect::<Result<Vec<_>, _>>()?;

    let merged = DataSketch::merge(&loaded)?;

    match format {
        "binary" | "bin" => {
            let bytes = merged.to_bytes()?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            let json = serde_json::to_string_pretty(&merged)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Merged {} sketches ({} total rows) -> {}",
        sketches.len(),
        merged.row_count,
        output.display()
    );

    Ok(())
}

fn cmd_drift_compare(
    reference: &PathBuf,
    current: &PathBuf,
    threshold: f64,
    format: &str,
) -> crate::Result<()> {
    let ref_sketch = load_sketch(reference)?;
    let cur_sketch = load_sketch(current)?;

    let detector = DistributedDriftDetector::new()
        .with_sketch_type(ref_sketch.sketch_type)
        .with_threshold(threshold);

    let results = detector.compare(&ref_sketch, &cur_sketch)?;

    let drift_detected = results.iter().any(|r| r.severity.is_drift());

    if format == "json" {
        let json = serde_json::json!({
            "reference": reference.display().to_string(),
            "current": current.display().to_string(),
            "drift_detected": drift_detected,
            "threshold": threshold,
            "columns": results.iter().map(|r| {
                serde_json::json!({
                    "column": r.column,
                    "statistic": r.statistic,
                    "severity": format!("{:?}", r.severity),
                    "drift_detected": r.severity.is_drift(),
                })
            }).collect::<Vec<_>>()
        });

        let json_str = serde_json::to_string_pretty(&json)
            .map_err(|e| crate::Error::Format(e.to_string()))?;
        println!("{}", json_str);
    } else {
        // Text format
        println!("Sketch Drift Comparison");
        println!("=======================");
        println!("Reference: {}", reference.display());
        println!("Current:   {}", current.display());
        println!("Threshold: {}", threshold);
        println!();

        if results.is_empty() {
            println!("No numeric columns to compare.");
        } else {
            println!(
                "{:<20} {:>10} {:>10} DRIFT",
                "COLUMN", "STATISTIC", "SEVERITY"
            );
            println!("{}", "-".repeat(55));

            for result in &results {
                let drift_symbol = if result.severity.is_drift() {
                    severity_symbol(result.severity)
                } else {
                    "✓"
                };
                println!(
                    "{:<20} {:>10.4} {:>10} {}",
                    result.column,
                    result.statistic,
                    format!("{:?}", result.severity),
                    drift_symbol
                );
            }

            println!();
            if drift_detected {
                println!("⚠ Drift detected in one or more columns");
            } else {
                println!("✓ No significant drift detected");
            }
        }
    }

    Ok(())
}

// Quality checking commands

fn cmd_quality_check(
    path: &PathBuf,
    null_threshold: f64,
    _duplicate_threshold: f64,
    detect_outliers: bool,
    format: &str,
) -> crate::Result<()> {
    let dataset = load_dataset(path)?;

    let mut checker = QualityChecker::new();

    if !detect_outliers {
        checker = checker.with_outlier_check(false);
    }

    let report = checker.check(&dataset)?;

    if format == "json" {
        let json = serde_json::json!({
            "path": path.display().to_string(),
            "rows": report.row_count,
            "columns": report.column_count,
            "has_issues": !report.issues.is_empty(),
            "score": report.score,
            "issues": report.issues.iter().map(|i| format!("{:?}", i)).collect::<Vec<_>>(),
            "column_qualities": report.columns.iter().map(|(name, c)| {
                serde_json::json!({
                    "column": name,
                    "null_ratio": c.null_ratio,
                    "unique_count": c.unique_count,
                    "is_constant": c.is_constant(),
                    "is_mostly_null": c.null_ratio > null_threshold,
                })
            }).collect::<Vec<_>>()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json)
                .map_err(|e| crate::Error::Format(e.to_string()))?
        );
    } else {
        println!("Data Quality Report");
        println!("===================");
        println!("File: {}", path.display());
        println!("Rows: {}", report.row_count);
        println!("Columns: {}", report.column_count);
        println!();

        println!("Quality Score: {:.1}%", report.score);
        println!();

        if report.issues.is_empty() {
            println!("✓ No quality issues found\n");
        } else {
            println!("Issues Found:");
            println!("-------------");
            for issue in &report.issues {
                println!("  - {:?}", issue);
            }
            println!();
        }

        println!(
            "{:<20} {:<12} {:<12} {:<10}",
            "COLUMN", "NULL %", "UNIQUE", "STATUS"
        );
        println!("{}", "-".repeat(60));

        for (name, col) in &report.columns {
            let status = if col.is_constant() {
                "CONSTANT"
            } else if col.null_ratio > null_threshold {
                "HIGH NULL"
            } else {
                "OK"
            };

            println!(
                "{:<20} {:<12.2} {:<12} {:<10}",
                name,
                col.null_ratio * 100.0,
                col.unique_count,
                status
            );
        }
    }

    Ok(())
}

fn cmd_quality_report(path: &PathBuf, output: Option<&PathBuf>) -> crate::Result<()> {
    let dataset = load_dataset(path)?;
    let report = QualityChecker::new().check(&dataset)?;

    let json = serde_json::json!({
        "path": path.display().to_string(),
        "rows": report.row_count,
        "columns": report.column_count,
        "has_issues": !report.issues.is_empty(),
        "score": report.score,
        "issues": report.issues.iter().map(|i| format!("{:?}", i)).collect::<Vec<_>>(),
        "column_qualities": report.columns.iter().map(|(name, c)| {
            serde_json::json!({
                "column": name,
                "null_ratio": c.null_ratio,
                "unique_count": c.unique_count,
                "is_constant": c.is_constant(),
            })
        }).collect::<Vec<_>>()
    });

    let json_str =
        serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?;

    if let Some(output_path) = output {
        std::fs::write(output_path, &json_str).map_err(|e| crate::Error::io(e, output_path))?;
        println!("Quality report written to: {}", output_path.display());
    } else {
        println!("{}", json_str);
    }

    Ok(())
}

/// Calculate 100-point quality score with letter grade (GH-6)
///
/// Implements the Doctest Corpus QA Checklist for Publication with
/// weighted scoring per Toyota Way Jidoka principles.
fn cmd_quality_score(
    path: &PathBuf,
    profile_name: &str,
    suggest: bool,
    json_output: bool,
    badge_output: bool,
) -> crate::Result<()> {
    use crate::quality::{QualityProfile, QualityScore, Severity};

    // Load the quality profile
    let profile = QualityProfile::by_name(profile_name).ok_or_else(|| {
        crate::Error::Format(format!(
            "Unknown quality profile '{}'. Available: {:?}",
            profile_name,
            QualityProfile::available_profiles()
        ))
    })?;

    let dataset = load_dataset(path)?;
    let report = QualityChecker::new().check(&dataset)?;

    // Wire QualityReport to ChecklistItems per the 100-point checklist
    let checklist = build_checklist_from_report(&report, &profile);
    let score = QualityScore::from_checklist(checklist);

    // Output based on flags
    if badge_output {
        println!("{}", score.badge_url());
    } else if json_output {
        println!("{}", score.to_json());
    } else {
        // Text output (Andon-style visual management)
        let grade_symbol = match score.grade {
            crate::quality::LetterGrade::A | crate::quality::LetterGrade::B => "✓",
            crate::quality::LetterGrade::C => "○",
            crate::quality::LetterGrade::D => "△",
            crate::quality::LetterGrade::F => "✗",
        };

        println!("═══════════════════════════════════════════════════════════════");
        println!("  Data Quality Score: {} {} ({:.1}%)  ", grade_symbol, score.grade, score.score);
        println!("  Profile: {}  ", profile.name);
        println!("  Decision: {}  ", score.grade.publication_decision());
        println!("═══════════════════════════════════════════════════════════════");
        println!();
        println!("File: {}", path.display());
        println!("Points: {:.1} / {:.1}", score.points_earned, score.max_points);
        println!();

        // Severity breakdown
        println!("Severity Breakdown:");
        for severity in [Severity::Critical, Severity::High, Severity::Medium, Severity::Low] {
            if let Some(stats) = score.severity_breakdown.get(&severity) {
                let status = if stats.failed == 0 { "✓" } else { "✗" };
                println!(
                    "  {} {:8}: {}/{} passed ({:.1}/{:.1} pts)",
                    status, format!("{}", severity), stats.passed, stats.total,
                    stats.points_earned, stats.max_points
                );
            }
        }
        println!();

        // Critical failures get highlighted
        let critical_failures = score.critical_failures();
        if !critical_failures.is_empty() {
            println!("CRITICAL FAILURES (blocks publication):");
            for item in critical_failures {
                println!("  ✗ #{}: {}", item.id, item.description);
                if suggest {
                    if let Some(ref suggestion) = item.suggestion {
                        println!("    → {}", suggestion);
                    }
                }
            }
            println!();
        }

        // Show suggestions for all failed items if --suggest flag
        if suggest {
            let failed = score.failed_items();
            let non_critical: Vec<_> = failed
                .iter()
                .filter(|i| i.severity != Severity::Critical)
                .collect();

            if !non_critical.is_empty() {
                println!("Other Issues ({}):", non_critical.len());
                for item in non_critical {
                    let sev = match item.severity {
                        Severity::High => "[HIGH]",
                        Severity::Medium => "[MED]",
                        Severity::Low => "[LOW]",
                        Severity::Critical => "[CRIT]",
                    };
                    println!("  {} #{}: {}", sev, item.id, item.description);
                    if let Some(ref suggestion) = item.suggestion {
                        println!("      → {}", suggestion);
                    }
                }
            }
        }
    }

    // Exit with non-zero code if critical failures (for CI/CD)
    if score.has_critical_failures() {
        std::process::exit(1);
    }

    Ok(())
}

/// List available quality profiles
#[allow(clippy::unnecessary_wraps)]
fn cmd_quality_profiles() -> crate::Result<()> {
    use crate::quality::QualityProfile;

    println!("Available Quality Profiles");
    println!("══════════════════════════");
    println!();

    for name in QualityProfile::available_profiles() {
        if let Some(profile) = QualityProfile::by_name(name) {
            println!("  {} - {}", profile.name, profile.description);
            if !profile.expected_constant_columns.is_empty() {
                let cols: Vec<_> = profile.expected_constant_columns.iter().collect();
                println!("    Expected constants: {:?}", cols);
            }
            if !profile.nullable_columns.is_empty() {
                let cols: Vec<_> = profile.nullable_columns.iter().collect();
                println!("    Nullable columns: {:?}", cols);
            }
            println!("    Max null ratio: {:.0}%", profile.max_null_ratio * 100.0);
            println!("    Max duplicate ratio: {:.0}%", profile.max_duplicate_ratio * 100.0);
            println!();
        }
    }

    println!("Usage: alimentar quality score <path> --profile <name>");
    Ok(())
}

/// Build checklist items from `QualityReport`
///
/// Maps `QualityReport` findings to the 100-point checklist defined in GH-6.
/// This wires the existing quality checks to the weighted scoring system.
#[allow(clippy::too_many_lines)]
fn build_checklist_from_report(
    report: &crate::quality::QualityReport,
    profile: &crate::quality::QualityProfile,
) -> Vec<crate::quality::ChecklistItem> {
    use crate::quality::{ChecklistItem, ColumnQuality, Severity};

    let mut items = Vec::new();
    let mut id: u8 = 1;

    // === Critical Checks (2.0x weight) ===

    // Check 1: Dataset not empty
    let has_rows = report.row_count > 0;
    items.push(
        ChecklistItem::new(id, "Dataset contains rows", Severity::Critical, has_rows)
            .with_suggestion("Extract more doctests or check input source"),
    );
    id += 1;

    // Check 2: No empty schema
    let has_columns = report.column_count > 0;
    items.push(
        ChecklistItem::new(id, "Schema has columns defined", Severity::Critical, has_columns)
            .with_suggestion("Verify parser is extracting fields correctly"),
    );
    id += 1;

    // Check 3: No unexpected constant columns (would break training)
    // Filter out columns that the profile expects to be constant (e.g., source, version)
    // Also allow nullable columns to be all-null (constant null is OK for optional fields)
    let unexpected_constant_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(name, c): &(&String, &ColumnQuality)| {
            c.is_constant()
                && !profile.is_expected_constant(name)
                && !profile.is_nullable(name)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let no_unexpected_constants = unexpected_constant_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "No unexpected constant columns (zero variance)",
            Severity::Critical,
            no_unexpected_constants,
        )
        .with_suggestion(format!(
            "Remove or investigate constant columns: {:?}",
            unexpected_constant_cols
        )),
    );
    id += 1;

    // === High Priority Checks (1.5x weight) ===

    // Check 4: Duplicate ratio below threshold (default 5%)
    let duplicate_ratio = report
        .issues
        .iter()
        .find_map(|i| {
            if let crate::quality::QualityIssue::DuplicateRows {
                duplicate_ratio: dr,
                ..
            } = i
            {
                Some(*dr)
            } else {
                None
            }
        })
        .unwrap_or(0.0);
    let low_duplicates = duplicate_ratio <= 0.05;
    items.push(
        ChecklistItem::new(
            id,
            format!("Duplicate ratio <= 5% (actual: {:.1}%)", duplicate_ratio * 100.0),
            Severity::High,
            low_duplicates,
        )
        .with_suggestion("Run deduplication: alimentar dedupe <file>"),
    );
    id += 1;

    // Check 5: No columns with >50% nulls (except nullable columns per profile)
    let high_null_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(name, c): &(&String, &ColumnQuality)| {
            c.null_ratio > 0.5 && !profile.is_nullable(name)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let no_high_null = high_null_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "No columns with >50% null values",
            Severity::High,
            no_high_null,
        )
        .with_suggestion(format!(
            "Investigate high-null columns: {:?}",
            high_null_cols
        )),
    );
    id += 1;

    // Check 6: Minimum row count (at least 100 for meaningful training)
    let min_rows = report.row_count >= 100;
    items.push(
        ChecklistItem::new(
            id,
            format!("Minimum 100 rows (actual: {})", report.row_count),
            Severity::High,
            min_rows,
        )
        .with_suggestion("Extract more data or combine with other sources"),
    );
    id += 1;

    // === Medium Priority Checks (1.0x weight) ===

    // Check 7: Overall quality score from existing checker
    let good_score = report.score >= 70.0;
    items.push(
        ChecklistItem::new(
            id,
            format!("Quality score >= 70% (actual: {:.1}%)", report.score),
            Severity::Medium,
            good_score,
        )
        .with_suggestion("Address issues reported by quality check"),
    );
    id += 1;

    // Check 8: No columns with >10% nulls (stricter, except nullable columns per profile)
    let moderate_null_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(name, c): &(&String, &ColumnQuality)| {
            c.null_ratio > 0.1 && c.null_ratio <= 0.5 && !profile.is_nullable(name)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let low_null_ratio = moderate_null_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "No columns with >10% null values",
            Severity::Medium,
            low_null_ratio,
        )
        .with_suggestion(format!(
            "Consider imputation for: {:?}",
            moderate_null_cols
        )),
    );
    id += 1;

    // Check 9: Reasonable column count (not too few for ML)
    let enough_columns = report.column_count >= 2;
    items.push(
        ChecklistItem::new(
            id,
            format!("At least 2 columns (actual: {})", report.column_count),
            Severity::Medium,
            enough_columns,
        )
        .with_suggestion("Ensure input and target columns are present"),
    );
    id += 1;

    // Check 10: No outlier issues detected
    let outlier_issues: Vec<(String, f64)> = report
        .issues
        .iter()
        .filter_map(|i| {
            if let crate::quality::QualityIssue::OutliersDetected {
                column,
                outlier_ratio: or,
                ..
            } = i
            {
                Some((column.clone(), *or))
            } else {
                None
            }
        })
        .collect();
    let no_severe_outliers = outlier_issues.iter().all(|(_, r)| *r < 0.1);
    items.push(
        ChecklistItem::new(
            id,
            "No columns with >10% outliers",
            Severity::Medium,
            no_severe_outliers,
        )
        .with_suggestion("Review outlier columns for data quality issues"),
    );
    id += 1;

    // === Low Priority Checks (0.5x weight) ===

    // Check 11: No warnings at all
    let no_issues = report.issues.is_empty();
    items.push(
        ChecklistItem::new(id, "No quality warnings", Severity::Low, no_issues)
            .with_suggestion("Address all warnings for best results"),
    );
    id += 1;

    // Check 12: Good cardinality (unique values)
    let low_cardinality_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(_, c): &(&String, &ColumnQuality)| c.unique_count < 10 && !c.is_constant())
        .map(|(n, _)| n.clone())
        .collect();
    let good_cardinality = low_cardinality_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "All columns have reasonable cardinality (>10 unique)",
            Severity::Low,
            good_cardinality,
        )
        .with_suggestion(format!(
            "Low cardinality columns: {:?}",
            low_cardinality_cols
        )),
    );
    let _ = id; // suppress warning

    items
}

// Federated split coordination commands

fn parse_fed_strategy(
    strategy: &str,
    train_ratio: f64,
    seed: u64,
    stratify_column: Option<&str>,
) -> Option<FederatedSplitStrategy> {
    match strategy.to_lowercase().as_str() {
        "local" | "local-seed" => Some(FederatedSplitStrategy::LocalWithSeed { seed, train_ratio }),
        "proportional" | "iid" => Some(FederatedSplitStrategy::ProportionalIID { train_ratio }),
        "stratified" => {
            let column = stratify_column.unwrap_or("label").to_string();
            Some(FederatedSplitStrategy::GlobalStratified {
                label_column: column,
                target_distribution: std::collections::HashMap::new(),
            })
        }
        _ => None,
    }
}

fn cmd_fed_manifest(
    input: &PathBuf,
    output: &PathBuf,
    node_id: &str,
    train_ratio: f64,
    seed: u64,
    format: &str,
) -> crate::Result<()> {
    let dataset = load_dataset(input)?;

    // Create a split to generate the manifest
    let split =
        DatasetSplit::from_ratios(&dataset, train_ratio, 1.0 - train_ratio, None, Some(seed))?;

    let manifest = NodeSplitManifest::from_split(node_id, &split);

    match format {
        "binary" | "bin" => {
            let bytes = rmp_serde::to_vec(&manifest)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            let json = serde_json::to_string_pretty(&manifest)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Created manifest for node '{}' ({} rows) -> {}",
        node_id,
        dataset.len(),
        output.display()
    );

    Ok(())
}

fn load_manifest(path: &PathBuf) -> crate::Result<NodeSplitManifest> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "bin" | "binary" => {
            let bytes = std::fs::read(path).map_err(|e| crate::Error::io(e, path))?;
            rmp_serde::from_slice(&bytes)
                .map_err(|e| crate::Error::Format(format!("Invalid manifest binary: {}", e)))
        }
        _ => {
            let json = std::fs::read_to_string(path).map_err(|e| crate::Error::io(e, path))?;
            serde_json::from_str(&json)
                .map_err(|e| crate::Error::Format(format!("Invalid manifest JSON: {}", e)))
        }
    }
}

fn cmd_fed_plan(
    manifests: &[PathBuf],
    output: &PathBuf,
    strategy: &str,
    train_ratio: f64,
    seed: u64,
    stratify_column: Option<&str>,
    format: &str,
) -> crate::Result<()> {
    if manifests.is_empty() {
        return Err(crate::Error::invalid_config("No manifests provided"));
    }

    let loaded: Vec<NodeSplitManifest> = manifests
        .iter()
        .map(load_manifest)
        .collect::<Result<Vec<_>, _>>()?;

    let strategy =
        parse_fed_strategy(strategy, train_ratio, seed, stratify_column).ok_or_else(|| {
            crate::Error::invalid_config(format!(
                "Unknown strategy: {}. Use 'local', 'proportional', or 'stratified'",
                strategy
            ))
        })?;

    let coordinator = FederatedSplitCoordinator::new(strategy);
    let instructions = coordinator.compute_split_plan(&loaded)?;

    match format {
        "binary" | "bin" => {
            let bytes = rmp_serde::to_vec(&instructions)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            let json = serde_json::to_string_pretty(&instructions)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Created split plan for {} nodes -> {}",
        instructions.len(),
        output.display()
    );

    Ok(())
}

fn load_plan(path: &PathBuf) -> crate::Result<Vec<NodeSplitInstruction>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "bin" | "binary" => {
            let bytes = std::fs::read(path).map_err(|e| crate::Error::io(e, path))?;
            rmp_serde::from_slice(&bytes)
                .map_err(|e| crate::Error::Format(format!("Invalid plan binary: {}", e)))
        }
        _ => {
            let json = std::fs::read_to_string(path).map_err(|e| crate::Error::io(e, path))?;
            serde_json::from_str(&json)
                .map_err(|e| crate::Error::Format(format!("Invalid plan JSON: {}", e)))
        }
    }
}

fn cmd_fed_split(
    input: &PathBuf,
    plan: &PathBuf,
    node_id: &str,
    train_output: &PathBuf,
    test_output: &PathBuf,
    validation_output: Option<&PathBuf>,
) -> crate::Result<()> {
    let dataset = load_dataset(input)?;
    let instructions = load_plan(plan)?;

    // Find instruction for this node
    let instruction = instructions
        .iter()
        .find(|i| i.node_id == node_id)
        .ok_or_else(|| {
            crate::Error::invalid_config(format!(
                "No instruction found for node '{}' in plan",
                node_id
            ))
        })?;

    // Execute the split
    let split = FederatedSplitCoordinator::execute_local_split(&dataset, instruction)?;

    // Save outputs
    split.train.to_parquet(train_output)?;
    split.test.to_parquet(test_output)?;

    if let (Some(val_output), Some(val_data)) = (validation_output, &split.validation) {
        val_data.to_parquet(val_output)?;
    }

    println!(
        "Split executed for node '{}': {} train, {} test{}",
        node_id,
        split.train.len(),
        split.test.len(),
        split
            .validation
            .as_ref()
            .map_or(String::new(), |v| format!(", {} validation", v.len()))
    );

    Ok(())
}

fn cmd_fed_verify(manifests: &[PathBuf], format: &str) -> crate::Result<()> {
    if manifests.is_empty() {
        return Err(crate::Error::invalid_config("No manifests provided"));
    }

    let loaded: Vec<NodeSplitManifest> = manifests
        .iter()
        .map(load_manifest)
        .collect::<Result<Vec<_>, _>>()?;

    let report = FederatedSplitCoordinator::verify_global_split(&loaded)?;

    if format == "json" {
        let json = serde_json::json!({
            "total_rows": report.total_rows,
            "total_train_rows": report.total_train_rows,
            "total_test_rows": report.total_test_rows,
            "total_validation_rows": report.total_validation_rows,
            "effective_train_ratio": report.effective_train_ratio,
            "effective_test_ratio": report.effective_test_ratio,
            "effective_validation_ratio": report.effective_validation_ratio,
            "quality_passed": report.quality_passed,
            "issues": report.issues.iter().map(|i| format!("{:?}", i)).collect::<Vec<_>>(),
            "node_summaries": report.node_summaries.iter().map(|n| {
                serde_json::json!({
                    "node_id": n.node_id,
                    "contribution_ratio": n.contribution_ratio,
                    "train_ratio": n.train_ratio,
                    "test_ratio": n.test_ratio,
                })
            }).collect::<Vec<_>>()
        });

        let json_str = serde_json::to_string_pretty(&json)
            .map_err(|e| crate::Error::Format(e.to_string()))?;
        println!("{}", json_str);
    } else {
        // Text format
        println!("Federated Split Verification");
        println!("============================");
        println!();
        println!("Global Statistics:");
        println!("  Total rows:        {}", report.total_rows);
        println!(
            "  Train rows:        {} ({:.1}%)",
            report.total_train_rows,
            report.effective_train_ratio * 100.0
        );
        println!(
            "  Test rows:         {} ({:.1}%)",
            report.total_test_rows,
            report.effective_test_ratio * 100.0
        );
        if let Some(val) = report.total_validation_rows {
            println!(
                "  Validation rows:   {} ({:.1}%)",
                val,
                report.effective_validation_ratio.unwrap_or(0.0) * 100.0
            );
        }
        println!();

        println!("Node Summaries:");
        println!(
            "{:<15} {:>12} {:>10} {:>10}",
            "NODE", "CONTRIBUTION", "TRAIN", "TEST"
        );
        println!("{}", "-".repeat(50));

        for summary in &report.node_summaries {
            println!(
                "{:<15} {:>11.1}% {:>9.1}% {:>9.1}%",
                summary.node_id,
                summary.contribution_ratio * 100.0,
                summary.train_ratio * 100.0,
                summary.test_ratio * 100.0
            );
        }

        println!();
        if report.quality_passed {
            println!("✓ Quality check passed");
        } else {
            println!("⚠ Quality issues detected:");
            for issue in &report.issues {
                println!("  - {:?}", issue);
            }
        }
    }

    Ok(())
}

// =============================================================================
// Doctest Commands
// =============================================================================

#[cfg(feature = "doctest")]
fn cmd_doctest_extract(
    input: &Path,
    output: &Path,
    source: &str,
    version: &str,
) -> crate::Result<()> {
    use crate::DocTestParser;

    if !input.is_dir() {
        return Err(crate::Error::invalid_config(format!(
            "Input path must be a directory: {}",
            input.display()
        )));
    }

    let parser = DocTestParser::new();
    let corpus = parser.parse_directory(input, source, version)?;

    println!(
        "Extracted {} doctests from {} ({})",
        corpus.len(),
        source,
        version
    );

    if corpus.is_empty() {
        println!("Warning: No doctests found in {}", input.display());
        return Ok(());
    }

    let dataset = corpus.to_dataset()?;
    dataset.to_parquet(output)?;

    println!("Wrote {} to {}", corpus.len(), output.display());
    Ok(())
}

#[cfg(feature = "doctest")]
fn cmd_doctest_merge(inputs: &[PathBuf], output: &Path) -> crate::Result<()> {
    if inputs.is_empty() {
        return Err(crate::Error::invalid_config("No input files provided"));
    }

    // Load all datasets and concatenate
    let mut all_batches = Vec::new();
    let mut total_rows = 0;

    for input in inputs {
        let dataset = ArrowDataset::from_parquet(input)?;
        total_rows += dataset.len();
        for batch in dataset.iter() {
            all_batches.push(batch.clone());
        }
    }

    if all_batches.is_empty() {
        return Err(crate::Error::invalid_config(
            "No data found in input files",
        ));
    }

    // Create merged dataset
    let merged = ArrowDataset::new(all_batches)?;
    merged.to_parquet(output)?;

    println!(
        "Merged {} doctests from {} files to {}",
        total_rows,
        inputs.len(),
        output.display()
    );
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::redundant_clone,
    clippy::cast_lossless,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_late_init,
    clippy::redundant_pattern_matching
)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_parquet(path: &PathBuf, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));
    }

    #[test]
    fn test_cmd_info() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_info(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_head() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_head(&path, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_schema() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 10);

        let result = cmd_schema(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.parquet");
        create_test_parquet(&input, 50);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());

        // Verify output was created and has same data
        let original = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load original"));
        let converted = ArrowDataset::from_parquet(&output)
            .ok()
            .unwrap_or_else(|| panic!("Should load converted"));

        assert_eq!(original.len(), converted.len());
    }

    #[test]
    fn test_load_dataset_unsupported() {
        let path = PathBuf::from("test.xyz");
        let result = load_dataset(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format() {
        assert_eq!(get_format(Path::new("test.parquet")), "Parquet");
        assert_eq!(get_format(Path::new("test.arrow")), "Arrow IPC");
        assert_eq!(get_format(Path::new("test.csv")), "CSV");
        assert_eq!(get_format(Path::new("test.json")), "JSON");
        assert_eq!(get_format(Path::new("test.unknown")), "Unknown");
    }

    // Registry CLI tests

    #[test]
    fn test_cmd_registry_init() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        let result = cmd_registry_init(&registry_path);
        assert!(result.is_ok());
        assert!(registry_path.exists());
    }

    #[test]
    fn test_cmd_registry_list_empty() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        // Init first
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let result = cmd_registry_list(&registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_push_and_pull() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");
        let output = temp_dir.path().join("pulled.parquet");

        // Create test data
        create_test_parquet(&input, 25);

        // Push
        let result = cmd_registry_push(
            &input,
            "test-dataset",
            "1.0.0",
            "A test dataset",
            "MIT",
            "test,example",
            &registry_path,
        );
        assert!(result.is_ok());

        // List should show the dataset
        let result = cmd_registry_list(&registry_path);
        assert!(result.is_ok());

        // Pull
        let result = cmd_registry_pull("test-dataset", &output, Some("1.0.0"), &registry_path);
        assert!(result.is_ok());
        assert!(output.exists());

        // Verify data
        let original = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load original"));
        let pulled = ArrowDataset::from_parquet(&output)
            .ok()
            .unwrap_or_else(|| panic!("Should load pulled"));
        assert_eq!(original.len(), pulled.len());
    }

    #[test]
    fn test_cmd_registry_search() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push with description
        cmd_registry_push(
            &input,
            "ml-dataset",
            "1.0.0",
            "Machine learning training data",
            "Apache-2.0",
            "ml,training",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        // Search by name
        let result = cmd_registry_search("ml", &registry_path);
        assert!(result.is_ok());

        // Search by description
        let result = cmd_registry_search("machine", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        cmd_registry_push(
            &input,
            "info-test",
            "1.0.0",
            "Test description",
            "MIT",
            "test",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        let result = cmd_registry_show_info("info-test", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_delete() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push
        cmd_registry_push(
            &input,
            "delete-test",
            "1.0.0",
            "Will be deleted",
            "",
            "",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        // Delete
        let result = cmd_registry_delete("delete-test", "1.0.0", &registry_path);
        assert!(result.is_ok());

        // Should no longer exist
        let result = cmd_registry_show_info("delete-test", &registry_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_registry_pull_latest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input1 = temp_dir.path().join("v1.parquet");
        let input2 = temp_dir.path().join("v2.parquet");
        let output = temp_dir.path().join("pulled.parquet");

        create_test_parquet(&input1, 10);
        create_test_parquet(&input2, 20);

        // Push v1
        cmd_registry_push(&input1, "versioned", "1.0.0", "V1", "", "", &registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should push v1"));

        // Push v2
        cmd_registry_push(&input2, "versioned", "2.0.0", "V2", "", "", &registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should push v2"));

        // Pull latest (no version specified)
        let result = cmd_registry_pull("versioned", &output, None, &registry_path);
        assert!(result.is_ok());

        // Should be v2 (20 rows)
        let pulled = ArrowDataset::from_parquet(&output)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        assert_eq!(pulled.len(), 20);
    }

    #[test]
    fn test_cmd_head_with_more_rows_than_dataset() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 5);

        // Request more rows than exist
        let result = cmd_head(&path, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert_parquet_to_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.csv");
        create_test_parquet(&input, 25);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_convert_parquet_to_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.json");
        create_test_parquet(&input, 15);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_save_dataset_unsupported_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("data.parquet");
        let output = temp_dir.path().join("output.xyz");
        create_test_parquet(&input, 5);

        let dataset = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));

        let result = save_dataset(&dataset, &output);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format_ipc() {
        assert_eq!(get_format(Path::new("test.ipc")), "Arrow IPC");
    }

    #[test]
    fn test_get_format_jsonl() {
        assert_eq!(get_format(Path::new("test.jsonl")), "JSON");
    }

    #[test]
    fn test_get_format_no_extension() {
        assert_eq!(get_format(Path::new("testfile")), "Unknown");
    }

    #[test]
    fn test_cmd_registry_search_no_results() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        // Init registry
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        // Search for something that doesn't exist
        let result = cmd_registry_search("nonexistent-dataset-xyz", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_push_with_long_description() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push with a very long description
        let long_desc = "This is a very long description that exceeds thirty characters and will be truncated in the list view";
        let result = cmd_registry_push(
            &input,
            "long-desc-test",
            "1.0.0",
            long_desc,
            "MIT",
            "",
            &registry_path,
        );
        assert!(result.is_ok());

        // List should truncate the description
        let result = cmd_registry_list(&registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info_with_all_metadata() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push with all metadata
        cmd_registry_push(
            &input,
            "full-metadata",
            "1.0.0",
            "Full metadata test",
            "Apache-2.0",
            "test,metadata,full",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        let result = cmd_registry_show_info("full-metadata", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_registry_new_directory() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("new_registry_dir");

        // Directory doesn't exist yet
        assert!(!registry_path.exists());

        let result = create_registry(&registry_path);
        assert!(result.is_ok());

        // Directory should now exist
        assert!(registry_path.exists());
    }

    #[test]
    fn test_load_dataset_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let csv_path = temp_dir.path().join("data.csv");

        create_test_parquet(&parquet_path, 10);

        // Convert to CSV first
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_csv(&csv_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write csv"));

        // Load from CSV
        let loaded = load_dataset(&csv_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_load_dataset_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let json_path = temp_dir.path().join("data.json");

        create_test_parquet(&parquet_path, 10);

        // Convert to JSON first
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_json(&json_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write json"));

        // Load from JSON
        let loaded = load_dataset(&json_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_load_dataset_jsonl() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let jsonl_path = temp_dir.path().join("data.jsonl");

        create_test_parquet(&parquet_path, 10);

        // Convert to JSON first (jsonl is same format)
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_json(&jsonl_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write jsonl"));

        // Load from JSONL
        let loaded = load_dataset(&jsonl_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_cmd_registry_delete_nonexistent() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        // Init registry
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        // Try to delete something that doesn't exist
        let result = cmd_registry_delete("nonexistent", "1.0.0", &registry_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_registry_pull_nonexistent() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let output = temp_dir.path().join("output.parquet");

        // Init registry
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        // Try to pull something that doesn't exist
        let result = cmd_registry_pull("nonexistent", &output, None, &registry_path);
        assert!(result.is_err());
    }

    // Drift CLI tests

    #[test]
    fn test_parse_drift_tests() {
        let tests = parse_drift_tests("ks,psi");
        assert_eq!(tests.len(), 2);

        let tests = parse_drift_tests("ks, chi2, psi, js");
        assert_eq!(tests.len(), 4);

        let tests = parse_drift_tests("invalid");
        assert!(tests.is_empty());

        let tests = parse_drift_tests("KS,PSI");
        assert_eq!(tests.len(), 2);
    }

    #[test]
    fn test_severity_symbol() {
        assert_eq!(severity_symbol(DriftSeverity::None), "✓");
        assert_eq!(severity_symbol(DriftSeverity::Low), "○");
        assert_eq!(severity_symbol(DriftSeverity::Medium), "●");
        assert_eq!(severity_symbol(DriftSeverity::High), "▲");
        assert_eq!(severity_symbol(DriftSeverity::Critical), "✖");
    }

    #[test]
    fn test_cmd_drift_detect_same_data() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        // Compare dataset with itself - should detect no drift
        let result = cmd_drift_detect(&path, &path, "ks,psi", 0.05, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_detect_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_drift_detect(&path, &path, "ks", 0.05, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_detect_invalid_tests() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_drift_detect(&path, &path, "invalid", 0.05, "text");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_report() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        // Report without output file (prints to stdout)
        let result = cmd_drift_report(&path, &path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_report_to_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let output_path = temp_dir.path().join("report.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_drift_report(&data_path, &data_path, Some(&output_path));
        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify JSON is valid
        let content = std::fs::read_to_string(&output_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert!(parsed.get("drift_detected").is_some());
    }

    // Quality CLI tests

    #[test]
    fn test_cmd_quality_check_text() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_check(&path, 0.1, 0.05, true, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_check_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_check(&path, 0.1, 0.05, true, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_check_no_outliers() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_check(&path, 0.1, 0.05, false, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_report(&path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_to_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let output_path = temp_dir.path().join("quality.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_quality_report(&data_path, Some(&output_path));
        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify JSON is valid
        let content = std::fs::read_to_string(&output_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert!(parsed.get("score").is_some());
        assert!(parsed.get("has_issues").is_some());
    }

    // Sketch CLI tests

    fn create_test_float_parquet(path: &PathBuf, rows: usize) {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let values: Vec<f64> = ids.iter().map(|i| *i as f64 * 1.5).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Float64Array::from(values)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));
    }

    #[test]
    fn test_parse_sketch_type() {
        assert!(matches!(
            parse_sketch_type("tdigest"),
            Some(crate::SketchType::TDigest)
        ));
        assert!(matches!(
            parse_sketch_type("TDIGEST"),
            Some(crate::SketchType::TDigest)
        ));
        assert!(matches!(
            parse_sketch_type("ddsketch"),
            Some(crate::SketchType::DDSketch)
        ));
        assert!(matches!(
            parse_sketch_type("DDSKETCH"),
            Some(crate::SketchType::DDSketch)
        ));
        assert!(parse_sketch_type("invalid").is_none());
    }

    #[test]
    fn test_cmd_drift_sketch_tdigest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "tdigest", None, "json");
        assert!(result.is_ok());
        assert!(sketch_path.exists());

        // Verify sketch file is valid JSON
        let content = std::fs::read_to_string(&sketch_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert!(parsed.get("sketch_type").is_some());
        assert!(parsed.get("row_count").is_some());
    }

    #[test]
    fn test_cmd_drift_sketch_ddsketch_type() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "ddsketch", None, "json");
        assert!(result.is_ok());
        assert!(sketch_path.exists());
    }

    #[test]
    fn test_cmd_drift_sketch_with_source() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 50);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "tdigest", Some("node-1"), "json");
        assert!(result.is_ok());

        // Verify source is in the output
        let content = std::fs::read_to_string(&sketch_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert_eq!(
            parsed.get("source").and_then(|v| v.as_str()),
            Some("node-1")
        );
    }

    #[test]
    fn test_cmd_drift_sketch_binary_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.bin");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "tdigest", None, "binary");
        assert!(result.is_ok());
        assert!(sketch_path.exists());

        // Binary file should exist and be non-empty
        let metadata = std::fs::metadata(&sketch_path)
            .ok()
            .unwrap_or_else(|| panic!("Should get metadata"));
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_cmd_drift_sketch_invalid_type() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "invalid", None, "json");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_merge_tdigest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        // Create two datasets
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");
        let merged = temp_dir.path().join("merged.json");

        create_test_float_parquet(&data1, 50);
        create_test_float_parquet(&data2, 50);

        // Create sketches
        cmd_drift_sketch(&data1, &sketch1, "tdigest", Some("node-1"), "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data2, &sketch2, "tdigest", Some("node-2"), "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        // Merge
        let sketches = vec![sketch1.clone(), sketch2.clone()];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_ok());
        assert!(merged.exists());

        // Verify merged sketch
        let content = std::fs::read_to_string(&merged)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert_eq!(parsed.get("row_count").and_then(|v| v.as_u64()), Some(100));
    }

    #[test]
    fn test_cmd_drift_merge_single_sketch() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch = temp_dir.path().join("sketch.json");
        let merged = temp_dir.path().join("merged.json");

        create_test_float_parquet(&data, 100);
        cmd_drift_sketch(&data, &sketch, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch"));

        // Merge single sketch
        let sketches = vec![sketch.clone()];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_merge_empty_fails() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let merged = temp_dir.path().join("merged.json");

        let sketches: Vec<PathBuf> = vec![];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_compare_no_drift() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");

        create_test_float_parquet(&data, 100);

        // Create two identical sketches
        cmd_drift_sketch(&data, &sketch1, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data, &sketch2, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        let result = cmd_drift_compare(&sketch1, &sketch2, 0.1, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_compare_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");

        create_test_float_parquet(&data, 100);

        cmd_drift_sketch(&data, &sketch1, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data, &sketch2, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        let result = cmd_drift_compare(&sketch1, &sketch2, 0.1, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_compare_with_different_data() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        // Create two different datasets
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");

        // First dataset: values 0-149
        create_test_float_parquet(&data1, 100);

        // Second dataset: values with different distribution
        {
            use arrow::array::Float64Array;

            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("value", DataType::Float64, false),
            ]));

            let ids: Vec<i32> = (1000..1100).collect();
            let values: Vec<f64> = ids.iter().map(|i| *i as f64 * 10.0).collect();

            let batch = arrow::array::RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(Float64Array::from(values)),
                ],
            )
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

            let dataset = ArrowDataset::from_batch(batch)
                .ok()
                .unwrap_or_else(|| panic!("Should create dataset"));

            dataset
                .to_parquet(&data2)
                .ok()
                .unwrap_or_else(|| panic!("Should write parquet"));
        }

        cmd_drift_sketch(&data1, &sketch1, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data2, &sketch2, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        // Compare - should work (drift may or may not be detected depending on
        // threshold)
        let result = cmd_drift_compare(&sketch1, &sketch2, 0.1, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_merge_binary_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch = temp_dir.path().join("sketch.bin");
        let merged = temp_dir.path().join("merged.bin");

        create_test_float_parquet(&data, 100);
        cmd_drift_sketch(&data, &sketch, "tdigest", None, "binary")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch"));

        let sketches = vec![sketch.clone()];
        let result = cmd_drift_merge(&sketches, &merged, "binary");
        assert!(result.is_ok());
        assert!(merged.exists());
    }

    // Federated CLI tests

    #[test]
    fn test_parse_fed_strategy() {
        assert!(matches!(
            parse_fed_strategy("local", 0.8, 42, None),
            Some(_)
        ));
        assert!(matches!(
            parse_fed_strategy("proportional", 0.8, 42, None),
            Some(_)
        ));
        assert!(matches!(
            parse_fed_strategy("stratified", 0.8, 42, Some("label")),
            Some(_)
        ));
        assert!(parse_fed_strategy("invalid", 0.8, 42, None).is_none());
    }

    #[test]
    fn test_cmd_fed_manifest_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json");
        assert!(result.is_ok());
        assert!(manifest_path.exists());

        // Verify manifest contents
        let content = std::fs::read_to_string(&manifest_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert_eq!(
            parsed.get("node_id").and_then(|v| v.as_str()),
            Some("node-1")
        );
        assert_eq!(parsed.get("total_rows").and_then(|v| v.as_u64()), Some(100));
    }

    #[test]
    fn test_cmd_fed_manifest_binary() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.bin");
        create_test_parquet(&data_path, 50);

        let result = cmd_fed_manifest(&data_path, &manifest_path, "node-2", 0.8, 42, "binary");
        assert!(result.is_ok());
        assert!(manifest_path.exists());
    }

    #[test]
    fn test_cmd_fed_plan_local_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        // Create manifests for two nodes
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let manifest1 = temp_dir.path().join("manifest1.json");
        let manifest2 = temp_dir.path().join("manifest2.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data1, 100);
        create_test_parquet(&data2, 150);

        cmd_fed_manifest(&data1, &manifest1, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest1"));
        cmd_fed_manifest(&data2, &manifest2, "node-2", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest2"));

        let manifests = vec![manifest1.clone(), manifest2.clone()];
        let result = cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json");
        assert!(result.is_ok());
        assert!(plan_path.exists());

        // Verify plan contents
        let content = std::fs::read_to_string(&plan_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        let instructions = parsed.as_array();
        assert!(instructions.is_some());
        assert_eq!(instructions.map(|a| a.len()), Some(2));
    }

    #[test]
    fn test_cmd_fed_plan_empty_manifests_fails() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let plan_path = temp_dir.path().join("plan.json");

        let manifests: Vec<PathBuf> = vec![];
        let result = cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_split_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");
        let train_path = temp_dir.path().join("train.parquet");
        let test_path = temp_dir.path().join("test.parquet");

        create_test_parquet(&data_path, 100);

        // Create manifest
        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        // Create plan
        let manifests = vec![manifest_path.clone()];
        cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create plan"));

        // Execute split
        let result = cmd_fed_split(
            &data_path,
            &plan_path,
            "node-1",
            &train_path,
            &test_path,
            None,
        );
        assert!(result.is_ok());
        assert!(train_path.exists());
        assert!(test_path.exists());

        // Verify split sizes
        let train_ds = ArrowDataset::from_parquet(&train_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load train"));
        let test_ds = ArrowDataset::from_parquet(&test_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load test"));

        assert!(train_ds.len() > 0);
        assert!(test_ds.len() > 0);
        assert_eq!(train_ds.len() + test_ds.len(), 100);
    }

    #[test]
    fn test_cmd_fed_split_node_not_found() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");
        let train_path = temp_dir.path().join("train.parquet");
        let test_path = temp_dir.path().join("test.parquet");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create plan"));

        // Try to split with wrong node ID
        let result = cmd_fed_split(
            &data_path,
            &plan_path,
            "wrong-node",
            &train_path,
            &test_path,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_verify_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let manifest1 = temp_dir.path().join("manifest1.json");
        let manifest2 = temp_dir.path().join("manifest2.json");

        create_test_parquet(&data1, 100);
        create_test_parquet(&data2, 150);

        cmd_fed_manifest(&data1, &manifest1, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest1"));
        cmd_fed_manifest(&data2, &manifest2, "node-2", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest2"));

        let manifests = vec![manifest1.clone(), manifest2.clone()];
        let result = cmd_fed_verify(&manifests, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_fed_verify_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_verify(&manifests, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_fed_verify_empty_manifests_fails() {
        let manifests: Vec<PathBuf> = vec![];
        let result = cmd_fed_verify(&manifests, "text");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_plan_proportional_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_plan(
            &manifests,
            &plan_path,
            "proportional",
            0.7,
            42,
            None,
            "json",
        );
        assert!(result.is_ok());

        // Verify train ratio is 0.7
        let content = std::fs::read_to_string(&plan_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        let instructions = parsed
            .as_array()
            .unwrap_or_else(|| panic!("Should be array"));
        let train_ratio = instructions[0].get("train_ratio").and_then(|v| v.as_f64());
        assert!((train_ratio.unwrap_or(0.0) - 0.7).abs() < 0.01);
    }

    // Additional error path tests for coverage

    #[test]
    fn test_load_sketch_invalid_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let sketch_path = temp_dir.path().join("invalid.json");

        std::fs::write(&sketch_path, "{ invalid json }")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_sketch(&sketch_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_manifest_invalid_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let manifest_path = temp_dir.path().join("invalid.json");

        std::fs::write(&manifest_path, "not valid json")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_manifest(&manifest_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_plan_invalid_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let plan_path = temp_dir.path().join("invalid.json");

        std::fs::write(&plan_path, "{ broken }")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_plan(&plan_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_plan_invalid_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_plan(
            &manifests,
            &plan_path,
            "invalid_strategy",
            0.8,
            42,
            None,
            "json",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_plan_stratified_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_plan(
            &manifests,
            &plan_path,
            "stratified",
            0.8,
            42,
            Some("name"),
            "json",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert_unsupported_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.xyz");
        create_test_parquet(&input, 10);

        let result = cmd_convert(&input, &output);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_dataset_xyz_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.xyz");

        std::fs::write(&path, "some data")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_dataset(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_compare_text_with_drift() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");

        create_test_float_parquet(&data1, 100);

        // Second dataset with very different values
        {
            use arrow::array::Float64Array;

            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("value", DataType::Float64, false),
            ]));

            let ids: Vec<i32> = (0..100).collect();
            let values: Vec<f64> = ids.iter().map(|i| (*i as f64 + 1000.0) * 100.0).collect();

            let batch = arrow::array::RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(Float64Array::from(values)),
                ],
            )
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

            let dataset = ArrowDataset::from_batch(batch)
                .ok()
                .unwrap_or_else(|| panic!("Should create dataset"));

            dataset
                .to_parquet(&data2)
                .ok()
                .unwrap_or_else(|| panic!("Should write parquet"));
        }

        cmd_drift_sketch(&data1, &sketch1, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data2, &sketch2, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        let result = cmd_drift_compare(&sketch1, &sketch2, 0.001, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_check_with_constant_column() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("constant", DataType::Int32, false),
        ]));

        let ids: Vec<i32> = (0..100).collect();
        let constants: Vec<i32> = vec![42; 100];

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(constants)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let result = cmd_quality_check(&path, 0.1, 0.05, true, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_fed_verify_with_quality_issues() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("small.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");

        create_test_parquet(&data_path, 15);

        cmd_fed_manifest(&data_path, &manifest_path, "small-node", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_verify(&manifests, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_format_arrow() {
        assert_eq!(get_format(Path::new("test.arrow")), "Arrow IPC");
    }

    #[test]
    fn test_get_format_unknown() {
        assert_eq!(get_format(Path::new("test.feather")), "Unknown");
        assert_eq!(get_format(Path::new("test.txt")), "Unknown");
    }

    #[test]
    fn test_cmd_drift_detect_all_tests() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_drift_detect(&path, &path, "ks,chi2,psi,js", 0.05, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_detect_with_drift() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let ref_path = temp_dir.path().join("ref.parquet");
        let cur_path = temp_dir.path().join("cur.parquet");

        create_test_parquet(&ref_path, 100);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (500..600).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("different_{}", i)).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(&cur_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let result = cmd_drift_detect(&ref_path, &cur_path, "ks,psi", 0.05, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_compare_empty_results() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        // Create dataset with only string column (no numeric for sketch)
        let data_path = temp_dir.path().join("strings.parquet");
        let sketch1_path = temp_dir.path().join("sketch1.json");
        let sketch2_path = temp_dir.path().join("sketch2.json");

        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));

        let names: Vec<String> = (0..50).map(|i| format!("name_{}", i)).collect();

        let batch =
            arrow::array::RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(names))])
                .ok()
                .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(&data_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        cmd_drift_sketch(&data_path, &sketch1_path, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data_path, &sketch2_path, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        let result = cmd_drift_compare(&sketch1_path, &sketch2_path, 0.1, "text");
        assert!(result.is_ok());
    }

    // Additional coverage tests

    #[test]
    fn test_parse_drift_tests_single() {
        let tests = parse_drift_tests("ks");
        assert_eq!(tests.len(), 1);
        assert!(matches!(tests[0], DriftTest::KolmogorovSmirnov));
    }

    #[test]
    fn test_parse_drift_tests_multiple() {
        let tests = parse_drift_tests("ks,chi2,psi,js");
        assert_eq!(tests.len(), 4);
    }

    #[test]
    fn test_parse_drift_tests_with_spaces() {
        let tests = parse_drift_tests("ks, chi2 ,psi");
        assert_eq!(tests.len(), 3);
    }

    #[test]
    fn test_parse_drift_tests_unknown() {
        let tests = parse_drift_tests("unknown");
        assert!(tests.is_empty());
    }

    #[test]
    fn test_parse_sketch_type_tdigest() {
        assert!(matches!(parse_sketch_type("tdigest"), Some(SketchType::TDigest)));
    }

    #[test]
    fn test_parse_sketch_type_ddsketch() {
        assert!(matches!(parse_sketch_type("ddsketch"), Some(SketchType::DDSketch)));
    }

    #[test]
    fn test_parse_sketch_type_unknown() {
        assert!(parse_sketch_type("unknown").is_none());
    }

    #[test]
    fn test_save_dataset_parquet() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        ).unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.csv");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        ).unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.json");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        ).unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_unknown_extension() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.xyz");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        ).unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_quality_report_default_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 50);

        let result = cmd_quality_report(&path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_with_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        let output = temp_dir.path().join("report.html");
        create_test_parquet(&path, 50);

        let result = cmd_quality_report(&path, Some(&output));
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_quality_score() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", false, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_badge() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", false, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_suggest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", true, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_doctest_profile() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "doctest-corpus", false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_profiles() {
        let result = cmd_quality_profiles();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_fed_strategy_iid() {
        let result = parse_fed_strategy("iid", 0.8, 42, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_proportional() {
        let result = parse_fed_strategy("proportional", 0.8, 42, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_local() {
        let result = parse_fed_strategy("local", 0.8, 42, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_stratified() {
        let result = parse_fed_strategy("stratified", 0.8, 42, Some("label"));
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_unknown() {
        assert!(parse_fed_strategy("invalid", 0.8, 42, None).is_none());
    }

    #[test]
    fn test_cmd_convert_to_csv_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.csv");
        create_test_parquet(&input, 20);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_convert_to_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.json");
        create_test_parquet(&input, 20);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_head_more_than_available() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("small.parquet");
        create_test_parquet(&path, 5);

        // Request more rows than available
        let result = cmd_head(&path, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_report_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let ref_path = temp_dir.path().join("ref.parquet");
        let cur_path = temp_dir.path().join("cur.parquet");
        create_test_parquet(&ref_path, 100);
        create_test_parquet(&cur_path, 100);

        let result = cmd_drift_report(&ref_path, &cur_path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_report_with_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let ref_path = temp_dir.path().join("ref.parquet");
        let cur_path = temp_dir.path().join("cur.parquet");
        let output = temp_dir.path().join("report.html");
        create_test_parquet(&ref_path, 100);
        create_test_parquet(&cur_path, 100);

        let result = cmd_drift_report(&ref_path, &cur_path, Some(&output));
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_drift_sketch_ddsketch_json_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "ddsketch", None, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_sketch_with_column() {
        use arrow::array::Float64Array;

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");

        // Create dataset with numeric column
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
        ]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        ).unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();
        dataset.to_parquet(&data_path).unwrap();

        let result = cmd_drift_sketch(&data_path, &sketch_path, "tdigest", Some("value"), "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_search_with_data() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let data_path = temp_dir.path().join("data.parquet");

        create_test_parquet(&data_path, 20);

        // Push a dataset first
        cmd_registry_push(
            &data_path,
            "searchable-data",
            "1.0.0",
            "Dataset for search test",
            "MIT",
            "search,test",
            &registry_path,
        ).unwrap();

        let result = cmd_registry_search("search", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_search_empty_results() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        cmd_registry_init(&registry_path).unwrap();

        let result = cmd_registry_search("nonexistent", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let data_path = temp_dir.path().join("data.parquet");

        create_test_parquet(&data_path, 20);

        cmd_registry_push(
            &data_path,
            "info-dataset",
            "1.0.0",
            "Dataset for info test",
            "Apache-2.0",
            "info,test",
            &registry_path,
        ).unwrap();

        let result = cmd_registry_show_info("info-dataset", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info_not_found() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        cmd_registry_init(&registry_path).unwrap();

        let result = cmd_registry_show_info("nonexistent", &registry_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_registry_delete_existing() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let data_path = temp_dir.path().join("data.parquet");

        create_test_parquet(&data_path, 20);

        cmd_registry_push(
            &data_path,
            "delete-test",
            "1.0.0",
            "Dataset to delete",
            "MIT",
            "delete,test",
            &registry_path,
        ).unwrap();

        let result = cmd_registry_delete("delete-test", "1.0.0", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_dataset_csv_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let csv_path = temp_dir.path().join("test.csv");

        // Create a simple CSV
        std::fs::write(&csv_path, "id,name\n1,foo\n2,bar\n").unwrap();

        let result = load_dataset(&csv_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_dataset_json_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let json_path = temp_dir.path().join("test.json");

        // Create a simple JSON Lines file
        std::fs::write(&json_path, r#"{"id":1,"name":"foo"}
{"id":2,"name":"bar"}"#).unwrap();

        let result = load_dataset(&json_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_fed_plan() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let manifest1 = temp_dir.path().join("node1.json");
        let manifest2 = temp_dir.path().join("node2.json");
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let output = temp_dir.path().join("plan.json");

        create_test_parquet(&data1, 50);
        create_test_parquet(&data2, 50);

        cmd_fed_manifest(&data1, &manifest1, "node1", 0.8, 42, "json").unwrap();
        cmd_fed_manifest(&data2, &manifest2, "node2", 0.8, 42, "json").unwrap();

        let manifests = vec![manifest1, manifest2];
        let result = cmd_fed_plan(&manifests, &output, "iid", 0.8, 42, None, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_merge() {
        use arrow::array::Float64Array;

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");
        let merged = temp_dir.path().join("merged.json");

        // Create datasets with float column
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
        ]));

        let batch1 = arrow::array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        ).unwrap();
        let dataset1 = ArrowDataset::from_batch(batch1).unwrap();
        dataset1.to_parquet(&data1).unwrap();

        let batch2 = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![6.0, 7.0, 8.0, 9.0, 10.0]))],
        ).unwrap();
        let dataset2 = ArrowDataset::from_batch(batch2).unwrap();
        dataset2.to_parquet(&data2).unwrap();

        cmd_drift_sketch(&data1, &sketch1, "tdigest", None, "json").unwrap();
        cmd_drift_sketch(&data2, &sketch2, "tdigest", None, "json").unwrap();

        let sketches = vec![sketch1, sketch2];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_ok());
    }
}
