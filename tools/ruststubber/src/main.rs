mod stubber;

use clap::Parser;
use std::fs;
use std::process;

#[derive(Parser)]
#[command(name = "ruststubber", about = "Replace Rust function bodies with todo!(\"STUB\")")]
struct Cli {
    #[arg(long, help = "Input Rust source file")]
    input: String,
    #[arg(long, help = "Output file (defaults to stdout)")]
    output: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let source = match fs::read_to_string(&cli.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read {}: {}", cli.input, e);
            process::exit(1);
        }
    };

    let result = match stubber::stub_file(&source) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to parse {}: {}", cli.input, e);
            process::exit(1);
        }
    };

    match cli.output {
        Some(path) => {
            if let Err(e) = fs::write(&path, &result) {
                eprintln!("Failed to write {}: {}", path, e);
                process::exit(1);
            }
        }
        None => print!("{}", result),
    }
}
