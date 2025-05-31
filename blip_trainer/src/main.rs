use blip_ai::model::Model;
use clap::Parser;
use crate::trainer::{create_training_data, load_training_data};

mod trainer;

#[derive(Parser, Debug)]
#[command(name="Blip Trainer", author = "J-F Bilodeau (jfbilodeau@chronogears.com)", version = "1.0", about, long_about = None)]
struct TrainingArgs {
    #[arg(short = 'e', long, default_value = "256")]
    pub embedding_dim: usize,
    #[arg(short = 'n', long, default_value = "512")]
    pub num_epochs: usize,
    #[arg(short, long, default_value = "0.001")]
    pub learning_rate: f32,
    #[arg(short = 'g', long, default_value = "3")]
    pub ngram_size: usize,
    #[arg(short = 'i', long, default_values = vec!["training/basic.txt"])]
    pub input_files: Vec<String>,
    #[arg(short = 'o', long, default_value = "models/basic.json")]
    pub output_file: String,
}

impl std::fmt::Display for TrainingArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TrainingArgs {{ embedding_dim: {}, num_epochs: {}, learning_rate: {}, ngram_size: {} }}",
            self.embedding_dim, self.num_epochs, self.learning_rate, self.ngram_size
        )
    }
}

fn main() {
    println!("Blip trainer started...");

    let args = TrainingArgs::parse();

    println!("Using the following parameters");
    println!(" - Embedding size (dimension): {}", args.embedding_dim);
    println!(" - Number of epochs: {}", args.num_epochs);
    println!(" - Learning rate: {}", args.learning_rate);
    println!(" - N-gram size: {}", args.ngram_size);
    println!(" - Training input files: {:?}", args.input_files);
    println!(" - Output file: {}", args.output_file);

    println!();

    println!("Creating model...");
    let start = std::time::Instant::now();

    let mut model = Model::new(
        args.embedding_dim,
    );

    println!("Model created in {} seconds", start.elapsed().as_secs_f32());


    println!("Loading training data...");
    let start = std::time::Instant::now();
    let mut training_data = create_training_data();
    for file_name in &args.input_files {
        if let Err(e) = trainer::load_training_data(file_name, &mut model, &mut training_data) {
            eprintln!("Error loading training data from {}: {}", file_name, e);
            return;
        }
    }
    println!("Model loaded in {} seconds", start.elapsed().as_secs_f32());

    println!("Initializing embeddings...");
    let start = std::time::Instant::now();
    model.initialize_embeddings();
    println!("Embeddings initialized in {} seconds", start.elapsed().as_secs_f32());

    println!("Saving model...");
    let start = std::time::Instant::now();
    if let Err(e) = model.save(&args.output_file) {
        eprintln!("Error saving model to {}: {}", args.output_file, e);
        return;
    }

    println!("Done!");
}
