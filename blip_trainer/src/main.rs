use blip_ai::model::Model;
use blip_ai::trainer::TrainingData;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name="Blip Trainer", author = "J-F Bilodeau (jfbilodeau@chronogears.com)", version = "1.0", about, long_about = None)]
struct TrainingArgs {
    #[arg(short = 'e', long, default_value = "256", help = "Embedding dimension (size of the embedding vector)")]
    pub embedding_dim: usize,
    #[arg(short = 'd', long, default_value = "8", help = "Neural network depth (number of layers)")]
    pub depth: usize,
    #[arg(short = 'n', long = "epochs", default_value = "512", help = "Number of epochs to train the model")]
    pub num_epochs: usize,
    #[arg(short = 'l', long, default_value = "0.001", help = "Learning rate for training the model")]
    pub learning_rate: f32,
    #[arg(short = 'b', long, default_value = "32", help = "Batch size for training the model")]
    pub batch_size: usize,
    #[arg(short = 'g', long, default_value = "3", help = "N-gram size for training")]
    pub ngram_size: usize,
    #[arg(short = 'i', long, default_values = vec!["training/basic.txt"], help = "Input files for training the model")]
    pub input_files: Vec<String>,
    #[arg(short = 'o', long, default_value = "models/basic.json", help = "Output file to save the trained model")]
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
    println!(" - Neural Network Depth: {}", args.depth);
    println!(" - Number of epochs: {}", args.num_epochs);
    println!(" - Learning rate: {}", args.learning_rate);
    println!(" - Batch size: {}", args.batch_size);
    println!(" - N-gram size: {}", args.ngram_size);
    println!(" - Training input files: {:?}", args.input_files);
    println!(" - Output file: {}", args.output_file);

    println!();

    let program_start = std::time::Instant::now();

    println!("Creating model...");
    let start = std::time::Instant::now();

    let mut model = Model::new(
        args.embedding_dim,
        args.depth
    );

    println!("Model created in {} seconds", start.elapsed().as_secs_f32());

    println!("Loading training data...");
    let start = std::time::Instant::now();
    let mut training_data = TrainingData::new(model);
    for file_name in &args.input_files {
        if let Err(e) = training_data.load(file_name) {
            eprintln!("Error loading training data from {}: {}", file_name, e);
            return;
        }
    }
    println!("Model loaded in {} seconds", start.elapsed().as_secs_f32());

    println!("Initializing embeddings...");
    let start = std::time::Instant::now();
    training_data.get_model_mut().initialize_embeddings();
    println!(
        "Embeddings initialized in {} seconds",
        start.elapsed().as_secs_f32()
    );

    println!("Training multi-head attention...");
    let start = std::time::Instant::now();
    training_data.train_multi_head_attention(args.num_epochs, args.learning_rate);
    println!(
        "Multi-head attention trained in {} seconds",
        start.elapsed().as_secs_f32()
    );

    println!("Training neural network...");
    let start = std::time::Instant::now();
    training_data.train_neural_network(args.num_epochs, args.learning_rate, args.batch_size);
    println!(
        "Neural network trained in {} seconds",
        start.elapsed().as_secs_f32()
    );

    println!("Saving model...");
    let start = std::time::Instant::now();
    if let Err(e) = training_data.get_model().save(&args.output_file) {
        eprintln!("Error saving model to {}: {}", args.output_file, e);
        return;
    }
    println!("Model saved in {} seconds", start.elapsed().as_secs_f32());

    println!(
        "Training completed in {} seconds",
        program_start.elapsed().as_secs_f32()
    );
}
