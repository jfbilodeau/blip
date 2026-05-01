use blip_ai::model::Model;
use blip_ai::nn;
use blip_ai::trainer::{LrSchedule, TrainingConfig, TrainingData};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "Blip Trainer",
    author = "J-F Bilodeau (jfbilodeau@chronogears.com)",
    version,
    about,
    long_about = None
)]
struct TrainingArgs {
    #[arg(short = 'e', long, default_value = "128", help = "Embedding dimension")]
    pub embedding_dim: usize,

    #[arg(short = 'd', long, default_value = "2", help = "Number of decoder blocks")]
    pub depth: usize,

    #[arg(long, default_value = "4", help = "Number of attention heads")]
    pub n_heads: usize,

    #[arg(short = 'n', long = "epochs", default_value = "200")]
    pub num_epochs: usize,

    #[arg(short = 'l', long, default_value = "0.001", help = "Adam learning rate")]
    pub learning_rate: f32,

    #[arg(short = 'b', long, default_value = "1", help = "Sequences per optimizer step")]
    pub batch_size: usize,

    #[arg(long, default_value = "0.0", help = "Dropout on attention/FFN outputs during training")]
    pub dropout: f32,

    #[arg(long, default_value = "0.0", help = "Validation split fraction (0..1)")]
    pub val_split: f32,

    #[arg(long, default_value = "0", help = "Warmup steps for cosine LR schedule (0 = constant LR)")]
    pub warmup_steps: usize,

    #[arg(long, default_value = "0.0", help = "Minimum LR reached by cosine decay after warmup")]
    pub min_lr: f32,

    #[arg(long, default_value = "0", help = "Random seed (0 = OS entropy)")]
    pub seed: u64,

    #[arg(long, default_value = "0", help = "Save checkpoint every N epochs (0 = end only)")]
    pub checkpoint_every: usize,

    #[arg(long, default_value = "1", help = "Drop tokens whose usage_count is below this (specials always kept)")]
    pub min_count: u32,

    #[arg(short = 'i', long, default_values = vec!["training/basic.txt"])]
    pub input_files: Vec<String>,

    #[arg(short = 'o', long, default_value = "models/basic.json",
          help = "Output path. .json => JSON, otherwise bincode.")]
    pub output_file: String,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = TrainingArgs::parse();

    if args.seed != 0 {
        nn::seed(args.seed);
    }

    println!("Blip trainer");
    println!(" - embedding_dim: {}", args.embedding_dim);
    println!(" - depth:         {}", args.depth);
    println!(" - n_heads:       {}", args.n_heads);
    println!(" - epochs:        {}", args.num_epochs);
    println!(" - learning_rate: {}", args.learning_rate);
    println!(" - batch_size:    {}", args.batch_size);
    println!(" - dropout:       {}", args.dropout);
    println!(" - val_split:     {}", args.val_split);
    println!(" - warmup_steps:  {}", args.warmup_steps);
    println!(" - min_lr:        {}", args.min_lr);
    println!(" - seed:          {}", args.seed);
    println!(" - inputs:        {:?}", args.input_files);
    println!(" - output:        {}", args.output_file);
    println!();

    let program_start = std::time::Instant::now();

    let model = Model::new(args.embedding_dim, args.depth, args.n_heads);
    let mut training_data = TrainingData::new(model);
    for file_name in &args.input_files {
        if let Err(e) = training_data.load(file_name) {
            eprintln!("Error loading training data from {}: {}", file_name, e);
            return;
        }
    }
    println!(
        "Loaded {} prompts, vocab = {}",
        training_data.num_prompts(),
        training_data.get_model().vocab_size()
    );

    if args.min_count > 1 {
        let removed = training_data.trim_vocab(args.min_count);
        println!(
            "Trimmed {} rare tokens (min_count={}), vocab = {}",
            removed,
            args.min_count,
            training_data.get_model().vocab_size()
        );
    }

    training_data.get_model_mut().initialize_embeddings();

    let lr_schedule = if args.warmup_steps > 0 {
        LrSchedule::CosineWithWarmup {
            warmup_steps: args.warmup_steps,
            min_lr: args.min_lr,
        }
    } else {
        LrSchedule::Constant
    };

    let cfg = TrainingConfig {
        num_epochs: args.num_epochs,
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        val_split: args.val_split,
        seed: args.seed,
        checkpoint_every: args.checkpoint_every,
        checkpoint_path: Some(args.output_file.clone()),
        lr_schedule,
        dropout: args.dropout,
    };

    let train_start = std::time::Instant::now();
    training_data.train(&cfg);
    println!("Training completed in {:.2}s", train_start.elapsed().as_secs_f32());

    if let Err(e) = training_data.get_model().save(&args.output_file) {
        eprintln!("Error saving model to {}: {}", args.output_file, e);
        return;
    }
    println!(
        "Total runtime {:.2}s, model saved to {}",
        program_start.elapsed().as_secs_f32(),
        args.output_file
    );
}
