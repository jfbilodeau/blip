use blip_ai::model::{Model, TrainingMetadata};
use blip_ai::nn;
use blip_ai::pretraining::PretrainingData;
use blip_ai::trainer::{LrSchedule, StopTokenMode, TrainingConfig, TrainingData};
use clap::Parser;
use std::collections::HashSet;
use std::path::Path;

#[cfg(windows)]
struct SleepGuard;

#[cfg(windows)]
impl SleepGuard {
    fn new() -> Self {
        use windows_sys::Win32::System::Power::{
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED, SetThreadExecutionState,
        };

        unsafe {
            SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
        }
        Self
    }
}

#[cfg(windows)]
impl Drop for SleepGuard {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Power::{ES_CONTINUOUS, SetThreadExecutionState};

        unsafe {
            SetThreadExecutionState(ES_CONTINUOUS);
        }
    }
}

#[cfg(not(windows))]
struct SleepGuard;

#[cfg(not(windows))]
impl SleepGuard {
    fn new() -> Self {
        Self
    }
}

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

    #[arg(short = 'd', long, default_value = "4", help = "Number of decoder blocks")]
    pub depth: usize,

    #[arg(long, default_value = "4", help = "Number of attention heads")]
    pub n_heads: usize,

    // ---- Pretraining-phase config -----------------------------------------
    #[arg(long = "pretrain-epochs", default_value = "4", help = "Pretraining epochs over the corpus")]
    pub pretrain_epochs: usize,

    #[arg(long = "pretrain-lr", default_value = "0.0005", help = "Pretraining Adam learning rate")]
    pub pretrain_lr: f32,

    #[arg(long = "pretrain-batch-size", default_value = "128", help = "Pretraining sequences per optimizer step")]
    pub pretrain_batch_size: usize,

    #[arg(long = "pretrain-warmup", default_value = "200", help = "Pretraining warmup steps (0 = constant LR)")]
    pub pretrain_warmup: usize,

    #[arg(long = "pretrain-min-lr", default_value = "0.00001", help = "Pretraining cosine-decay floor")]
    pub pretrain_min_lr: f32,

    // ---- Tuning-phase config ----------------------------------------------
    #[arg(short = 'n', long = "epochs", default_value = "60", help = "Tuning epochs over the chat dataset")]
    pub num_epochs: usize,

    #[arg(short = 'l', long, default_value = "0.001", help = "Tuning Adam learning rate")]
    pub learning_rate: f32,

    #[arg(short = 'b', long, default_value = "128", help = "Tuning sequences per optimizer step")]
    pub batch_size: usize,

    #[arg(long, default_value = "0.10", help = "Dropout on attention/FFN outputs during training")]
    pub dropout: f32,

    #[arg(long, default_value = "0.10", help = "Validation split fraction (0..1)")]
    pub val_split: f32,

    #[arg(long, default_value = "50", help = "Tuning warmup steps for cosine LR schedule (0 = constant LR)")]
    pub warmup_steps: usize,

    #[arg(long, default_value = "0.0001", help = "Tuning minimum LR reached by cosine decay after warmup")]
    pub min_lr: f32,

    #[arg(long, default_value = "1.0", help = "Global L2 gradient clipping threshold (<=0 disables)")]
    pub grad_clip: f32,

    #[arg(long, default_value = "false", help = "Disable rayon gradient accumulation for strict deterministic training")]
    pub deterministic: bool,

    #[arg(long, default_value = "42", help = "Random seed (0 = OS entropy)")]
    pub seed: u64,

    #[arg(long, default_value = "10", help = "Save checkpoint every N epochs (0 = end only)")]
    pub checkpoint_every: usize,

    #[arg(long, default_value = "1", help = "Drop corpus (pre-training) tokens whose usage_count is below this (specials always kept)")]
    pub min_count: u32,

    #[arg(long, default_value = "256", help = "Target sequence length (tokens) for pretraining corpus loader")]
    pub seq_length: usize,

    #[arg(short = 'p', long, default_values = vec!["training/pretraining/*"], help = "Pretraining (corpus) files to use. Wild cards are supported but does not recurse in subdirectories.")]
    pub pretraining_files: Vec<String>,

    #[arg(short = 't', long = "tuning-files", default_values = vec!["training/tuning/*"], help = "Tuning files to use. Wild cards are supported but does not recurse in subdirectories.")]
    pub tuning_files: Vec<String>,

    #[arg(short = 'o', long, default_value = "models/basic.json", help = "Output path (.json only).")]
    pub output_file: String,

    #[arg(long = "resume-from", help = "Resume training from an existing .json checkpoint")]
    pub resume_from: Option<String>,

    #[arg(long, default_value = "false", help = "When resuming, skip the pretraining phase and run tuning only")]
    pub resume_skip_pretraining: bool,
}

fn has_glob_meta(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?') || pattern.contains('[')
}

fn expand_input_files(patterns: &[String], kind: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for pattern in patterns {
        if has_glob_meta(pattern) {
            let mut matched_any = false;
            let entries = glob::glob(pattern)
                .map_err(|e| format!("Invalid {} wildcard pattern '{}': {}", kind, pattern, e))?;
            for entry in entries {
                let path = entry.map_err(|e| {
                    format!(
                        "Error expanding {} wildcard pattern '{}': {}",
                        kind, pattern, e
                    )
                })?;
                if path.is_file() {
                    matched_any = true;
                    let file_name = path.to_string_lossy().to_string();
                    if seen.insert(file_name.clone()) {
                        out.push(file_name);
                    }
                }
            }
            if !matched_any {
                return Err(format!(
                    "No {} files matched wildcard pattern '{}'",
                    kind, pattern
                ));
            }
        } else {
            let path = Path::new(pattern);
            if !path.is_file() {
                return Err(format!("{} file '{}' does not exist", kind, pattern));
            }
            if seen.insert(pattern.clone()) {
                out.push(pattern.clone());
            }
        }
    }

    out.sort();
    Ok(out)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = TrainingArgs::parse();

    let pretraining_files = match expand_input_files(&args.pretraining_files, "pretraining") {
        Ok(files) => files,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    let tuning_files = match expand_input_files(&args.tuning_files, "tuning") {
        Ok(files) => files,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };

    if args.seed != 0 {
        nn::seed(args.seed);
    }

    println!("Blip trainer");
    println!(" - embedding_dim:        {}", args.embedding_dim);
    println!(" - depth:                {}", args.depth);
    println!(" - n_heads:              {}", args.n_heads);
    println!(" - pretrain epochs:      {}", args.pretrain_epochs);
    println!(" - pretrain lr:          {}", args.pretrain_lr);
    println!(" - pretrain batch_size:  {}", args.pretrain_batch_size);
    println!(" - pretrain warmup:      {}", args.pretrain_warmup);
    println!(" - pretrain min_lr:      {}", args.pretrain_min_lr);
    println!(" - tune epochs:          {}", args.num_epochs);
    println!(" - tune learning_rate:   {}", args.learning_rate);
    println!(" - tune batch_size:      {}", args.batch_size);
    println!(" - tune warmup_steps:    {}", args.warmup_steps);
    println!(" - tune min_lr:          {}", args.min_lr);
    println!(" - grad_clip:            {}", args.grad_clip);
    println!(" - deterministic:        {}", args.deterministic);
    println!(" - dropout:              {}", args.dropout);
    println!(" - val_split:            {}", args.val_split);
    println!(" - seed:                 {}", args.seed);
    println!(" - min_count:            {}", args.min_count);
    println!(" - seq_length:           {} (pretraining)", args.seq_length);
    println!(" - pretraining:          {:?}", pretraining_files);
    println!(" - tuning:               {:?}", tuning_files);
    println!(" - resume_from:          {:?}", args.resume_from);
    println!(" - resume_skip_pretrain: {}", args.resume_skip_pretraining);
    println!(" - output:               {}", args.output_file);
    println!();

    let program_start = std::time::Instant::now();

    let resuming = args.resume_from.is_some();
    let mut model = if let Some(checkpoint_path) = args.resume_from.as_deref() {
        match Model::load(checkpoint_path) {
            Ok(loaded) => {
                if loaded.embedding_dim() != args.embedding_dim
                    || loaded.depth() != args.depth
                    || loaded.n_heads() != args.n_heads
                {
                    println!(
                        "Warning: checkpoint architecture ({}, {}, {}) differs from CLI ({}, {}, {}). Using checkpoint architecture.",
                        loaded.embedding_dim(),
                        loaded.depth(),
                        loaded.n_heads(),
                        args.embedding_dim,
                        args.depth,
                        args.n_heads
                    );
                }
                println!(
                    "Resuming from checkpoint: {} (vocab={}, embed_dim={}, depth={}, heads={})",
                    checkpoint_path,
                    loaded.vocab_size(),
                    loaded.embedding_dim(),
                    loaded.depth(),
                    loaded.n_heads()
                );
                loaded
            }
            Err(e) => {
                eprintln!("Error loading checkpoint from {}: {}", checkpoint_path, e);
                return;
            }
        }
    } else {
        Model::new(args.embedding_dim, args.depth, args.n_heads)
    };
    let model_embedding_dim = model.embedding_dim();
    let model_depth = model.depth();
    let model_n_heads = model.n_heads();
    model.set_training_metadata(TrainingMetadata {
        embedding_dim: model_embedding_dim,
        depth: model_depth,
        n_heads: model_n_heads,
        pretrain_epochs: args.pretrain_epochs,
        pretrain_lr: args.pretrain_lr,
        pretrain_batch_size: args.pretrain_batch_size,
        pretrain_warmup: args.pretrain_warmup,
        pretrain_min_lr: args.pretrain_min_lr,
        num_epochs: args.num_epochs,
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        dropout: args.dropout,
        val_split: args.val_split,
        warmup_steps: args.warmup_steps,
        min_lr: args.min_lr,
        grad_clip: args.grad_clip,
        deterministic: args.deterministic,
        seed: args.seed,
        checkpoint_every: args.checkpoint_every,
        min_count: args.min_count,
        seq_length: args.seq_length,
        pretraining_files: pretraining_files.clone(),
        tuning_files: tuning_files.clone(),
        output_file: args.output_file.clone(),
    });
    let mut training_data = TrainingData::new(model);

    if !resuming {
        // Build vocabulary from pretraining data first, so min_count pruning applies
        // only to corpus-derived tokens.
        for file_name in &pretraining_files {
            if let Err(e) = training_data.load(file_name) {
                eprintln!("Error loading pretraining data from {}: {}", file_name, e);
                return;
            }
        }
        println!(
            "Loaded {} pretraining prompts for vocabulary build, vocab = {}",
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

        // Clear temporary pretraining prompts, then add tuning data to vocabulary
        // after pruning so tuning-only tokens are never removed by min_count.
        training_data.clear_prompts();
        for file_name in &tuning_files {
            if let Err(e) = training_data.load(file_name) {
                eprintln!("Error loading tuning data from {}: {}", file_name, e);
                return;
            }
        }
        println!(
            "Loaded {} tuning prompts for vocabulary build after pruning, vocab = {}",
            training_data.num_prompts(),
            training_data.get_model().vocab_size()
        );
        training_data.clear_prompts();

        training_data.get_model_mut().initialize_embeddings();
    } else {
        println!(
            "Resume mode: vocabulary is frozen from checkpoint (min_count and vocab rebuild are skipped)."
        );
    }

    let pretrain_lr_schedule = if args.pretrain_warmup > 0 {
        Some(LrSchedule::CosineWithWarmup {
            warmup_steps: args.pretrain_warmup,
            min_lr: args.pretrain_min_lr,
        })
    } else {
        None
    };
    let tune_lr_schedule = if args.warmup_steps > 0 {
        Some(LrSchedule::CosineWithWarmup {
            warmup_steps: args.warmup_steps,
            min_lr: args.min_lr,
        })
    } else {
        None
    };

    let pretrain_cfg = TrainingConfig {
        num_epochs: args.pretrain_epochs,
        learning_rate: args.pretrain_lr,
        batch_size: args.pretrain_batch_size,
        val_split: 0.0,
        seed: args.seed,
        checkpoint_every: args.checkpoint_every,
        checkpoint_path: Some(args.output_file.clone()),
        lr_schedule: pretrain_lr_schedule,
        dropout: args.dropout,
        grad_clip: if args.grad_clip > 0.0 { Some(args.grad_clip) } else { None },
        deterministic: args.deterministic,
    };

    let tune_cfg = TrainingConfig {
        num_epochs: args.num_epochs,
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        val_split: args.val_split,
        seed: args.seed,
        checkpoint_every: args.checkpoint_every,
        checkpoint_path: Some(args.output_file.clone()),
        lr_schedule: tune_lr_schedule,
        dropout: args.dropout,
        grad_clip: if args.grad_clip > 0.0 { Some(args.grad_clip) } else { None },
        deterministic: args.deterministic,
    };

    let train_start = std::time::Instant::now();
    let _sleep_guard = SleepGuard::new();

    training_data.clear_prompts();

    // Load pretraining data with specified sequence length.
    let mut pretraining_data = PretrainingData::new(training_data.get_model().clone());
    for file_name in &pretraining_files {
        if let Err(e) = pretraining_data.load(file_name, args.seq_length) {
            eprintln!("Error loading pretraining data from {}: {}", file_name, e);
            return;
        }
    }

    if args.resume_skip_pretraining {
        println!("Skipping pretraining phase (--resume-skip-pretraining=true).");
    } else if pretraining_data.num_sequences() > 0 {
        println!(
            "Pretraining on {} sequences (avg ~{} tokens, without <stop>)",
            pretraining_data.num_sequences(),
            args.seq_length
        );

        training_data.clear_prompts();
        for seq in pretraining_data.sequences() {
            // Use the pre-tokenized, <unk>-filtered token stream directly so
            // we don't re-introduce <unk> ids by re-tokenizing the joined text.
            training_data.add_prompt_tokens(seq.text.clone(), seq.tokens.clone());
        }
        training_data.train_with_stop_mode(&pretrain_cfg, StopTokenMode::Bare);
    }

    training_data.clear_prompts();
    for file_name in &tuning_files {
        if let Err(e) = training_data.load_with_existing_vocab(file_name) {
            eprintln!("Error loading chat tuning data from {}: {}", file_name, e);
            return;
        }
    }
    if training_data.num_prompts() == 0 {
        eprintln!("No chat-tuning prompts loaded from tuning_files.");
        return;
    }
    println!(
        "Chat-tuning on {} prompts (with <stop>)",
        training_data.num_prompts()
    );
    training_data.train_with_stop_mode(&tune_cfg, StopTokenMode::AppendStop);

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
