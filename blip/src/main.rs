use blip_ai::model::{Model, SamplingConfig};
use blip_ai::tokenizer::{detokenize_text, tokenize_for_inference};
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::io::{BufRead, Write};

#[derive(Parser, Debug)]
#[command(
    name = "Blip",
    author = "J-F Bilodeau (jfbilodeau@chronogears.com)",
    version,
    about,
    long_about = None
)]
struct BlipArgs {
    #[arg(short = 'f', long, default_value = "models/basic.bin")]
    pub model_file: String,

    /// One-shot prompt. If omitted (and `--repl` not given), defaults to
    /// "Who are you?".
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    #[arg(short = 'm', long, default_value = "64")]
    pub max_new_tokens: usize,

    #[arg(short = 't', long, default_value = "0.0", help = "Sampling temperature; 0 = greedy")]
    pub temperature: f32,

    #[arg(long, help = "Top-k sampling cutoff")]
    pub top_k: Option<usize>,

    #[arg(long, help = "Top-p (nucleus) sampling cutoff in (0,1]")]
    pub top_p: Option<f32>,

    #[arg(long, default_value = "0", help = "RNG seed for sampling (0 = random)")]
    pub seed: u64,

    #[arg(long, help = "Interactive REPL mode")]
    pub repl: bool,
}

fn run_once(model: &Model, prompt: &str, cfg: &SamplingConfig, seed: u64) {
    let mut tokens = vec![model.get_begin_token_id()];
    tokens.extend(tokenize_for_inference(prompt, model));
    let result = if seed == 0 {
        let mut rng = rand::rng();
        model.generate_token_ids(&tokens, cfg, &mut rng)
    } else {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        model.generate_token_ids(&tokens, cfg, &mut rng)
    };
    match result {
        Ok(ids) => println!("{}", detokenize_text(&ids, model)),
        Err(e) => eprintln!("Generation failed: {}", e),
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = BlipArgs::parse();

    println!("Loading model from {}...", args.model_file);
    let model = Model::load(&args.model_file).expect("Failed to load model");

    let cfg = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        max_new_tokens: args.max_new_tokens,
    };

    if args.repl {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        loop {
            print!("blip> ");
            let _ = stdout.flush();
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => {
                    eprintln!("read error: {}", e);
                    break;
                }
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == ":quit" || trimmed == ":exit" {
                break;
            }
            run_once(&model, trimmed, &cfg, args.seed);
        }
    } else {
        let prompt = args.prompt.unwrap_or_else(|| "Who are you?".to_string());
        println!("Prompt: {}", prompt);
        run_once(&model, &prompt, &cfg, args.seed);
    }
}
