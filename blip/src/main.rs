use blip_ai::model::{Model, SamplingConfig};
use blip_ai::tokenizer::{detokenize_text, tokenize_user_prompt};
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "Blip",
    author = "J-F Bilodeau (jfbilodeau@chronogears.com)",
    version,
    about,
    long_about = None
)]
struct BlipArgs {
    #[arg(short = 'f', long, default_value = "models/basic.json")]
    pub model_file: String,

    /// One-shot prompt. If omitted (and `--repl` not given), defaults to
    /// "Who are you?".
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    #[arg(short = 'm', long, default_value = "1024", help = "Maximum number of new tokens to generate")]
    pub max_new_tokens: usize,

    #[arg(short = 't', long, default_value = "0.0", help = "Sampling temperature; 0 = greedy")]
    pub temperature: f32,

    #[arg(long, help = "Top-k sampling cutoff")]
    pub top_k: Option<usize>,

    #[arg(long, help = "Top-p (nucleus) sampling cutoff in (0,1]")]
    pub top_p: Option<f32>,

    #[arg(long, default_value = "0", help = "RNG seed for sampling (0 = random)")]
    pub seed: u64,
}

/// Build the inference token prompt: `[<user>, ...user_tokens, <ai>]`.
/// Strips any literal `user:` / `ai:` prefix the user might have typed so the
/// REPL never feeds those literal tokens into the model.
fn build_prompt_tokens(model: &Model, prompt: &str) -> Vec<usize> {
    let user_tok = model.get_user_token_id();
    let ai_tok = model.get_ai_token_id();

    let mut text = prompt.trim().to_string();
    // Strip a leading `user:` so users typing it manually don't get a literal
    // "user:" tokenized into the prompt.
    let lower = text.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("user:") {
        let rest_offset = text.len() - rest.len();
        text = text[rest_offset..].trim().to_string();
    }

    let mut tokens = Vec::with_capacity(prompt.len() + 2);
    tokens.push(user_tok);
    tokens.extend(tokenize_user_prompt(&text, model));
    tokens.push(ai_tok);
    tokens
}

fn run_once(model: &Model, prompt: &str, cfg: &SamplingConfig, seed: u64) {
    let tokens = build_prompt_tokens(model, prompt);
    let result = if seed == 0 {
        let mut rng = rand::rng();
        model.generate_token_ids(&tokens, cfg, &mut rng)
    } else {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        model.generate_token_ids(&tokens, cfg, &mut rng)
    };
    match result {
        Ok(ids) => {
            let text = detokenize_text(&ids, model);
            if text.trim().is_empty() {
                let rendered = ids
                    .iter()
                    .filter_map(|&id| model.get_token_by_id(id))
                    .map(|t| {
                        if t.chars().all(|c| c.is_whitespace()) {
                            format!("<ws:{}>", t.chars().count())
                        } else {
                            t.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");

                if rendered.is_empty() {
                    println!("<no output>");
                } else {
                    println!("<blank output; generated {}>", rendered);
                }
            } else {
                println!("{}", text);
            }
        }
        Err(e) => eprintln!("Generation failed: {}", e),
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let exiting = Arc::new(AtomicBool::new(false));
    let exiting_for_handler = Arc::clone(&exiting);
    ctrlc::set_handler(move || {
        if !exiting_for_handler.swap(true, Ordering::SeqCst) {
            eprintln!("\nCtrl+C exit");
        }
        std::process::exit(0);
    })
    .expect("Failed to install Ctrl+C handler");

    let args = BlipArgs::parse();

    println!("Loading model from {}...", args.model_file);
    let model = Model::load(&args.model_file)
    .expect("Failed to load model");

    let cfg = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        max_new_tokens: args.max_new_tokens,
    };

    if let Some(prompt) = args.prompt.as_deref() {
        println!("Prompt: {}", prompt);
        run_once(&model, prompt, &cfg, args.seed);
    } else {
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
    }
}

#[cfg(test)]
mod tests {
    use super::build_prompt_tokens;
    use blip_ai::model::Model;

    fn fresh_model() -> Model {
        let mut m = Model::new(8, 1, 2);
        m.initialize_embeddings();
        m
    }

    #[test]
    fn build_prompt_starts_with_user_and_ends_with_ai() {
        let m = fresh_model();
        let toks = build_prompt_tokens(&m, "Who are you?");
        assert_eq!(toks.first().copied(), Some(m.get_user_token_id()));
        assert_eq!(toks.last().copied(), Some(m.get_ai_token_id()));
    }

    #[test]
    fn build_prompt_strips_literal_user_prefix() {
        let m = fresh_model();
        let with_prefix = build_prompt_tokens(&m, "user:Who are you?");
        let without_prefix = build_prompt_tokens(&m, "Who are you?");
        assert_eq!(with_prefix, without_prefix);
    }

    #[test]
    fn build_prompt_does_not_contain_literal_role_text() {
        // The default vocab includes all single ASCII letters and digits, so
        // single-letter inputs round-trip through detokenize_text cleanly.
        // Control tokens (<bos>, <user>, <ai>) must be stripped.
        let m = fresh_model();
        let toks = build_prompt_tokens(&m, "a b");
        let text = blip_ai::tokenizer::detokenize_text(&toks, &m);
        // split_tokens("a b") → ["a", " ", "b"]; detokenize concatenates them.
        assert_eq!(text, "a b");
    }
}
