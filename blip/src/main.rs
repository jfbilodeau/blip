use blip_ai::model::{Model, SamplingConfig};
use blip_ai::tokenizer::{detokenize_text, tokenize_for_inference};
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::path::{Path, PathBuf};
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
    #[arg(short = 'f', long, default_value = "models/basic.json")]
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

    #[arg(long, default_value = "true", help = "Interactive REPL mode")]
    pub repl: bool,
}

fn format_inference_prompt(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return "user: ai:".to_string();
    }
    if trimmed.contains("ai:") {
        return trimmed.to_string();
    }
    if let Some(user_text) = trimmed.strip_prefix("user:") {
        return format!("user:{} ai:", user_text.trim());
    }

    format!("user:{} ai:", trimmed)
}

fn run_once(model: &Model, prompt: &str, cfg: &SamplingConfig, seed: u64) {
    let mut tokens = vec![model.get_begin_token_id()];
    let formatted_prompt = format_inference_prompt(prompt);
    tokens.extend(tokenize_for_inference(&formatted_prompt, model));
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

fn resolve_model_path(requested: &str) -> PathBuf {
    let requested_path = PathBuf::from(requested);
    if requested_path.exists() {
        return requested_path;
    }

    let ext_is_json = Path::new(requested)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    if !ext_is_json {
        return requested_path;
    }

    let mut fallback = requested_path.clone();
    fallback.set_extension("bin");
    if fallback.exists() {
        eprintln!(
            "Model {} not found; falling back to {}.",
            requested_path.display(),
            fallback.display()
        );
        return fallback;
    }

    requested_path
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = BlipArgs::parse();

    let model_path = resolve_model_path(&args.model_file);
    println!("Loading model from {}...", model_path.display());
    let model = Model::load(
        model_path
            .to_str()
            .expect("Model path is not valid UTF-8"),
    )
    .expect("Failed to load model");

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

#[cfg(test)]
mod tests {
    use super::format_inference_prompt;

    #[test]
    fn wraps_plain_prompt_in_user_ai_template() {
        assert_eq!(format_inference_prompt("Who are you?"), "user:Who are you? ai:");
    }

    #[test]
    fn appends_ai_tag_to_existing_user_prompt() {
        assert_eq!(format_inference_prompt("user:Who are you?"), "user:Who are you? ai:");
    }

    #[test]
    fn preserves_explicit_conversation_prompt() {
        assert_eq!(
            format_inference_prompt("user:Who are you? ai:I am Blip."),
            "user:Who are you? ai:I am Blip."
        );
    }
}
