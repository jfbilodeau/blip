use clap::Parser;
use blip_ai::model::Model;
use blip_ai::tokenizer::tokenize;

#[derive(Parser, Debug)]
#[command(
    name="Blip",
    author = "J-F Bilodeau (jfbilodeau@chronogears.com)",
    version = "1.0",
    about,
    long_about = None
)]
struct BlipArgs {
    #[arg(short = 'f', long, default_value = "models/basic.json")]
    pub model_file: String,
}

fn main() {
    let args = BlipArgs::parse();

    println!("Starting Blip AI...");
    println!("Loading model from file: {}...", args.model_file);

    let start = std::time::Instant::now();
    let mut model = Model::load(&args.model_file)
        .expect("Failed to load model");

    println!("Model loaded successfully in {} seconds", start.elapsed().as_secs_f32());

    // Test prediction
    let test_prompt = "Who are you?";
    println!("Testing model with prompt: '{}'", test_prompt);
    let tokens = tokenize(test_prompt, &mut model);
    let prediction = model.complete(&tokens);

    println!("Model prediction: '{}'", prediction.unwrap());
}
