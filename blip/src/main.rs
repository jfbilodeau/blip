use clap::Parser;
use blip_ai::model::Model;

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
    let model = Model::load(&args.model_file)
        .expect("Failed to load model");
    
    println!("Model loaded successfully in {} seconds", start.elapsed().as_secs_f32());
    
    
}
