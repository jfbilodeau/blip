use blip_ai::model::Model;
use blip_ai::tokenizer::tokenize;

pub struct TrainingPrompt {
    prompt: String,
    tokens: Vec<usize>,
}

pub struct TrainingData {
    // Define the structure of your training data here
    // For example, if it's a vector of strings:
    data: Vec<TrainingPrompt>,
}

pub fn create_training_data() -> TrainingData {
    TrainingData {
        data: Vec::new(),
    }
}

pub fn load_training_data(file_name: &str, model: &mut Model, training_data: &mut TrainingData) -> Result<(), String> {
    let lines = std::fs::read_to_string(file_name)
        .map_err(|e| e.to_string())?
        .lines()
        .map(|l| l.to_string())
        .collect::<Vec<String>>();
    
    for line in lines {
        let tokens = tokenize(&line, model);
        
        training_data.data.push(TrainingPrompt {
            prompt: line,
            tokens,
        });
    }
    
    Ok(())
}