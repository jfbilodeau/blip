use crate::model::Model;

pub fn tokenize(text: &str, model: &mut Model) -> Vec<usize> {
    let mut tokens = Vec::new();

    for word in text.split_whitespace() {
        let cleaned_word = word.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        let token_id = model.register_token(cleaned_word);
        
        tokens.push(token_id);
    }
    
    tokens
}