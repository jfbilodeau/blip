use crate::model::Model;

#[derive(PartialEq)]
enum TokenType {
    Word,
    Punctuation,
}

pub fn tokenize(text: &str, model: &mut Model) -> Vec<usize> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_type: Option<TokenType> = None;

    for c in text.chars() {
        if c.is_alphanumeric() {
            if current_type != Some(TokenType::Word) && !current.is_empty() {
                let token_id = model.register_token(&current.to_lowercase());
                tokens.push(token_id);
                current.clear();
            }
            current.push(c);
            current_type = Some(TokenType::Word);
        } else if c.is_ascii_punctuation() {
            if !current.is_empty() {
                let token_id = model.register_token(&current.to_lowercase());
                tokens.push(token_id);
                current.clear();
            }
            let token_id = model.register_token(&c.to_string());
            tokens.push(token_id);
            current_type = Some(TokenType::Punctuation);
        } else if c.is_whitespace() {
            if !current.is_empty() {
                let token_id = model.register_token(&current.to_lowercase());
                tokens.push(token_id);
                current.clear();
            }
            current_type = None;
        }
    }

    if !current.is_empty() {
        let token_id = model.register_token(&current.to_lowercase());
        tokens.push(token_id);
    }

    tokens
}

pub fn detokenize(tokens: &[usize], model: &Model) -> String {
    tokens.iter()
        .map(|&id| model.get_token_by_id(id).unwrap_or(&""))
        .collect::<Vec<&str>>()
        .join(" ")
}