use crate::model::{Model, TOKEN_UNKNOWN};

#[derive(PartialEq)]
enum TokenType {
    Letters,
    Digits,
    Symbols,
    Unknown,
}

const DEFAULT_SYMBOLS: &[char] = &[
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';',
    '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
];

pub fn default_token_texts() -> Vec<String> {
    let mut tokens = Vec::with_capacity(26 + 2 + DEFAULT_SYMBOLS.len());
    for letter in 'a'..='z' {
        tokens.push(letter.to_string());
    }
    for digit in ['0', '1'] {
        tokens.push(digit.to_string());
    }
    for symbol in DEFAULT_SYMBOLS {
        tokens.push(symbol.to_string());
    }
    tokens
}

fn classify_char(c: char) -> TokenType {
    if c.is_ascii_lowercase() {
        TokenType::Letters
    } else if matches!(c, '0' | '1') {
        TokenType::Digits
    } else if DEFAULT_SYMBOLS.contains(&c) {
        TokenType::Symbols
    } else {
        TokenType::Unknown
    }
}

fn push_current(tokens: &mut Vec<String>, current: &mut String) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

/// Split raw text into canonical lower-cased token strings.
/// Pure function: no model state is touched.
///
/// Tokens are ASCII-only runs of lowercase letters, digits, or symbols.
/// Non-ASCII letters and punctuation become `<unk>`.
pub fn split_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_type: Option<TokenType> = None;

    for c in text.chars() {
        let normalized = c.to_ascii_lowercase();
        let token_type = classify_char(normalized);

        if current_type.as_ref() != Some(&token_type) {
            push_current(&mut tokens, &mut current);
            current_type = Some(token_type);
        }

        match current_type {
            Some(TokenType::Letters) | Some(TokenType::Digits) | Some(TokenType::Symbols) => {
                current.push(normalized);
            }
            Some(TokenType::Unknown) => {
                current.push_str(TOKEN_UNKNOWN);
            }
            None => {}
        }
    }

    push_current(&mut tokens, &mut current);
    tokens
}

/// Tokenize for training: registers any new tokens in the model's vocabulary.
pub fn tokenize(text: &str, model: &mut Model) -> Vec<usize> {
    split_tokens(text)
        .into_iter()
        .map(|t| model.register_token(&t))
        .collect()
}

/// Tokenize for inference: read-only. Unknown tokens map to `<unk>`.
pub fn tokenize_for_inference(text: &str, model: &Model) -> Vec<usize> {
    let unk = model.get_unknown_token_id();
    split_tokens(text)
        .into_iter()
        .map(|t| model.get_token_id(&t).unwrap_or(unk))
        .collect()
}

pub fn detokenize(tokens: &[usize], model: &Model) -> String {
    tokens
        .iter()
        .map(|&id| model.get_token_by_id(id).unwrap_or(""))
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Detokenizer that reconstructs text by concatenating token text directly.
/// Special `<...>` tokens are skipped.
pub fn detokenize_text(tokens: &[usize], model: &Model) -> String {
    let mut out = String::new();
    for &id in tokens {
        let Some(text) = model.get_token_by_id(id) else {
            continue;
        };
        if text.starts_with('<') && text.ends_with('>') {
            continue;
        }
        out.push_str(text);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;

    #[test]
    fn split_lowercases_and_separates_punctuation() {
        let toks = split_tokens("Hello, World!");
        assert_eq!(toks, vec!["hello", ", ", "world", "!"]);
    }

    #[test]
    fn split_groups_ascii_letter_digit_and_symbol_runs() {
        let toks = split_tokens("abc==101  xyz-10");
        assert_eq!(toks, vec!["abc", "==", "101", "  ", "xyz", "-", "10"]);
    }

    #[test]
    fn split_maps_non_ascii_to_unk() {
        let toks = split_tokens("caf");
        assert_eq!(toks, vec!["caf", "<unk><unk>"]);

        let toks = split_tokens("café!");
        assert_eq!(toks, vec!["caf", "<unk>", "!"]);

        let toks = split_tokens("a😀b");
        assert_eq!(toks, vec!["a", "<unk>", "b"]);
    }

    #[test]
    fn default_tokens_include_ascii_letters_digits_and_symbols() {
        let defaults = default_token_texts();
        assert!(defaults.iter().any(|t| t == "a"));
        assert!(defaults.iter().any(|t| t == "z"));
        assert!(defaults.iter().any(|t| t == "0"));
        assert!(defaults.iter().any(|t| t == "1"));
        assert!(!defaults.iter().any(|t| t == "9"));
        assert!(defaults.iter().any(|t| t == " "));
        assert!(defaults.iter().any(|t| t == "="));
    }

    #[test]
    fn tokenize_registers_then_inference_finds() {
        let mut model = Model::new(8, 1, 2);
        let train_ids = tokenize("Hello world", &mut model);
        assert_eq!(train_ids.len(), 3);

        let infer_ids = tokenize_for_inference("hello WORLD", &model);
        assert_eq!(infer_ids, train_ids);
    }

    #[test]
    fn inference_maps_unknown_to_unk() {
        let mut model = Model::new(8, 1, 2);
        tokenize("hello", &mut model);
        let ids = tokenize_for_inference("hello bonjour café", &model);
        let unk = model.get_unknown_token_id();
        assert_eq!(ids.len(), 6);
        assert_ne!(ids[0], unk);
        assert_ne!(ids[1], unk);
        assert_eq!(ids[2], unk);
        assert_ne!(ids[3], unk);
        assert_eq!(ids[4], unk);
        assert_eq!(ids[5], unk);
    }
}
