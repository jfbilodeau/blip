use crate::model::Model;

#[derive(PartialEq)]
enum TokenType {
    Word,
    Punctuation,
}

/// Split raw text into the canonical lower-cased token strings.
/// Pure function: no model state is touched.
///
/// Apostrophes between alphanumeric characters are kept attached so that
/// English contractions (`I'm`, `don't`, `we're`) survive as single tokens.
pub fn split_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_type: Option<TokenType> = None;

    for (i, &c) in chars.iter().enumerate() {
        let next_alnum = chars.get(i + 1).map_or(false, |n| n.is_alphanumeric());
        let prev_alnum = i > 0 && chars[i - 1].is_alphanumeric();
        // Treat `'` as a word character if it sits between two alphanumerics.
        let is_word_apostrophe = c == '\'' && prev_alnum && next_alnum;

        if c.is_alphanumeric() || is_word_apostrophe {
            if current_type != Some(TokenType::Word) && !current.is_empty() {
                tokens.push(current.to_lowercase());
                current.clear();
            }
            current.push(c);
            current_type = Some(TokenType::Word);
        } else if c.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(current.to_lowercase());
                current.clear();
            }
            tokens.push(c.to_string());
            current_type = Some(TokenType::Punctuation);
        } else if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.to_lowercase());
                current.clear();
            }
            current_type = None;
        }
    }

    if !current.is_empty() {
        tokens.push(current.to_lowercase());
    }

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

/// Punctuation-aware detokenizer. Skips special `<...>` tokens, omits the
/// space before closing punctuation (`,`, `.`, `!`, `?`, `;`, `:`, `)`, `]`,
/// `}`, `'`) and after opening punctuation (`(`, `[`, `{`).
pub fn detokenize_text(tokens: &[usize], model: &Model) -> String {
    let mut out = String::new();
    let mut prev_was_open = false;
    for &id in tokens {
        let Some(text) = model.get_token_by_id(id) else { continue };
        if text.starts_with('<') && text.ends_with('>') {
            continue;
        }
        let no_leading_space = text.len() == 1
            && matches!(
                text.chars().next().unwrap(),
                ',' | '.' | '!' | '?' | ';' | ':' | ')' | ']' | '}' | '\'' | '"'
            );
        if !out.is_empty() && !no_leading_space && !prev_was_open {
            out.push(' ');
        }
        out.push_str(text);
        prev_was_open = text.len() == 1
            && matches!(text.chars().next().unwrap(), '(' | '[' | '{');
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
        assert_eq!(toks, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn split_keeps_contractions_together() {
        let toks = split_tokens("I'm sure we don't need you're split apart.");
        assert_eq!(
            toks,
            vec!["i'm", "sure", "we", "don't", "need", "you're", "split", "apart", "."]
        );
    }

    #[test]
    fn tokenize_registers_then_inference_finds() {
        let mut model = Model::new(8, 1, 2);
        let train_ids = tokenize("Hello world", &mut model);
        assert_eq!(train_ids.len(), 2);

        let infer_ids = tokenize_for_inference("hello WORLD", &model);
        assert_eq!(infer_ids, train_ids);
    }

    #[test]
    fn inference_maps_unknown_to_unk() {
        let mut model = Model::new(8, 1, 2);
        tokenize("hello", &mut model);
        let ids = tokenize_for_inference("hello bonjour", &model);
        let unk = model.get_unknown_token_id();
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], unk);
        assert_eq!(ids[1], unk);
    }
}
