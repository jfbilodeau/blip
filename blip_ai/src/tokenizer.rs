use crate::model::{Model, TOKEN_AI, TOKEN_BEGIN, TOKEN_STOP, TOKEN_TOOL, TOKEN_UNKNOWN, TOKEN_USER};

#[derive(PartialEq)]
enum TokenType {
    Letters,
    Digits,
    Symbols,
    Unknown,
}

const DEFAULT_SYMBOLS: &[char] = &[
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';',
    '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\n', '\t',
];

/// Multi-character symbol tokens that appear naturally in text once Unicode
/// punctuation/arrows are normalized. Listing them in the default vocab makes
/// sure they always have an embedding even on tiny corpora.
const DEFAULT_MULTI_CHAR_SYMBOLS: &[&str] = &["--", "->", "<-", "<->", "..."];

pub fn default_token_texts() -> Vec<String> {
    let mut tokens =
        Vec::with_capacity(26 + 10 + DEFAULT_SYMBOLS.len() + DEFAULT_MULTI_CHAR_SYMBOLS.len());
    for letter in 'a'..='z' {
        tokens.push(letter.to_string());
    }
    for digit in '0'..='9' {
        tokens.push(digit.to_string());
    }
    for symbol in DEFAULT_SYMBOLS {
        tokens.push(symbol.to_string());
    }
    for s in DEFAULT_MULTI_CHAR_SYMBOLS {
        tokens.push((*s).to_string());
    }
    tokens
}

fn classify_char(c: char) -> TokenType {
    if c.is_ascii_lowercase() {
        TokenType::Letters
    } else if c.is_ascii_digit() {
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

/// Map a non-ASCII character to its ASCII substitute. Returns:
///   * `Some("...")` to substitute with that string (possibly multi-char)
///   * `None`        to fall through to `ascii_fold_letter` / drop
fn unicode_substitute(c: char) -> Option<&'static str> {
    match c {
        // Single quotes (curly, low-9, high-reversed-9)
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => Some("'"),
        // Double quotes (curly, low-9, high-reversed-9, guillemets)
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{00AB}' | '\u{00BB}' => Some("\""),
        // Hyphens / non-breaking dashes
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' => Some("-"),
        // Em dash, horizontal bar, double hyphen punctuation
        '\u{2014}' | '\u{2015}' | '\u{2E3A}' | '\u{2E3B}' => Some("--"),
        // Ellipsis
        '\u{2026}' => Some("..."),
        // Arrows
        '\u{2192}' | '\u{21D2}' | '\u{27F6}' | '\u{27F9}' => Some("->"),
        '\u{2190}' | '\u{21D0}' | '\u{27F5}' | '\u{27F8}' => Some("<-"),
        '\u{2194}' | '\u{21D4}' | '\u{27F7}' | '\u{27FA}' => Some("<->"),
        // Math operators
        '\u{00D7}' => Some("x"),
        '\u{00F7}' => Some("/"),
        '\u{2212}' => Some("-"),
        '\u{00B1}' => Some("+/-"),
        '\u{2260}' => Some("!="),
        '\u{2264}' => Some("<="),
        '\u{2265}' => Some(">="),
        // Bullets / middots
        '\u{00B7}' | '\u{2022}' | '\u{2027}' => Some("*"),
        // Various Unicode spaces (NBSP, en/em spaces, thin space, ideographic)
        '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
        | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200A}' | '\u{202F}' | '\u{205F}'
        | '\u{3000}' => Some(" "),
        // Soft hyphen / zero-width chars: drop entirely
        '\u{00AD}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => Some(""),
        _ => None,
    }
}

/// Fold a Latin letter with diacritics to its closest ASCII letter. Returns
/// `None` if `c` has no obvious ASCII equivalent.
fn ascii_fold_letter(c: char) -> Option<char> {
    // Lowercase first so the table only needs the lowercase variants.
    let lower = c.to_lowercase().next().unwrap_or(c);
    match lower {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' | 'ǎ' => Some('a'),
        'æ' => Some('a'),
        'ç' | 'ć' | 'č' | 'ĉ' | 'ċ' => Some('c'),
        'ð' | 'ď' | 'đ' => Some('d'),
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => Some('e'),
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => Some('g'),
        'ĥ' | 'ħ' => Some('h'),
        'ì' | 'í' | 'î' | 'ï' | 'ī' | 'ĭ' | 'į' | 'ı' => Some('i'),
        'ĵ' => Some('j'),
        'ķ' => Some('k'),
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => Some('l'),
        'ñ' | 'ń' | 'ņ' | 'ň' | 'ŋ' => Some('n'),
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => Some('o'),
        'œ' => Some('o'),
        'ŕ' | 'ŗ' | 'ř' => Some('r'),
        'ś' | 'ŝ' | 'ş' | 'š' => Some('s'),
        'ß' => Some('s'),
        'ţ' | 'ť' | 'ŧ' => Some('t'),
        'þ' => Some('t'),
        'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => Some('u'),
        'ŵ' => Some('w'),
        'ý' | 'ÿ' | 'ŷ' => Some('y'),
        'ź' | 'ż' | 'ž' => Some('z'),
        _ => None,
    }
}

/// Fold a string to ASCII-only by:
///   1. Substituting known Unicode punctuation/symbols/arrows.
///   2. Folding accented Latin letters to their base ASCII letter.
///   3. Dropping anything else that doesn't naturally map to ASCII.
///
/// This is the entry point used by [`split_tokens`]. It is exposed publicly so
/// callers (and tests) can inspect the normalized text directly.
pub fn normalize_to_ascii(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
            continue;
        }
        if let Some(replacement) = unicode_substitute(c) {
            out.push_str(replacement);
            continue;
        }
        if let Some(folded) = ascii_fold_letter(c) {
            out.push(folded);
            continue;
        }
        // Unknown non-ASCII (emoji, CJK, symbols we don't map): drop silently.
    }
    out
}

/// Split raw text into canonical lower-cased token strings.
/// Pure function: no model state is touched.
///
/// The input is first normalized by [`normalize_to_ascii`], so curly quotes,
/// dashes, arrows, accented letters, etc. become their ASCII equivalents (or
/// are dropped). Tokens are then runs of lowercase letters, digits, or
/// symbols. Anything that survives normalization but still doesn't classify
/// is mapped to `<unk>`.
pub fn split_tokens(text: &str) -> Vec<String> {
    let normalized = normalize_to_ascii(text);

    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_type: Option<TokenType> = None;

    for c in normalized.chars() {
        let normalized_c = c.to_ascii_lowercase();
        let token_type = classify_char(normalized_c);

        if current_type.as_ref() != Some(&token_type) {
            push_current(&mut tokens, &mut current);
            current_type = Some(token_type);
        }

        match current_type {
            Some(TokenType::Letters) | Some(TokenType::Digits) | Some(TokenType::Symbols) => {
                current.push(normalized_c);
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

/// Tokenize for vocabulary construction: registers every produced token in
/// the model's vocabulary. Use this only during the initial vocab-build pass.
pub fn tokenize_build_vocab(text: &str, model: &mut Model) -> Vec<usize> {
    split_tokens(text)
        .into_iter()
        .map(|t| model.register_token(&t))
        .collect()
}

/// Tokenize against an already-frozen vocabulary. Tokens missing from the
/// vocab map to `<unk>`. Use this for all training data after the vocab is
/// finalized, for pretraining sequences, and for inference input.
pub fn tokenize_with_vocab(text: &str, model: &Model) -> Vec<usize> {
    let unk = model.get_unknown_token_id();
    split_tokens(text)
        .into_iter()
        .map(|t| model.get_token_id(&t).unwrap_or(unk))
        .collect()
}

/// Backward-compatible alias for [`tokenize_build_vocab`].
pub fn tokenize(text: &str, model: &mut Model) -> Vec<usize> {
    tokenize_build_vocab(text, model)
}

/// Backward-compatible alias for [`tokenize_with_vocab`].
pub fn tokenize_for_inference(text: &str, model: &Model) -> Vec<usize> {
    tokenize_with_vocab(text, model)
}

pub fn detokenize(tokens: &[usize], model: &Model) -> String {
    tokens
        .iter()
        .map(|&id| model.get_token_by_id(id).unwrap_or(""))
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Detokenizer that reconstructs text by concatenating token text directly.
/// Control tokens (`<bos>`, `<stop>`, `<tool>`, `<user>`, `<ai>`) are skipped.
/// `<unk>` is preserved so unknown-heavy generations do not appear as blank
/// output.
pub fn detokenize_text(tokens: &[usize], model: &Model) -> String {
    let mut out = String::new();
    for &id in tokens {
        let Some(text) = model.get_token_by_id(id) else {
            continue;
        };
        if matches!(text, TOKEN_BEGIN | TOKEN_STOP | TOKEN_TOOL | TOKEN_USER | TOKEN_AI) {
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
    fn split_normalizes_non_ascii_or_drops_it() {
        // Plain ASCII passes through.
        let toks = split_tokens("caf");
        assert_eq!(toks, vec!["caf"]);

        // Accented letters fold to their ASCII base, so 'é' -> 'e'.
        let toks = split_tokens("café!");
        assert_eq!(toks, vec!["cafe", "!"]);

        // Emoji has no ASCII mapping and is dropped silently.
        let toks = split_tokens("a😀b");
        assert_eq!(toks, vec!["ab"]);
    }

    #[test]
    fn unicode_quotes_dashes_and_ellipsis_normalize() {
        let toks = split_tokens("\u{2018}hi\u{2019} \u{201C}there\u{201D}\u{2014}you\u{2026}");
        assert_eq!(toks, vec!["'", "hi", "' \"", "there", "\"--", "you", "..."]);
    }

    #[test]
    fn unicode_arrows_normalize() {
        let toks = split_tokens("a\u{2192}b\u{2190}c\u{2194}d");
        assert_eq!(toks, vec!["a", "->", "b", "<-", "c", "<->", "d"]);
    }

    #[test]
    fn default_tokens_include_ascii_letters_digits_symbols_and_multichars() {
        let defaults = default_token_texts();
        assert!(defaults.iter().any(|t| t == "a"));
        assert!(defaults.iter().any(|t| t == "z"));
        assert!(defaults.iter().any(|t| t == "0"));
        assert!(defaults.iter().any(|t| t == "9"));
        assert!(defaults.iter().any(|t| t == " "));
        assert!(defaults.iter().any(|t| t == "="));
        assert!(defaults.iter().any(|t| t == "--"));
        assert!(defaults.iter().any(|t| t == "->"));
        assert!(defaults.iter().any(|t| t == "<->"));
        assert!(defaults.iter().any(|t| t == "..."));
    }

    #[test]
    fn tokenize_build_vocab_then_with_vocab_match() {
        let mut model = Model::new(8, 1, 2);
        let train_ids = tokenize_build_vocab("Hello world", &mut model);
        assert_eq!(train_ids.len(), 3);

        let infer_ids = tokenize_with_vocab("hello WORLD", &model);
        assert_eq!(infer_ids, train_ids);
    }

    #[test]
    fn tokenize_with_vocab_maps_unknown_to_unk() {
        let mut model = Model::new(8, 1, 2);
        tokenize_build_vocab("hello", &mut model);
        // "bonjour" is unknown; "café" folds to "cafe" which is also unknown.
        let ids = tokenize_with_vocab("hello bonjour café", &model);
        let unk = model.get_unknown_token_id();
        // Tokens after split: hello, ' ', bonjour, ' ', cafe -> 5 tokens.
        assert_eq!(ids.len(), 5);
        assert_ne!(ids[0], unk);
        assert_ne!(ids[1], unk);
        assert_eq!(ids[2], unk);
        assert_ne!(ids[3], unk);
        assert_eq!(ids[4], unk);
    }
}
