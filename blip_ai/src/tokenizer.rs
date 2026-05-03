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

// ---------------------------------------------------------------------------
// Special-token-aware tokenization (used by tuning data loader)
// ---------------------------------------------------------------------------

/// The literal special-token names the tuning loader recognizes inline. A
/// match is emitted as a single special-token id; prefix the literal with a
/// backslash (`\<unk>`) to escape — the backslash is dropped and the rest
/// tokenizes through the normal pipeline (so `<`, `unk`, `>` come out as
/// three ordinary tokens).
const RECOGNIZED_SPECIALS: &[&str] = &[
    TOKEN_UNKNOWN,
    TOKEN_STOP,
    TOKEN_TOOL,
    TOKEN_BEGIN,
    TOKEN_USER,
    TOKEN_AI,
];

/// One chunk produced by `split_with_specials`: either a literal special
/// token name to be emitted as a single id, or a run of plain text to be
/// fed through `split_tokens`.
enum SpecialChunk<'a> {
    Special(&'a str),
    Plain(String),
}

/// Scan `text` left-to-right, splitting on recognized special-token literals
/// (`<unk>`, `<stop>`, `<tool>`, `<bos>`, `<user>`, `<ai>`). A backslash
/// immediately before such a literal escapes it: the backslash is removed
/// and the literal is returned as plain text. Other backslashes are
/// preserved verbatim.
fn split_with_specials(text: &str) -> Vec<SpecialChunk<'_>> {
    let bytes = text.as_bytes();
    let mut out: Vec<SpecialChunk<'_>> = Vec::new();
    let mut plain = String::new();
    let mut i = 0;
    while i < bytes.len() {
        // Escaped special: `\<name>` -> drop backslash, emit literal as plain.
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'<' {
            if let Some(name) = match_special_at(text, i + 1) {
                plain.push_str(name);
                i += 1 + name.len();
                continue;
            }
            // Lone `\<` that is not followed by a recognized special: keep
            // the backslash literally and let the `<` start normal text.
            plain.push('\\');
            i += 1;
            continue;
        }
        if bytes[i] == b'<' {
            if let Some(name) = match_special_at(text, i) {
                if !plain.is_empty() {
                    out.push(SpecialChunk::Plain(std::mem::take(&mut plain)));
                }
                out.push(SpecialChunk::Special(name));
                i += name.len();
                continue;
            }
        }
        // Plain byte. Push as a char (text is &str so byte i is a char start
        // unless we're inside a multi-byte sequence — handle that by walking
        // chars when the byte isn't ASCII).
        if bytes[i] < 0x80 {
            plain.push(bytes[i] as char);
            i += 1;
        } else {
            let rest = &text[i..];
            let c = rest.chars().next().unwrap();
            plain.push(c);
            i += c.len_utf8();
        }
    }
    if !plain.is_empty() {
        out.push(SpecialChunk::Plain(plain));
    }
    out
}

/// If `text[at..]` starts with one of the recognized special-token literals,
/// return that literal (with its angle brackets). Comparison is exact and
/// case-sensitive to match how specials are stored in the vocabulary.
fn match_special_at(text: &str, at: usize) -> Option<&'static str> {
    let rest = &text[at..];
    for &name in RECOGNIZED_SPECIALS {
        if rest.starts_with(name) {
            return Some(name);
        }
    }
    None
}

/// Like [`tokenize_build_vocab`], but recognizes literal special-token names
/// (`<unk>`, `<stop>`, `<tool>`, `<bos>`, `<user>`, `<ai>`) and emits each
/// as the corresponding special-token id. Prefix with a backslash to
/// escape (`\<unk>` becomes the three ordinary tokens `<`, `unk`, `>`).
///
/// Specials map to ids the model already has registered (those ids are
/// created in `Model::new`); plain text runs go through the normal
/// vocabulary-building tokenizer.
pub fn tokenize_build_vocab_with_specials(text: &str, model: &mut Model) -> Vec<usize> {
    let mut ids = Vec::new();
    for chunk in split_with_specials(text) {
        match chunk {
            SpecialChunk::Special(name) => {
                if let Some(id) = model.get_token_id(name) {
                    ids.push(id);
                }
            }
            SpecialChunk::Plain(s) => {
                ids.extend(tokenize_build_vocab(&s, model));
            }
        }
    }
    ids
}

/// Like [`tokenize_with_vocab`], but recognizes literal special-token names
/// (`<unk>`, `<stop>`, etc.) and emits each as the corresponding special-token
/// id. Prefix with a backslash to escape.
pub fn tokenize_with_vocab_and_specials(text: &str, model: &Model) -> Vec<usize> {
    let unk = model.get_unknown_token_id();
    let mut ids = Vec::new();
    for chunk in split_with_specials(text) {
        match chunk {
            SpecialChunk::Special(name) => {
                ids.push(model.get_token_id(name).unwrap_or(unk));
            }
            SpecialChunk::Plain(s) => {
                ids.extend(tokenize_with_vocab(&s, model));
            }
        }
    }
    ids
}

/// Tokenize a *user prompt* against a frozen vocabulary using greedy
/// longest-prefix decomposition.
///
/// Each piece produced by [`split_tokens`] is first looked up whole; if it
/// is in the vocab, that single id is emitted. Otherwise the piece is
/// scanned left-to-right and at each position the longest prefix that
/// exists in the vocab is consumed. Single characters never present in the
/// vocab map to `<unk>`.
///
/// Examples (assuming default vocab + the words "does", "not", "exist"):
///   * `"doesnotexist"` -> `["does", "not", "exist"]`
///   * `"abc"`          -> `["a", "b", "c"]` (single ASCII letters always
///                         exist in the default vocab)
///   * `"hello"`        -> `["hello"]` if registered, else `["h", "e", "l",
///                         "l", "o"]`
pub fn tokenize_user_prompt(text: &str, model: &Model) -> Vec<usize> {
    let unk = model.get_unknown_token_id();
    let mut ids = Vec::new();
    for piece in split_tokens(text) {
        // Common case: the whole piece is one vocab token.
        if let Some(id) = model.get_token_id(&piece) {
            ids.push(id);
            continue;
        }
        // Otherwise greedily consume the longest prefix that exists in
        // the vocabulary at each position. The default vocab includes all
        // single ASCII letters, digits, and DEFAULT_SYMBOLS, so a single-
        // char fallback nearly always succeeds; anything that doesn't
        // resolve maps to `<unk>`.
        let chars: Vec<char> = piece.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut matched: Option<(usize, usize)> = None;
            for end in (i + 1..=chars.len()).rev() {
                let candidate: String = chars[i..end].iter().collect();
                if let Some(id) = model.get_token_id(&candidate) {
                    matched = Some((id, end));
                    break;
                }
            }
            match matched {
                Some((id, end)) => {
                    ids.push(id);
                    i = end;
                }
                None => {
                    ids.push(unk);
                    i += 1;
                }
            }
        }
    }
    ids
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

    #[test]
    fn specials_aware_maps_literal_specials_to_special_ids() {
        let mut model = Model::new(8, 1, 2);
        tokenize_build_vocab("hi there", &mut model);

        let ids = tokenize_with_vocab_and_specials("hi <stop> there", &model);
        let stop = model.get_stop_token_id();
        // Plain split: "hi" " " "<stop>" " " "there" — but the special is one id.
        // Around the special, the surrounding spaces tokenize as single space tokens.
        assert!(ids.contains(&stop));
        // The literal "<stop>" must NOT appear as a sequence of "<", "stop", ">".
        let lt = model.get_token_id("<").unwrap_or(usize::MAX);
        let gt = model.get_token_id(">").unwrap_or(usize::MAX);
        // No `<` `stop` `>` sequence should appear (we don't even register "stop"
        // as a plain word here, so this is mostly a structural check).
        for w in ids.windows(3) {
            assert!(!(w[0] == lt && w[2] == gt), "literal <stop> leaked through");
        }
    }

    #[test]
    fn specials_aware_escapes_with_backslash() {
        let mut model = Model::new(8, 1, 2);
        // Build vocab using the specials-aware path so escaped `<unk>` registers
        // its component tokens (`<`, `unk`, `>`) into the vocab.
        let escaped_ids = tokenize_build_vocab_with_specials("\\<unk>", &mut model);
        let unk = model.get_unknown_token_id();
        // Escaped form must NOT collapse to a single <unk> token.
        assert_ne!(escaped_ids, vec![unk]);
        // It should produce three plain tokens: `<`, `unk`, `>`.
        let lt = model.get_token_id("<").expect("'<' should be in vocab");
        let unk_word = model.get_token_id("unk").expect("'unk' should be in vocab");
        let gt = model.get_token_id(">").expect("'>' should be in vocab");
        assert_eq!(escaped_ids, vec![lt, unk_word, gt]);

        // And the unescaped form maps to the single special id.
        let plain_ids = tokenize_with_vocab_and_specials("<unk>", &model);
        assert_eq!(plain_ids, vec![unk]);
    }

    #[test]
    fn specials_aware_emits_role_tokens_inline() {
        let mut model = Model::new(8, 1, 2);
        let ids = tokenize_build_vocab_with_specials("<user>hi<ai>hello", &mut model);
        let user = model.get_user_token_id();
        let ai = model.get_ai_token_id();
        assert_eq!(ids.first().copied(), Some(user));
        assert!(ids.contains(&ai));
    }

    #[test]
    fn user_prompt_decomposes_unknown_word_into_known_subwords() {
        let mut model = Model::new(8, 1, 2);
        for w in ["does", "not", "exist"] {
            tokenize_build_vocab(w, &mut model);
        }
        let ids = tokenize_user_prompt("doesnotexist", &model);
        let does = model.get_token_id("does").unwrap();
        let not = model.get_token_id("not").unwrap();
        let exist = model.get_token_id("exist").unwrap();
        assert_eq!(ids, vec![does, not, exist]);
    }

    #[test]
    fn user_prompt_falls_back_to_single_chars_for_unknown_word() {
        // Default vocab includes all single ASCII letters, so an unknown
        // word like "abc" decomposes into ['a', 'b', 'c'].
        let model = Model::new(8, 1, 2);
        let ids = tokenize_user_prompt("abc", &model);
        let a = model.get_token_id("a").unwrap();
        let b = model.get_token_id("b").unwrap();
        let c = model.get_token_id("c").unwrap();
        assert_eq!(ids, vec![a, b, c]);
    }

    #[test]
    fn user_prompt_prefers_whole_word_when_present() {
        let mut model = Model::new(8, 1, 2);
        tokenize_build_vocab("hello", &mut model);
        let ids = tokenize_user_prompt("hello", &model);
        let hello = model.get_token_id("hello").unwrap();
        assert_eq!(ids, vec![hello]);
    }

    #[test]
    fn user_prompt_picks_longest_matching_prefix() {
        let mut model = Model::new(8, 1, 2);
        // Register both "do" and "does" so greedy longest-match must pick
        // "does" rather than "do" + "es".
        for w in ["do", "does", "not"] {
            tokenize_build_vocab(w, &mut model);
        }
        let ids = tokenize_user_prompt("doesnot", &model);
        let does = model.get_token_id("does").unwrap();
        let not = model.get_token_id("not").unwrap();
        assert_eq!(ids, vec![does, not]);
    }
}
