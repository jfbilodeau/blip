//! End-to-end pipeline test: train a tiny grammar, save, reload, generate.
//!
//! Uses a deterministic 3-token cycle (a -> b -> c -> a -> ...). After enough
//! Adam steps the greedy continuation of `a` should be `b`.

use blip_ai::model::{Model, SamplingConfig};
use blip_ai::nn::{self, Optimizer};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

#[test]
fn train_save_load_generate_roundtrip() {
    nn::seed(7);

    let mut model = Model::new(16, 2, 4);
    for tok in ["a", "b", "c"] {
        model.register_token(tok);
    }
    model.initialize_embeddings();

    let user = model.get_user_token_id();
    let stop = model.get_stop_token_id();
    let a = model.get_token_id("a").unwrap();
    let b = model.get_token_id("b").unwrap();
    let c = model.get_token_id("c").unwrap();

    let mut seq = vec![user];
    for _ in 0..6 {
        seq.extend_from_slice(&[a, b, c]);
    }
    seq.push(stop);

    let opt = Optimizer::adam(0.01);
    let l0 = model.train_sequence(&seq);
    model.apply_grads(opt);
    let mut last = l0;
    for _ in 0..300 {
        last = model.train_sequence(&seq);
        model.apply_grads(opt);
    }
    assert!(
        last < l0 * 0.3,
        "loss did not decrease enough: l0={l0} last={last}"
    );

    // Save -> load roundtrip (bincode).
    let path = std::env::temp_dir().join("blip_e2e_model.bin");
    model.save(path.to_str().unwrap()).expect("save");
    let reloaded = Model::load(path.to_str().unwrap()).expect("load");
    let _ = std::fs::remove_file(&path);

    assert_eq!(reloaded.vocab_size(), model.vocab_size());
    assert_eq!(reloaded.embedding_dim(), model.embedding_dim());
    assert_eq!(reloaded.n_heads(), model.n_heads());

    // Greedy generation from "<user> a" should produce "b" first.
    let cfg = SamplingConfig::greedy(5);
    let mut rng = ChaCha12Rng::seed_from_u64(0);
    let ids = reloaded
        .generate_token_ids(&[user, a], &cfg, &mut rng)
        .expect("generate");
    assert!(!ids.is_empty(), "generation produced nothing");
    assert_eq!(
        ids[0], b,
        "expected 'b' after 'a', got token id {} ({:?})",
        ids[0],
        reloaded.get_token_by_id(ids[0])
    );
}

#[test]
fn vocab_trim_collapses_rare_tokens_to_unk() {
    nn::seed(11);
    let mut model = Model::new(8, 1, 2);
    // "common" appears 3 times, "rare" once.
    model.register_token("common");
    model.register_token("common");
    model.register_token("common");
    model.register_token("rare");

    let removed = model.trim_vocab(2);
    assert!(removed >= 1);
    assert!(model.get_token_id("rare").is_none());
    assert!(model.get_token_id("common").is_some());
    // Specials always survive.
    assert!(model.get_token_id("<unk>").is_some());
    assert!(model.get_token_id("<stop>").is_some());
}

#[test]
fn detokenize_text_handles_punctuation() {
    use blip_ai::tokenizer::{detokenize_text, tokenize};

    let mut model = Model::new(8, 1, 2);
    let ids = tokenize("Hello, world! How are you?", &mut model);
    let text = detokenize_text(&ids, &model);
    assert_eq!(text, "hello, world! how are you?");
}
