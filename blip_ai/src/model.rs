use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Embedding {
    id: String,
    usage_count: u32,
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    embedding_dim: usize,
    embeddings: Vec<Embedding>,
    network: Vec<Vec<f32>>,
}

impl Model {
    pub fn new(embedding_dim: usize) -> Self {
        Model {
            embedding_dim,
            embeddings: Vec::new(),
            network: Vec::new(),
        }
    }

    pub fn load(file_name: &str) -> Result<Self, String> {
        let file = std::fs::File::open(file_name).map_err(|e| e.to_string())?;
        let model: Model = serde_json::from_reader(file).map_err(|e| e.to_string())?;

        Ok(model)
    }

    pub fn save(&self, file_name: &str) -> Result<(), String> {
        let file = std::fs::File::create(file_name).map_err(|e| e.to_string())?;
        serde_json::to_writer(file, self).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn get_token_id(&self, token: &str) -> Option<usize> {
        self.embeddings.iter().position(|e| e.id == token.to_string())
    }

    pub fn register_token(&mut self, id: String) -> usize {
        if let Some(index) = self.get_token_id(&id) {
            self.embeddings[index].usage_count += 1;

            return index;
        }

        let new_index = self.embeddings.len();

        self.embeddings.push(Embedding {
            id,
            usage_count: 1,
            embedding: vec![0.0; self.embedding_dim],
        });

        new_index
    }

    pub fn initialize_embeddings(&mut self) {
        for embedding in &mut self.embeddings {
            for i in 0..self.embedding_dim {
                embedding.embedding[i] = rand::random::<f32>() * 2.0 - 1.0; // Random values between -1.0 and 1.0
            }
        }
    }
}