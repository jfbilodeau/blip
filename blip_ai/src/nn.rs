use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl Layer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();
        let weights = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| rand::random::<f32>() - 0.5).collect())
            .collect();
        let bias = vec![0.0; output_dim];
        Layer { weights, bias }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(w, b)| {
                w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum::<f32>() + b
            })
            .map(|x| x.max(0.0)) // ReLU
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(sizes: &[usize]) -> Self {
        let layers = sizes.windows(2)
            .map(|w| Layer::new(w[0], w[1]))
            .collect();
        NeuralNetwork { layers }
    }

    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            input = layer.forward(&input);
        }
        input
    }

    pub fn train_step(&mut self, input: &[f32], target: &[f32], learning_rate: f32) {
        // 1. Propagation avant : stocke les activations et entrées de chaque couche
        let mut activations = vec![input.to_vec()];
        let mut pre_activations = Vec::new();

        let mut x = input.to_vec();
        for layer in &self.layers {
            // Calcul linéaire
            let z: Vec<f32> = layer.weights.iter()
                .zip(layer.bias.iter())
                .map(|(w, b)| w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f32>() + b)
                .collect();
            pre_activations.push(z.clone());
            // ReLU
            x = z.iter().map(|v| v.max(0.0)).collect();
            activations.push(x.clone());
        }

        // 2. Calcul du gradient de la perte (MSE)
        let mut delta: Vec<f32> = activations.last().unwrap()
            .iter()
            .zip(target.iter())
            .map(|(o, t)| 2.0 * (o - t))
            .collect();

        // 3. Rétropropagation
        for l in (0..self.layers.len()).rev() {
            // Dérivée de ReLU
            let relu_grad: Vec<f32> = pre_activations[l].iter().map(|&z| if z > 0.0 { 1.0 } else { 0.0 }).collect();
            let delta_relu: Vec<f32> = delta.iter().zip(relu_grad.iter()).map(|(d, r)| d * r).collect();

            // Gradient pour les poids et biais
            let a_prev = &activations[l];
            for i in 0..self.layers[l].weights.len() {
                for j in 0..self.layers[l].weights[i].len() {
                    self.layers[l].weights[i][j] -= learning_rate * delta_relu[i] * a_prev[j];
                }
                self.layers[l].bias[i] -= learning_rate * delta_relu[i];
            }

            // Calcul du delta pour la couche précédente (si ce n'est pas la première)
            if l > 0 {
                let mut new_delta = vec![0.0; self.layers[l].weights[0].len()];
                for j in 0..new_delta.len() {
                    for i in 0..self.layers[l].weights.len() {
                        new_delta[j] += self.layers[l].weights[i][j] * delta_relu[i];
                    }
                }
                delta = new_delta;
            }
        }
    }
}