use crate::backend::tensor::RTensor;
use crate::nn::{components::Module, layers::Dense};

pub struct MLP {
    layers: Vec<Dense>,
}

impl MLP {
    pub fn new(n_in: usize, n_units: Vec<usize>) -> Self {
        let mut layers: Vec<Dense> = vec![];
        let mut aux_in = n_in;
        for aux_out in n_units.iter() {
            layers.push(Dense::new(aux_in, *aux_out));
            aux_in = *aux_out;
        }
        MLP { layers }
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<RTensor> {
        self.layers
            .iter()
            .map(|l| l.parameters())
            .collect::<Vec<Vec<RTensor>>>() // Vector of parameters' vectors
            .concat()
    }

    fn forward(&self, x: &RTensor) -> RTensor {
        self.layers
            .iter()
            .fold(x.clone(), |input, l| l.forward(&input))
    }
}
