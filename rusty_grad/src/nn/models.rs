use crate::backend::tensor::Tensor;
use crate::nn::{components::Module, layers::Dense};
use std::cell::RefCell;
use std::rc::Rc;

pub struct MLP {
    layers: Vec<Dense>,
}

impl MLP {
    pub fn new(n_in: usize, n_units: Vec<usize>) -> Self {
        let mut layers: Vec<Dense> = vec![];
        let mut aux_in = n_in;
        for (i, aux_out) in n_units.iter().enumerate() {
            layers.push(Dense::new(aux_in, *aux_out, i < n_units.len() - 1));
            aux_in = *aux_out;
        }
        MLP { layers }
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        self.layers
            .iter()
            .map(|l| l.parameters())
            .collect::<Vec<Vec<Rc<RefCell<Tensor>>>>>() // Vector of parameters' vectors
            .concat()
    }

    fn forward(&self, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Rc<RefCell<Tensor>>> {
        self.layers
            .iter()
            .fold(x.clone(), |input, l| l.forward(&input))
            .to_vec()
    }
}
