use crate::backend::value::Value;
use crate::nn::{components::Module, layers::Dense};
use std::{cell::RefCell, rc::Rc};

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
    fn parameters(&self) -> Vec<Rc<RefCell<Value>>> {
        self.layers
            .iter()
            .map(|l| l.parameters())
            .collect::<Vec<Vec<Rc<RefCell<Value>>>>>() // Vector of parameters' vectors
            .concat()
    }

    fn forward(&self, x: &Vec<Rc<RefCell<Value>>>) -> Vec<Rc<RefCell<Value>>> {
        self.layers
            .iter()
            .fold(x.clone(), |input, l| l.forward(&input))
            .to_vec()
    }
}
