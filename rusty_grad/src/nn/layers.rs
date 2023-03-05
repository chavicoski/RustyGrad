use crate::backend::value::Value;
use crate::nn::components::{Module, Neuron};
use std::{cell::RefCell, rc::Rc};

pub struct Dense {
    neurons: Vec<Neuron>,
}

impl Dense {
    pub fn new(n_in: usize, n_out: usize, non_lin: bool) -> Self {
        Dense {
            neurons: (0..n_out).map(|_| Neuron::new(n_in, non_lin)).collect(),
        }
    }
}

impl Module for Dense {
    fn parameters(&self) -> Vec<Rc<RefCell<Value>>> {
        self.neurons
            .iter()
            .map(|n| n.parameters())
            .collect::<Vec<Vec<Rc<RefCell<Value>>>>>() // Vector of parameters' vectors
            .concat()
    }
    fn forward(&self, x: &Vec<Rc<RefCell<Value>>>) -> Vec<Rc<RefCell<Value>>> {
        self.neurons
            .iter()
            .map(|n| n.forward(x))
            .collect::<Vec<Vec<Rc<RefCell<Value>>>>>()
            .concat()
    }
}
