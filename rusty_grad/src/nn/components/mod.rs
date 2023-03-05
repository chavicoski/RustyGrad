use crate::backend::ops::{add, mul};
use crate::backend::value::Value;
use crate::nn::activations::relu;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use std::cell::RefCell;
use std::rc::Rc;

pub trait Module {
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.borrow_mut().grad = 0.0;
        }
    }
    fn parameters(&self) -> Vec<Rc<RefCell<Value>>>;
    fn forward(&self, x: &Vec<Rc<RefCell<Value>>>) -> Vec<Rc<RefCell<Value>>>;
}

pub struct Neuron {
    weight: Vec<Rc<RefCell<Value>>>,
    bias: Rc<RefCell<Value>>,
    non_lin: bool,
}

impl Neuron {
    pub fn new(n_in: usize, non_lin: bool) -> Self {
        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = rand::thread_rng();
        Neuron {
            weight: (0..n_in)
                .map(|_| Value::new_rc(uniform.sample(&mut rng)))
                .collect(),
            bias: Value::new_rc(uniform.sample(&mut rng)),
            non_lin,
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Rc<RefCell<Value>>> {
        [
            self.weight.iter().map(|v| v.clone()).collect(),
            vec![self.bias.clone()],
        ]
        .concat()
    }
    fn forward(&self, x: &Vec<Rc<RefCell<Value>>>) -> Vec<Rc<RefCell<Value>>> {
        // Compute the product between X and W
        let mut aux = self
            .weight
            .iter()
            .zip(x)
            .map(|(ref w, ref x)| mul(w, x))
            .reduce(|ref v1, ref v2| add(v1, v2))
            .unwrap();
        // Add the Bias
        aux = add(&aux, &self.bias);
        // Apply the non linear function
        if self.non_lin {
            aux = relu(&aux);
        }
        vec![aux]
    }
}
