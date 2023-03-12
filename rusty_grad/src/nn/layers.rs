use ndarray::prelude::*;

use crate::backend::tensor::Tensor;
use crate::nn::components::Module;
use rand::{distributions::Uniform, prelude::Distribution};
use std::{cell::RefCell, rc::Rc};

pub struct Dense {
    w: Rc<RefCell<Tensor>>,
    b: Rc<RefCell<Tensor>>,
}

impl Dense {
    pub fn new(n_in: usize, n_out: usize, non_lin: bool) -> Self {
        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = rand::thread_rng();
        Dense {
            w: Tensor::new_rc(
                &ArrayD::from_shape_vec(
                    IxDyn(&[n_in, n_out]),
                    (0..n_in * n_out)
                        .map(|_| uniform.sample(&mut rng).into())
                        .collect(),
                )
                .unwrap(),
            ),
            b: Tensor::new_rc(
                &Array::from_shape_vec(
                    IxDyn(&[n_out]),
                    (0..n_out)
                        .map(|_| uniform.sample(&mut rng).into())
                        .collect(),
                )
                .unwrap(),
            ),
        }
    }
}

impl Module for Dense {
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.w.clone(), self.b.clone()]
    }
    fn forward(&self, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Rc<RefCell<Tensor>>> {
        todo!("Implement dot product + bias")
    }
}
