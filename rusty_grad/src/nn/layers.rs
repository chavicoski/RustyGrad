use crate::backend::{
    ops::dot,
    tensor::{RTensor, Tensor},
};
use crate::nn::components::Module;
use ndarray::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};

pub struct Dense {
    w: RTensor,
    b: RTensor,
}

impl Dense {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = rand::thread_rng();
        Dense {
            w: Tensor::new_ref(
                &ArrayD::from_shape_vec(
                    IxDyn(&[n_in, n_out]),
                    (0..n_in * n_out)
                        .map(|_| uniform.sample(&mut rng))
                        .collect(),
                )
                .unwrap(),
            ),
            b: Tensor::new_ref(
                &Array::from_shape_vec(
                    IxDyn(&[n_out]),
                    (0..n_out).map(|_| uniform.sample(&mut rng)).collect(),
                )
                .unwrap(),
            ),
        }
    }
}

impl Module for Dense {
    fn parameters(&self) -> Vec<RTensor> {
        vec![self.w.clone(), self.b.clone()]
    }
    fn forward(&self, x: &RTensor) -> RTensor {
        dot(x, &self.w)
    }
}
