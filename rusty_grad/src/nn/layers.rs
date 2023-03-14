use crate::backend::{ops::dot, tensor::Tensor};
use crate::nn::components::Module;
use ndarray::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};
use std::{cell::RefCell, rc::Rc};

pub struct Dense {
    w: Rc<RefCell<Tensor>>,
    b: Rc<RefCell<Tensor>>,
    non_lin: bool,
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
            non_lin,
        }
    }
}

impl Module for Dense {
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.w.clone(), self.b.clone()]
    }
    fn forward(&self, x: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        dot(x, &self.w)
        /*
        println!("res = {:?}", res.borrow().data);
        let x = x.borrow();
        let w = self.w.borrow();
        let x_2d = x.data.slice(s![.., ..]);
        let w_2d = w.data.slice(s![.., ..]);
        let out = x_2d.dot(&w_2d);
        println!("x = {:?}", x.data);
        println!("w = {:?}", w.data);
        println!("out = {:?}", out);
        todo!("Implement Dot op (with backward). Implement a generic version (maybe with .broadcast())")
        */
    }
}
