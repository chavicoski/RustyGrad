use crate::backend::tensor::Tensor;
use ndarray::Array;
use std::{cell::RefCell, rc::Rc};

pub fn relu(t: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let t_data = &t.borrow().data;
    Rc::new(RefCell::new(Tensor {
        data: t_data.mapv(|x| if x > 0. { x } else { 0. }),
        grad: Array::zeros(t_data.raw_dim()),
        prev: vec![t.clone()],
        backward_fn: Box::new(relu_backward),
    }))
}

fn relu_backward(t: &Tensor) {
    match &t.prev[..] {
        [t_prev] => {
            let mut t_prev = t_prev.borrow_mut();
            t_prev.grad += &(&t.grad * &t.data.mapv(|x| (x > 0.) as u8 as f32));
        }
        _ => panic!(
            "[Error] The number of children in ReLU op must be 1, but is {}!",
            t.prev.len()
        ),
    }
}

pub fn tanh(t: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let aux_exp = t.borrow().data.mapv(|x| f32::exp(2.0 * x));
    let res = (&aux_exp - 1.) / (&aux_exp + 1.);
    Rc::new(RefCell::new(Tensor {
        data: res,
        grad: Array::zeros(t.borrow().data.raw_dim()),
        prev: vec![t.clone()],
        backward_fn: Box::new(tanh_backward),
    }))
}

fn tanh_backward(t: &Tensor) {
    match &t.prev[..] {
        [t_prev] => {
            let mut t_prev = t_prev.borrow_mut();
            t_prev.grad += &(&t.grad * &t.data.mapv(|x| 1. - f32::powi(x, 2)));
        }
        _ => panic!(
            "[Error] The number of children in Tanh op must be 1, but is {}!",
            t.prev.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn relu_ok() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2., 1.2, -0.3, -1.]).unwrap();
        let t = Tensor::new_rc(&arr);
        let res = relu(t);
        assert_eq!(
            res.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2., 1.2, 0., 0.]).unwrap()
        );
    }

    #[test]
    fn relu_backward_ok() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2., 1.2, -0.3, -1.]).unwrap();
        let t = Tensor::new_rc(&arr);
        let res = relu(t.clone());
        assert_eq!(
            res.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2., 1.2, 0., 0.]).unwrap()
        );
        res.borrow_mut().backward();
        assert_eq!(
            t.borrow().grad,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1., 1., 0., 0.]).unwrap()
        );
    }

    #[test]
    fn tanh_ok() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0., 0.7, 0., 0.7]).unwrap();
        let t = Tensor::new_rc(&arr);
        let res = tanh(t);
        assert_eq!(
            res.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0., 0.6043678, 0., 0.6043678]).unwrap()
        );
    }

    #[test]
    fn tanh_backward_ok() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0., 0.8814, 0., 0.8814]).unwrap();
        let t = Tensor::new_rc(&arr);
        let res = tanh(t.clone());
        assert_eq!(
            res.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0., 0.70712, 0., 0.70712]).unwrap()
        );
        res.borrow_mut().backward();
        assert_eq!(
            t.borrow().grad,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1., 0.49998128, 1., 0.49998128]).unwrap()
        );
    }
}
