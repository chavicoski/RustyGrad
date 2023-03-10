use crate::backend::tensor::Tensor;
use ndarray::Array;
use std::{cell::RefCell, rc::Rc};

pub fn relu(t: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let t_data = t.borrow().data;
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
    let res = (aux_exp - 1.) / (aux_exp + 1.);
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

    #[test]
    fn relu_ok() {
        // Test negative input
        let v1 = Value::new_rc(-2.0);
        let v2 = relu(&v1);
        assert_eq!(v2.borrow().data, 0.0);

        // Test positive input
        let v1 = Value::new_rc(0.7);
        let v2 = relu(&v1);
        assert_eq!(v2.borrow().data, 0.7);
    }

    #[test]
    fn relu_backward_ok() {
        // Test negative input
        let input = Value::new_rc(-1.0);
        let out = relu(&input);
        assert_eq!(out.borrow().data, 0.0);

        out.borrow_mut().backward();
        assert_eq!(input.borrow().grad, 0.0);

        // Test positive input
        let input = Value::new_rc(0.8814);
        let out = relu(&input);
        assert_eq!(out.borrow().data, 0.8814);

        out.borrow_mut().backward();
        assert_eq!(input.borrow().grad, 1.0);
    }

    #[test]
    fn tanh_ok() {
        let v1 = Value::new_rc(0.0);
        let v2 = tanh(&v1);
        assert_eq!(v2.borrow().data, 0.0);

        let v1 = Value::new_rc(0.7);
        let v2 = tanh(&v1);
        assert_eq!(v2.borrow().data, 0.6043678);
    }

    #[test]
    fn tanh_backward_ok() {
        let input = Value::new_rc(0.8814);
        let out = tanh(&input);
        assert_eq!(out.borrow().data, 0.70712);

        out.borrow_mut().backward();
        assert_eq!(input.borrow().grad, 0.49998128);
    }
}
