use crate::backend::tensor::Tensor;
use ndarray::prelude::*;
use std::{cell::RefCell, rc::Rc};

pub fn add(t1: Rc<RefCell<Tensor>>, t2: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    Rc::new(RefCell::new(Tensor {
        data: &t1.borrow().data + &t2.borrow().data,
        grad: Array::zeros(t1.borrow().grad.raw_dim()),
        prev: vec![t1.clone(), t2.clone()],
        backward_fn: Box::new(add_backward),
    }))
}

fn add_backward(t: &Tensor) {
    for child in t.prev.iter() {
        child.borrow_mut().grad += &t.grad;
    }
}

pub fn diff(t1: Rc<RefCell<Tensor>>, t2: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    add(
        t1.clone(),
        mul(
            t2,
            Tensor::new_rc(&Array::from_elem(t1.borrow().grad.raw_dim(), -1.0)),
        ),
    )
}

pub fn mul(t1: Rc<RefCell<Tensor>>, t2: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    Rc::new(RefCell::new(Tensor {
        data: &t1.borrow().data * &t2.borrow().data,
        grad: Array::zeros(t1.borrow().grad.raw_dim()),
        prev: vec![t1.clone(), t2.clone()],
        backward_fn: Box::new(mul_backward),
    }))
}

fn mul_backward(t: &Tensor) {
    match &t.prev[..] {
        [t1, t2] => {
            let mut t1 = t1.borrow_mut();
            let mut t2 = t2.borrow_mut();
            t1.grad += &(&t.grad * &t2.data);
            t2.grad += &(&t.grad * &t1.data);
        }
        _ => panic!(
            "[Error] The number of children in Mul op must be 2, but is {})!",
            t.prev.len()
        ),
    }
}

pub fn div(t1: Rc<RefCell<Tensor>>, t2: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    mul(t1, pow(t2, -1.0))
}

pub fn pow(t1: Rc<RefCell<Tensor>>, power: f32) -> Rc<RefCell<Tensor>> {
    Rc::new(RefCell::new(Tensor {
        data: t1.borrow().data.mapv(|x| x.powf(power)),
        grad: Array::zeros(t1.borrow().grad.raw_dim()),
        prev: vec![t1.clone()],
        backward_fn: Box::new(move |x| pow_backward(x, power)),
    }))
}

fn pow_backward(t: &Tensor, power: f32) {
    match &t.prev[..] {
        [prev] => {
            let mut mut_prev = prev.borrow_mut();
            mut_prev.grad +=
                &(&t.grad * (prev.borrow().data.mapv(|x| x.powf(power - 1.0))) * power);
        }
        _ => panic!(
            "[Error] The number of children in Pow op must be 1, but is {})!",
            t.prev.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn add_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![6., 5., 4., 3., 2., 1.]).unwrap();
        let t1 = Tensor::new_rc(&arr1);
        let t2 = Tensor::new_rc(&arr2);
        let t3 = add(t1, t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![7., 7., 7., 7., 7., 7.]).unwrap()
        );
    }

    #[test]
    fn add_backward_ok() {
        let t1 = Tensor::new_rc(&array![[1., 2., -3.], [4., 5., 6.]]);
        let t2 = Tensor::new_rc(&array![[-6., 5., 4.], [3., -2., 1.]]);
        let t3 = add(t1, t2);
        assert_eq!(t3.data, array![[-5., 7., 1.], [7., 3., 7.]]);

        t3.backward();
        let target_grad = Array::<f32, _>::ones(t1.grad.raw_dim());
        assert_eq!(t1.grad, target_grad);
        assert_eq!(t2.grad, target_grad);
    }

    #[test]
    fn diff_ok() {
        let t1 = Tensor::new_rc(&array![[1., 2., 3.], [4., 5., 6.]]);
        let t2 = Tensor::new_rc(&array![[6., 5., 4.], [3., 2., 1.]]);
        let t3 = diff(t1, t2);
        assert_eq!(t3.data, array![[-5., -3., -1.], [1., 3., 5.]]);
    }

    #[test]
    fn diff_backward_ok() {
        let t1 = Tensor::new_rc(&array![[1., 2., -3.], [4., 5., 6.]]);
        let t2 = Tensor::new_rc(&array![[-6., 5., 4.], [3., -2., 1.]]);
        let t3 = diff(t1, t2);
        assert_eq!(t3.data, array![[7., -3., -7.], [1., 7., 5.]]);

        t3.backward();
        let target_grad = Array::<f32, _>::ones(t1.grad.raw_dim());
        assert_eq!(t1.grad, target_grad);
        assert_eq!(t2.grad, -target_grad);
    }

    #[test]
    fn mul_ok() {
        let t1 = Tensor::new_rc(&array![[1., 2., 3.], [4., 5., 6.]]);
        let t2 = Tensor::new_rc(&array![[6., 5., 4.], [3., 2., 1.]]);
        let t3 = mul(t1, t2);
        assert_eq!(t3.data, array![[6., 10., 12.], [12., 10., 6.]]);
    }

    #[test]
    fn mul_backward_ok() {
        let arr1 = array![[1., 2., -3.], [4., 5., 6.]];
        let arr2 = array![[-6., 5., 4.], [3., -2., 1.]];
        let t1 = Tensor::new_rc(&arr1);
        let t2 = Tensor::new_rc(&arr2);
        let t3 = mul(t1, t2);
        assert_eq!(t3.data, array![[-6., 10., -12.], [12., -10., 6.]]);

        t3.backward();
        assert_eq!(t1.grad, arr2);
        assert_eq!(t2.grad, arr1);
    }

    #[test]
    fn div_ok() {
        let t1 = Tensor::new_rc(&array![[1., 2., 9.], [10., 1., 6.]]);
        let t2 = Tensor::new_rc(&array![[2., 2., 3.], [5., 4., 1.]]);
        let t3 = div(t1, t2);
        assert_eq!(t3.data, array![[0.5, 1., 3.], [2., 0.25, 6.]]);
    }

    #[test]
    fn div_backward_ok() {
        let t1 = Tensor::new_rc(&array![[1., 2., 9.], [10., 1., 6.]]);
        let t2 = Tensor::new_rc(&array![[2., 2., 3.], [5., 4., 1.]]);
        let t3 = div(t1, t2);
        assert_eq!(t3.data, array![[0.5, 1., 3.], [2., 0.25, 6.]]);

        t3.backward();
        assert_eq!(t1.grad, array![[1., 1., 1.], [1., 1., 1.]]);
        assert_eq!(t2.grad, array![[1., 1., 1.], [1., 1., 1.]]);
    }

    #[test]
    fn pow_ok() {
        let t = Tensor::new_rc(&array![[1., 2., 4.], [3., 5., 6.]]);
        let res = pow(t, 2.0);
        assert_eq!(res.data, array![[1., 4., 16.], [9., 25., 36.]]);
    }

    #[test]
    fn pow_backward_ok() {
        let t = Tensor::new_rc(&array![[1., 2., 4.], [3., 5., 6.]]);
        let res = pow(t, 2.0);
        assert_eq!(res.data, array![[1., 4., 16.], [9., 25., 36.]]);

        res.backward();
        assert_eq!(res.grad, array![[2., 4., 8.], [6., 10., 12.]]);
    }
}
