use crate::backend::tensor::{RTensor, Tensor};
use ndarray::prelude::*;

pub fn add(t1: &RTensor, t2: &RTensor) -> RTensor {
    Tensor {
        data: &t1.borrow().data + &t2.borrow().data,
        grad: Array::zeros(t1.borrow().grad.raw_dim()),
        prev: vec![t1.clone(), t2.clone()],
        backward_fn: Box::new(add_backward),
    }
    .to_ref()
}

fn add_backward(t: &Tensor) {
    for child in t.prev.iter() {
        child.borrow_mut().grad += &t.grad;
    }
}

pub fn diff(t1: &RTensor, t2: &RTensor) -> RTensor {
    add(
        t1,
        &mul(
            t2,
            &Tensor::new_ref(&Array::from_elem(t1.borrow().grad.raw_dim(), -1.0)),
        ),
    )
}

pub fn mul(t1: &RTensor, t2: &RTensor) -> RTensor {
    Tensor {
        data: &t1.borrow().data * &t2.borrow().data,
        grad: Array::zeros(t1.borrow().grad.raw_dim()),
        prev: vec![t1.clone(), t2.clone()],
        backward_fn: Box::new(mul_backward),
    }
    .to_ref()
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

pub fn dot(t1: &RTensor, t2: &RTensor) -> RTensor {
    // Compute the dot product of t1 and t2.
    // Note: They must be converted to a 2D array view (Ix2) to be able to call .dot()
    let out_data = t1
        .borrow()
        .data
        .view()
        .into_dimensionality::<Ix2>()
        .unwrap() // TODO Handle the error
        .dot(
            &t2.borrow()
                .data
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap(), // TODO Handle the error
        )
        .into_dyn(); // Convert back to dynamic array to create the output Tensor
    let out_dim = out_data.raw_dim();
    Tensor {
        data: out_data,
        grad: Array::zeros(out_dim),
        prev: vec![t1.clone(), t2.clone()],
        backward_fn: Box::new(dot_backward),
    }
    .to_ref()
}

fn dot_backward(t: &Tensor) {
    match &t.prev[..] {
        [t1, t2] => {
            let mut t1 = t1.borrow_mut();
            let mut t2 = t2.borrow_mut();
            t1.grad += &(t
                .grad
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .dot(&t2.data.view().into_dimensionality::<Ix2>().unwrap().t()));
            t2.grad += &(t1
                .data
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .t()
                .dot(&t.grad.view().into_dimensionality::<Ix2>().unwrap()));
        }
        _ => panic!(
            "[Error] The number of children in Dot op must be 2, but is {})!",
            t.prev.len()
        ),
    }
}

pub fn div(t1: &RTensor, t2: &RTensor) -> RTensor {
    mul(t1, &pow(t2, -1.0))
}

pub fn pow(t1: &RTensor, power: f32) -> RTensor {
    Tensor {
        data: t1.borrow().data.mapv(|x| x.powf(power)),
        grad: Array::zeros(t1.borrow().grad.raw_dim()),
        prev: vec![t1.clone()],
        backward_fn: Box::new(move |x| pow_backward(x, power)),
    }
    .to_ref()
}

fn pow_backward(t: &Tensor, power: f32) {
    match &t.prev[..] {
        [prev] => {
            let dprev = prev.borrow().data.mapv(|x| x.powf(power - 1.0)) * power;
            prev.borrow_mut().grad += &(&t.grad * &dprev);
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

    #[test]
    fn add_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![6., 5., 4., 3., 2., 1.]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = add(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![7., 7., 7., 7., 7., 7.]).unwrap()
        );
    }

    #[test]
    fn add_backward_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., -3., 4., 5., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-6., 5., 4., 3., -2., 1.]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = add(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-5., 7., 1., 7., 3., 7.]).unwrap()
        );

        t3.borrow_mut().backward();
        let target_grad = Array::<f32, _>::ones(t1.borrow().grad.raw_dim());
        assert_eq!(t1.borrow().grad, target_grad);
        assert_eq!(t2.borrow().grad, target_grad);
    }

    #[test]
    fn diff_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![6., 5., 4., 3., 2., 1.]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = diff(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-5., -3., -1., 1., 3., 5.]).unwrap()
        );
    }

    #[test]
    fn diff_backward_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., -3., 4., 5., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-6., 5., 4., 3., -2., 1.]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = diff(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![7., -3., -7., 1., 7., 5.]).unwrap()
        );

        t3.borrow_mut().backward();
        let target_grad = Array::<f32, _>::ones(t1.borrow().grad.raw_dim());
        assert_eq!(t1.borrow().grad, target_grad);
        assert_eq!(t2.borrow().grad, -target_grad);
    }

    #[test]
    fn mul_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![6., 5., 4., 3., 2., 1.]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = mul(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![6., 10., 12., 12., 10., 6.]).unwrap()
        );
    }

    #[test]
    fn mul_backward_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., -3., 4., 5., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-6., 5., 4., 3., -2., 1.]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = mul(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-6., 10., -12., 12., -10., 6.]).unwrap()
        );

        t3.borrow_mut().backward();
        assert_eq!(t1.borrow().grad, arr2);
        assert_eq!(t2.borrow().grad, arr1);
    }

    #[test]
    fn div_ok() {
        let arr1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 9., 10., 1., 6.]).unwrap();
        let arr2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2., 2., 3., 5., 4., 1.]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = div(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 1., 3., 2., 0.25, 6.]).unwrap()
        );
    }

    #[test]
    fn div_backward_ok() {
        let arr1 =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.3, -0.6, 1.2, -0.6, -0.4, 2.2]).unwrap();
        let arr2 =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.9, 0.2, -0.4, -0.3, -0.2, 2.2]).unwrap();
        let t1 = Tensor::new_ref(&arr1);
        let t2 = Tensor::new_ref(&arr2);
        let t3 = div(&t1, &t2);
        assert_eq!(
            t3.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.33333337, -3., -3., 2., 2., 1.]).unwrap()
        );

        t3.borrow_mut().backward();
        assert_eq!(
            t1.borrow().grad,
            ArrayD::from_shape_vec(
                IxDyn(&[2, 3]),
                vec![1.1111112, 5., -2.5, -3.3333333, -5., 0.45454544]
            )
            .unwrap()
        );
        assert_eq!(
            t2.borrow().grad,
            ArrayD::from_shape_vec(
                IxDyn(&[2, 3]),
                vec![
                    -0.37037042,
                    15.000001,
                    -7.5000005,
                    6.6666665,
                    10.,
                    -0.45454544
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn pow_ok() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 4., 3., 5., 6.]).unwrap();
        let t = Tensor::new_ref(&arr);
        let res = pow(&t, 2.0);
        assert_eq!(
            res.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 4., 16., 9., 25., 36.]).unwrap()
        );
    }

    #[test]
    fn pow_backward_ok() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 4., 3., 5., 6.]).unwrap();
        let t = Tensor::new_ref(&arr);
        let res = pow(&t, 2.0);
        assert_eq!(
            res.borrow().data,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 4., 16., 9., 25., 36.]).unwrap()
        );

        res.borrow_mut().backward();
        assert_eq!(
            t.borrow().grad,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2., 4., 8., 6., 10., 12.]).unwrap()
        );
    }
}
