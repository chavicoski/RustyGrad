use by_address::ByAddress;
use core::fmt;
use ndarray::prelude::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub struct Tensor {
    pub data: ArrayD<f32>,
    pub grad: ArrayD<f32>,
    pub prev: Vec<RTensor>,
    pub backward_fn: Box<dyn Fn(&Tensor)>,
}

pub type RTensor = Rc<RefCell<Tensor>>;

impl Tensor {
    pub fn new(data: &ArrayD<f32>) -> Self {
        Tensor {
            data: data.clone(),
            grad: Array::zeros(data.raw_dim()),
            prev: vec![],
            backward_fn: Box::new(|_| ()),
        }
    }

    pub fn new_ref(data: &ArrayD<f32>) -> RTensor {
        Rc::new(RefCell::new(Self::new(data)))
    }

    pub fn to_ref(self) -> RTensor {
        Rc::new(RefCell::new(self))
    }

    fn topological_sort(
        value: RTensor,
        topo: &mut Vec<RTensor>,
        visited: &mut HashSet<ByAddress<RTensor>>,
    ) {
        if !visited.contains(&ByAddress(value.clone())) {
            visited.insert(ByAddress(value.clone()));
            for child in &value.borrow().prev {
                Self::topological_sort(child.clone(), topo, visited);
            }
            topo.push(value);
        }
    }

    pub fn backward(&mut self) {
        // Tracks the already visited values
        let mut visited: HashSet<ByAddress<RTensor>> = HashSet::new();
        // Stores the values in topological order
        let mut topo: Vec<RTensor> = vec![];

        // Compute the topological order from the childs of `self`. We already know
        // that `self` must be the fist value in topological order
        for child in &self.prev {
            Self::topological_sort(child.clone(), &mut topo, &mut visited);
        }

        // Set the initial gradients to 1.0 to start the backpropagation
        self.grad.fill(1.0);
        // Apply the backpropagation in topological order (from parents to childs)
        (self.backward_fn)(self);
        for v in topo.iter().rev() {
            (v.borrow().backward_fn)(&v.borrow());
        }
    }
}

#[macro_export]
macro_rules! tensor {
	(&[$($s:expr),*], &[$($d:expr),*]) => {
        {
            let shape = IxDyn(&[$($s,)*]);
            let data = vec![$($d,)*];
            let data_array = &ArrayD::from_shape_vec(shape, data).unwrap();
            Tensor::new(data_array)
        }
	};
}

#[macro_export]
macro_rules! rtensor {
	(&[$($s:expr),*], &[$($d:expr),*]) => {
        {
            let shape = IxDyn(&[$($s,)*]);
            let data = vec![$($d,)*];
            let data_array = &ArrayD::from_shape_vec(shape, data).unwrap();
            Tensor::new_ref(data_array)
        }
	};
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(data={}, shape={:?})",
            self.data,
            self.data.shape()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_array(shape: &[usize]) -> ArrayD<f32> {
        ArrayD::zeros(IxDyn(shape))
    }

    #[test]
    fn new_ok() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        let t = Tensor::new(&arr);

        assert_eq!(t.data, arr);
        assert_eq!(t.grad, zero_array(shape));
    }

    #[test]
    fn new_ref_ok() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        let t = Tensor::new_ref(&arr);

        assert_eq!(t.borrow().data, arr);
        assert_eq!(t.borrow().grad, zero_array(shape));
    }

    #[test]
    fn macro_tensor_ok() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        let t = tensor![&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];

        assert_eq!(t.data, arr);
        assert_eq!(t.grad, zero_array(shape));
    }

    #[test]
    fn macro_rtensor_ok() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        let t = rtensor![&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];

        assert_eq!(t.borrow().data, arr);
        assert_eq!(t.borrow().grad, zero_array(shape));
    }
}
