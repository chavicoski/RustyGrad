use by_address::ByAddress;
use core::fmt;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub struct Value {
    pub data: f32,
    pub grad: f32,
    prev: Vec<Rc<RefCell<Value>>>,
    backward_fn: Box<dyn Fn(&Value)>,
}

impl Value {
    pub fn new(value: f32) -> Self {
        Value {
            data: value,
            grad: 0.0,
            prev: vec![],
            backward_fn: Box::new(|_| ()),
        }
    }

    pub fn new_rc(value: f32) -> Rc<RefCell<Value>> {
        Rc::new(RefCell::new(Self::new(value)))
    }

    fn topological_sort(
        value: Rc<RefCell<Value>>,
        topo: &mut Vec<Rc<RefCell<Value>>>,
        visited: &mut HashSet<ByAddress<Rc<RefCell<Value>>>>,
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
        let mut visited: HashSet<ByAddress<Rc<RefCell<Value>>>> = HashSet::new();
        // Stores the values in topological order
        let mut topo: Vec<Rc<RefCell<Value>>> = vec![];

        // Compute the topological order from the childs of `self`. We already know
        // that `self` must be the fist value in topological order
        for child in &self.prev {
            Self::topological_sort(child.clone(), &mut topo, &mut visited);
        }

        // Apply the backpropagation in topological order (from parents to childs)
        self.grad = 1.0;
        (self.backward_fn)(self);
        for v in topo.iter().rev() {
            let v = v.borrow();
            (v.backward_fn)(&v);
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data, self.grad)
    }
}

pub fn add(v1: &Rc<RefCell<Value>>, v2: &Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    Rc::new(RefCell::new(Value {
        data: v1.borrow().data + v2.borrow().data,
        grad: 0.0,
        prev: vec![v1.clone(), v2.clone()],
        backward_fn: Box::new(add_backward),
    }))
}

fn add_backward(value: &Value) {
    for child in value.prev.iter() {
        child.borrow_mut().grad += value.grad;
    }
}

pub fn mul(v1: &Rc<RefCell<Value>>, v2: &Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    Rc::new(RefCell::new(Value {
        data: v1.borrow().data * v2.borrow().data,
        grad: 0.0,
        prev: vec![v1.clone(), v2.clone()],
        backward_fn: Box::new(mul_backward),
    }))
}

fn mul_backward(value: &Value) {
    match &value.prev[..] {
        [v1, v2] => {
            let mut v1 = v1.borrow_mut();
            let mut v2 = v2.borrow_mut();
            v1.grad += value.grad * v2.data;
            v2.grad += value.grad * v1.data;
        }
        _ => panic!(
            "[Error] The number of children in Mul op must be 2, but is {})!",
            value.prev.len()
        ),
    }
}

pub fn tanh(value: &Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    let aux_exp = f32::exp(2.0 * value.borrow().data);
    let t = (aux_exp - 1.0) / (aux_exp + 1.0);
    Rc::new(RefCell::new(Value {
        data: t,
        grad: 0.0,
        prev: vec![value.clone()],
        backward_fn: Box::new(tanh_backward),
    }))
}

fn tanh_backward(value: &Value) {
    match &value.prev[..] {
        [v] => {
            let mut v = v.borrow_mut();
            v.grad += value.grad * (1.0 - f32::powi(value.data, 2));
        }
        _ => panic!(
            "[Error] The number of children in Tanh op must be 1, but is {}!",
            value.prev.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_ok() {
        let v1 = Value::new_rc(27.0);
        let v2 = Value::new_rc(23.0);
        let v3 = add(&v1, &v2);
        assert_eq!(v3.borrow().data, 50.0);
    }

    #[test]
    fn add_backward_ok() {
        let v1 = Value::new_rc(21.0);
        let v2 = Value::new_rc(12.0);
        let v3 = add(&v1, &v2);
        assert_eq!(v3.borrow().data, 33.0);

        v3.borrow_mut().backward();
        assert_eq!(v1.borrow().grad, 1.0);
        assert_eq!(v2.borrow().grad, 1.0);
    }

    #[test]
    fn mul_ok() {
        let v1 = Value::new_rc(2.0);
        let v2 = Value::new_rc(6.0);
        let v3 = mul(&v1, &v2);
        assert_eq!(v3.borrow().data, 12.0);
    }

    #[test]
    fn mul_backward_ok() {
        let v1 = Value::new_rc(2.0);
        let v2 = Value::new_rc(6.0);
        let v3 = mul(&v1, &v2);
        assert_eq!(v3.borrow().data, 12.0);

        v3.borrow_mut().backward();
        assert_eq!(v1.borrow().grad, 6.0);
        assert_eq!(v2.borrow().grad, 2.0);
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
