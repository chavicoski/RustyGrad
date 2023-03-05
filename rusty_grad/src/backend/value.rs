use by_address::ByAddress;
use core::fmt;
use std::collections::HashSet;
use std::{cell::RefCell, rc::Rc};

pub struct Value {
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<Rc<RefCell<Value>>>,
    pub backward_fn: Box<dyn Fn(&Value)>,
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
