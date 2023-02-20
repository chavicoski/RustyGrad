use core::fmt;
use std::cell::RefCell;

pub struct Value<'a> {
    pub data: f32,
    pub grad: f32,
    prev: Option<Vec<&'a RefCell<Value<'a>>>>,
    op: Option<Op>,
}

enum Op {
    Add,
    Mul,
    Tanh,
}

impl Value<'_> {
    pub fn new(value: f32) -> Self {
        Value {
            data: value,
            grad: 0.0,
            prev: None,
            op: None,
        }
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data, self.grad)
    }
}

pub fn add<'a>(v1: &'a RefCell<Value<'a>>, v2: &'a RefCell<Value<'a>>) -> RefCell<Value<'a>> {
    RefCell::new(Value {
        data: v1.borrow().data + v2.borrow().data,
        grad: 0.0,
        prev: Some(vec![v1, v2]),
        op: Some(Op::Add),
    })
}

fn add_backward(value: &RefCell<Value>) {
    match &value.borrow_mut().prev {
        Some(children) => {
            for child in children.iter() {
                child.borrow_mut().grad = value.borrow().grad;
            }
        }
        _ => println!("[Warning] Calling add_backward without children"),
    }
}

pub fn mul<'a>(v1: &'a RefCell<Value<'a>>, v2: &'a RefCell<Value<'a>>) -> RefCell<Value<'a>> {
    RefCell::new(Value {
        data: v1.borrow().data * v2.borrow().data,
        grad: 0.0,
        prev: Some(vec![v1, v2]),
        op: Some(Op::Mul),
    })
}

fn mul_backward(value: &RefCell<Value>) {
    match &value.borrow_mut().prev {
        Some(children) => match &children[..] {
            &[v1, v2] => {
                v1.borrow_mut().grad = value.borrow().grad * v2.borrow().data;
                v2.borrow_mut().grad = value.borrow().grad * v1.borrow().data;
            }
            _ => panic!("[Error] The number of children in Mul op is not 2!"),
        },
        _ => println!("[Warning] Calling mul_backward without children"),
    }
}

pub fn tanh<'a>(value: &'a RefCell<Value<'a>>) -> RefCell<Value<'a>> {
    let aux_exp = f32::exp(2.0 * value.borrow().data);
    let t = (aux_exp - 1.0) / (aux_exp + 1.0);
    RefCell::new(Value {
        data: t,
        grad: 0.0,
        prev: Some(vec![value]),
        op: Some(Op::Tanh),
    })
}

fn tanh_backward(value: &RefCell<Value>) {
    let value = value.borrow_mut();
    match &value.prev {
        Some(children) => match &children[..] {
            &[v] => {
                let mut v = v.borrow_mut();
                v.grad = value.grad * (1.0 - f32::powi(value.data, 2));
            }
            _ => panic!("[Error] The number of children in Tanh op is not 1!"),
        },
        _ => println!("[Warning] Calling tanh_backward without children"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_ok() {
        let v1 = RefCell::new(Value::new(27.0));
        let v2 = RefCell::new(Value::new(23.0));
        let v3 = add(&v1, &v2);
        assert_eq!(v3.borrow().data, 50.0);
    }

    #[test]
    fn mul_ok() {
        let v1 = RefCell::new(Value::new(2.0));
        let v2 = RefCell::new(Value::new(6.0));
        let v3 = mul(&v1, &v2);
        assert_eq!(v3.borrow().data, 12.0);
    }

    #[test]
    fn tanh_ok() {
        let v1 = RefCell::new(Value::new(0.0));
        let v2 = tanh(&v1);
        assert_eq!(v2.borrow().data, 0.0);

        let v1 = RefCell::new(Value::new(0.7));
        let v2 = tanh(&v1);
        assert_eq!(v2.borrow().data, 0.6043678);
    }

    #[test]
    fn tanh_backward_ok() {
        let input = RefCell::new(Value::new(0.8814));
        let out = tanh(&input);
        assert_eq!(out.borrow().data, 0.70712);

        out.borrow_mut().grad = 1.0;
        tanh_backward(&out);
        assert_eq!(input.borrow().grad, 0.5);
    }
}
