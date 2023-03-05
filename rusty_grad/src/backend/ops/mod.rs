use crate::backend::value::Value;
use std::cell::RefCell;
use std::rc::Rc;

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

pub fn diff(v1: &Rc<RefCell<Value>>, v2: &Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    add(v1, &mul(v2, &Value::new_rc(-1.0)))
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

pub fn pow(v1: &Rc<RefCell<Value>>, power: f32) -> Rc<RefCell<Value>> {
    Rc::new(RefCell::new(Value {
        data: v1.borrow().data.powf(power),
        grad: 0.0,
        prev: vec![v1.clone(), Rc::new(RefCell::new(Value::new(power)))],
        backward_fn: Box::new(pow_backward),
    }))
}

fn pow_backward(value: &Value) {
    match &value.prev[..] {
        [v, power] => {
            let mut v = v.borrow_mut();
            let power = power.borrow();
            v.grad += value.grad * (power.data * v.data.powf(power.data - 1.0));
        }
        _ => panic!(
            "[Error] The number of children in Pow op must be 2, but is {})!",
            value.prev.len()
        ),
    }
}

pub fn squared_error(
    y_true: &Rc<RefCell<Value>>,
    y_pred: &Rc<RefCell<Value>>,
) -> Rc<RefCell<Value>> {
    pow(&diff(y_true, y_pred), 2.0)
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
    fn diff_ok() {
        let v1 = Value::new_rc(27.0);
        let v2 = Value::new_rc(23.0);
        let v3 = diff(&v1, &v2);
        assert_eq!(v3.borrow().data, 4.0);
    }

    #[test]
    fn diff_backward_ok() {
        let v1 = Value::new_rc(21.0);
        let v2 = Value::new_rc(12.0);
        let v3 = diff(&v1, &v2);
        assert_eq!(v3.borrow().data, 9.0);

        v3.borrow_mut().backward();
        assert_eq!(v1.borrow().grad, 1.0);
        assert_eq!(v2.borrow().grad, -1.0);
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
    fn pow_ok() {
        let v = Value::new_rc(3.0);
        let res = pow(&v, 2.0);
        assert_eq!(res.borrow().data, 9.0);
    }

    #[test]
    fn pow_backward_ok() {
        let v = Value::new_rc(3.0);
        let res = pow(&v, 2.0);
        assert_eq!(res.borrow().data, 9.0);

        res.borrow_mut().backward();
        assert_eq!(v.borrow().grad, 6.0);
    }
}
