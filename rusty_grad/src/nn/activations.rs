use crate::backend::value::Value;
use std::cell::RefCell;
use std::rc::Rc;

pub fn relu(value: &Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    let v_data = value.borrow().data;
    Rc::new(RefCell::new(Value {
        data: if v_data > 0.0 { v_data } else { 0.0 },
        grad: 0.0,
        prev: vec![value.clone()],
        backward_fn: Box::new(relu_backward),
    }))
}

fn relu_backward(value: &Value) {
    match &value.prev[..] {
        [v] => {
            let mut v = v.borrow_mut();
            v.grad += value.grad * (v.data > 0.0) as u8 as f32;
        }
        _ => panic!(
            "[Error] The number of children in ReLU op must be 1, but is {}!",
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
