use core::fmt;

pub struct Value<'a> {
    data: f32,
    grad: f32,
    prev: Option<Vec<&'a mut Value<'a>>>,
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

    pub fn get_data(&self) -> f32 {
        self.data
    }

    pub fn set_data(&mut self, data: f32) {
        self.data = data;
    }

    pub fn get_grad(&self) -> f32 {
        self.grad
    }

    pub fn set_grad(&mut self, grad: f32) {
        self.grad = grad;
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data, self.grad)
    }
}

pub fn add<'a>(v1: &'a mut Value<'a>, v2: &'a mut Value<'a>) -> Value<'a> {
    Value {
        data: v1.data + v2.data,
        grad: 0.0,
        prev: Some(vec![v1, v2]),
        op: Some(Op::Add),
    }
}

fn add_backward(value: &mut Value) {
    match value.prev {
        Some(ref mut children) => {
            for child in children.iter_mut() {
                child.set_grad(value.grad);
            }
        }
        _ => println!("[Warning] Calling add_backward without children"),
    }
}

pub fn mul<'a>(v1: &'a mut Value<'a>, v2: &'a mut Value<'a>) -> Value<'a> {
    Value {
        data: v1.data * v2.data,
        grad: 0.0,
        prev: Some(vec![v1, v2]),
        op: Some(Op::Mul),
    }
}

fn mul_backward(value: &mut Value) {
    match value.prev {
        Some(ref mut children) => match children[..] {
            [ref mut v1, ref mut v2] => {
                v1.set_grad(value.grad * v2.data);
                v2.set_grad(value.grad * v1.data);
            }
            _ => panic!("[Error] The number of children in Mul op is not 2!"),
        },
        _ => println!("[Warning] Calling mul_backward without children"),
    }
}

pub fn tanh<'a>(value: &'a mut Value<'a>) -> Value<'a> {
    let aux_exp = f32::exp(2.0 * value.data);
    let t = (aux_exp - 1.0) / (aux_exp + 1.0);
    Value {
        data: t,
        grad: 0.0,
        prev: Some(vec![value]),
        op: Some(Op::Tanh),
    }
}

fn tanh_backward(value: &mut Value) {
    match value.prev {
        Some(ref mut children) => match children[..] {
            [ref mut v] => {
                v.set_grad(value.grad * (1.0 - f32::powi(v.data, 2)));
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
        let mut v1 = Value::new(27.0);
        let mut v2 = Value::new(23.0);
        let v3 = add(&mut v1, &mut v2);
        assert_eq!(v3.data, 50.0);
    }

    #[test]
    fn mul_ok() {
        let mut v1 = Value::new(2.0);
        let mut v2 = Value::new(6.0);
        let v3 = mul(&mut v1, &mut v2);
        assert_eq!(v3.data, 12.0);
    }

    #[test]
    fn tanh_ok() {
        let mut v1 = Value::new(0.0);
        let v2 = tanh(&mut v1);
        assert_eq!(v2.data, 0.0);

        let mut v1 = Value::new(0.7);
        let v2 = tanh(&mut v1);
        assert_eq!(v2.data, 0.6043678);
    }

    #[test]
    fn tanh_backward_ok() {
        let mut input = Value::new(0.8814);
        let mut out = tanh(&mut input);
        assert_eq!(out.data, 0.70712);

        out.set_grad(1.0);
        tanh_backward(&mut out);
        // TODO Fix this: assert_eq!(input.get_grad(), 0.5);
    }
}
