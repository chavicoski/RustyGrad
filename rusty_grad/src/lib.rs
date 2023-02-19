use core::fmt;
use std::ops;

pub struct Value<'a> {
    pub data: f32,
    grad: f32,
    prev: Option<Vec<&'a Value<'a>>>,
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

    pub fn set_grad(&mut self, grad: f32) {
        self.grad = grad;
    }

    pub fn tanh<'a>(&'a self) -> Value<'a> {
        let aux_exp = f32::exp(2.0 * self.data);
        let t = (aux_exp - 1.0) / (aux_exp + 1.0);
        Value {
            data: t,
            grad: 0.0,
            prev: Some(vec![self]),
            op: Some(Op::Tanh),
        }
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}

impl<'a> ops::Add<&'a Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;

    fn add(self, rhs: &'a Value) -> Self::Output {
        Value {
            data: self.data + rhs.data,
            grad: 0.0,
            prev: Some(vec![&self, &rhs]),
            op: Some(Op::Add),
        }
    }
}

impl<'a> ops::Mul<&'a Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;

    fn mul(self, rhs: &'a Value) -> Self::Output {
        Value {
            data: self.data * rhs.data,
            grad: 0.0,
            prev: Some(vec![&self, &rhs]),
            op: Some(Op::Mul),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_add_ok() {
        let v1 = Value::new(27.0);
        let v2 = Value::new(23.0);
        let v3 = &v1 + &v2;
        assert_eq!(v3.data, 50.0);
    }

    #[test]
    fn values_mul_ok() {
        let v1 = Value::new(2.0);
        let v2 = Value::new(6.0);
        let v3 = &v1 * &v2;
        assert_eq!(v3.data, 12.0);
    }

    #[test]
    fn value_tanh_ok() {
        let v = Value::new(0.0);
        let tanh = v.tanh();
        assert_eq!(tanh.data, 0.0);

        let v = Value::new(0.7);
        let tanh = v.tanh();
        assert_eq!(tanh.data, 0.6043678);
    }
}
