use std::cell::RefCell;

use rusty_grad::{add, mul, tanh, Value};

fn main() {
    // Inputs
    let x1 = RefCell::new(Value::new(2.0));
    let x2 = RefCell::new(Value::new(0.0));

    // Weights
    let w1 = RefCell::new(Value::new(-3.0));
    let w2 = RefCell::new(Value::new(1.0));
    // Bias
    let b = RefCell::new(Value::new(6.8813735870195432));

    // Compute: x1*w1 + x2*w2 + b
    let x1w1 = mul(&x1, &w1);
    let x2w2 = mul(&x2, &w2);
    let x1w1_x2w2 = add(&x1w1, &x2w2);
    let n = add(&x1w1_x2w2, &b);
    let out = tanh(&n).into_inner();
    println!("output = {out}");
}
