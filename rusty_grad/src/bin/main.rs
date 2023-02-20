use rusty_grad::{add, mul, tanh, Value};

fn main() {
    // Inputs
    let mut x1 = Value::new(2.0);
    let mut x2 = Value::new(0.0);

    // Weights
    let mut w1 = Value::new(-3.0);
    let mut w2 = Value::new(1.0);
    // Bias
    let mut b = Value::new(6.8813735870195432);

    // Compute: x1*w1 + x2*w2 + b
    let mut x1w1 = mul(&mut x1, &mut w1);
    let mut x2w2 = mul(&mut x2, &mut w2);
    let mut x1w1_x2w2 = add(&mut x1w1, &mut x2w2);
    let mut n = add(&mut x1w1_x2w2, &mut b);
    let out = tanh(&mut n);
    println!("output = {out}");
}
