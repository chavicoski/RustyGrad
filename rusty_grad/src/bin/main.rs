use rusty_grad::{add, mul, tanh, Value};

fn main() {
    // Inputs
    let x1 = Value::new_rc(2.0);
    let x2 = Value::new_rc(0.0);

    // Weights
    let w1 = Value::new_rc(-3.0);
    let w2 = Value::new_rc(1.0);
    // Bias
    let b = Value::new_rc(6.8813735870195432);

    // Compute: x1*w1 + x2*w2 + b
    let x1w1 = mul(&x1, &w1);
    let x2w2 = mul(&x2, &w2);
    let x1w1_x2w2 = add(&x1w1, &x2w2);
    let n = add(&x1w1_x2w2, &b);
    let out = tanh(&n);
    let mut out = out.borrow_mut();
    out.backward();
    println!("x1 = {}", x1.borrow());
    println!("w1 = {}", w1.borrow());
    println!("x2 = {}", x2.borrow());
    println!("w2 = {}", w2.borrow());
    println!("output = {out}");
}
