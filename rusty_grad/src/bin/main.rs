use rusty_grad::{add, squared_error, Module, Value, MLP};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    // Prepare the dataset
    let dataset_x = vec![
        vec![Value::new_rc(2.0), Value::new_rc(3.0), Value::new_rc(-1.0)],
        vec![Value::new_rc(3.0), Value::new_rc(-1.0), Value::new_rc(0.5)],
        vec![Value::new_rc(0.5), Value::new_rc(1.0), Value::new_rc(1.0)],
        vec![Value::new_rc(1.0), Value::new_rc(1.0), Value::new_rc(-1.0)],
    ];
    let dataset_y = vec![
        Value::new_rc(1.0),
        Value::new_rc(-1.0),
        Value::new_rc(-1.0),
        Value::new_rc(-1.0),
    ];

    // Create the model
    let model = MLP::new(3, vec![4, 4, 1]);

    // Make predictions (forward pass)
    let pred = dataset_x
        .iter()
        .map(|x| model.forward(&x))
        .collect::<Vec<Vec<Rc<RefCell<Value>>>>>();

    // Sum up the loss for each prediction to compute the total loss
    let loss = dataset_y
        .iter()
        .zip(&pred)
        .map(|(y_true, y_pred)| squared_error(&y_true, &y_pred[0]))
        .reduce(|ref v1, ref v2| add(v1, v2))
        .unwrap();

    println!("MLP predictions: [");
    for p in pred {
        print!("\t");
        for v in p {
            print!("{},", v.borrow());
        }
        print!("\n");
    }
    println!("]");

    // Backpropagate the loss
    loss.borrow_mut().backward();

    // Show the parameters to see the computed gradients
    println!("MLP parameters: [");
    for param in model.parameters() {
        println!("\t{},", param.borrow());
    }
    println!("]");
}
