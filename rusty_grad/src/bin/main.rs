use rusty_grad::backend::{ops::add, value::Value};
use rusty_grad::nn::{components::Module, losses::squared_error, models::MLP};
use std::{cell::RefCell, rc::Rc};

const LEARNING_RATE: f32 = 0.005;
const EPOCHS: usize = 1000;

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

    for epoch in 0..EPOCHS {
        // Forward pass
        let pred = dataset_x
            .iter()
            .map(|x| model.forward(&x))
            .collect::<Vec<Vec<Rc<RefCell<Value>>>>>();

        // Compute loss
        let loss = dataset_y
            .iter()
            .zip(&pred)
            .map(|(y_true, y_pred)| squared_error(&y_true, &y_pred[0]))
            .reduce(|ref v1, ref v2| add(v1, v2))
            .unwrap();

        // Reset the gradients to zero
        model.zero_grad();
        // Backpropagate the loss
        loss.borrow_mut().backward();

        // Update parameters
        for param in model.parameters() {
            let mut param = param.borrow_mut();
            param.data -= param.grad * LEARNING_RATE;
        }

        // Show current loss
        println!(
            "Epoch: {}/{} - Loss {}",
            epoch,
            EPOCHS - 1,
            loss.borrow_mut().data
        );
    }
}
