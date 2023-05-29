use ndarray::{ArrayD, IxDyn};
use rusty_grad::backend::ops::add;
use rusty_grad::backend::tensor::{RTensor, Tensor};
use rusty_grad::nn::{components::Module, losses::squared_error, models::MLP};
use rusty_grad::rtensor;

const LEARNING_RATE: f32 = 0.001;
const EPOCHS: usize = 500;

fn main() {
    // Prepare the dataset
    let dataset_x = vec![
        rtensor![&[1, 3], &[2., 3., -1.]],
        rtensor![&[1, 3], &[3., -1., 0.5]],
        rtensor![&[1, 3], &[0.5, 1., 1.]],
        rtensor![&[1, 3], &[1., 1., -1.]],
    ];
    let dataset_y = vec![
        rtensor![&[1, 1], &[1.]],
        rtensor![&[1, 1], &[-1.]],
        rtensor![&[1, 1], &[-1.]],
        rtensor![&[1, 1], &[-1.]],
    ];

    // Create the model
    let model = MLP::new(3, vec![4, 4, 1]);

    for epoch in 0..EPOCHS {
        // Forward pass
        let pred: Vec<RTensor> = dataset_x.iter().map(|x| model.forward(x)).collect();

        // Compute loss
        let loss: RTensor = dataset_y
            .iter()
            .zip(&pred)
            .map(|(y_true, y_pred)| squared_error(y_true, y_pred))
            .reduce(|v1, v2| add(&v1, &v2))
            .unwrap();

        // Reset the gradients to zero
        model.zero_grad();
        // Backpropagate the loss
        loss.borrow_mut().backward();

        // Update parameters
        for param in model.parameters() {
            let param_delta = &param.borrow().grad * LEARNING_RATE;
            param.borrow_mut().data -= &param_delta;
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
