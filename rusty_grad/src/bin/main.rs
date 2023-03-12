use ndarray::{ArrayD, IxDyn};
use rusty_grad::backend::ops::add;
use rusty_grad::backend::tensor::Tensor;
use rusty_grad::nn::{components::Module, losses::squared_error, models::MLP};
use std::{cell::RefCell, rc::Rc};

const LEARNING_RATE: f32 = 0.005;
const EPOCHS: usize = 1000;

fn main() {
    // Prepare the dataset
    let dataset_x = vec![
        vec![Tensor::new_rc(
            &ArrayD::from_shape_vec(IxDyn(&[3]), vec![2., 3., -1.]).unwrap(),
        )],
        vec![Tensor::new_rc(
            &ArrayD::from_shape_vec(IxDyn(&[3]), vec![3., -1., 0.5]).unwrap(),
        )],
        vec![Tensor::new_rc(
            &ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1., 1.]).unwrap(),
        )],
        vec![Tensor::new_rc(
            &ArrayD::from_shape_vec(IxDyn(&[3]), vec![1., 1., -1.]).unwrap(),
        )],
    ];
    let dataset_y = vec![
        Tensor::new_rc(&ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.]).unwrap()),
        Tensor::new_rc(&ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.]).unwrap()),
        Tensor::new_rc(&ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.]).unwrap()),
        Tensor::new_rc(&ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.]).unwrap()),
    ];

    // Create the model
    let model = MLP::new(3, vec![4, 4, 1]);

    for epoch in 0..EPOCHS {
        // Forward pass
        let pred = dataset_x
            .iter()
            .map(|x| model.forward(&x))
            .collect::<Vec<Vec<Rc<RefCell<Tensor>>>>>();

        // Compute loss
        let loss = dataset_y
            .iter()
            .zip(&pred)
            .map(|(y_true, y_pred)| squared_error(y_true.clone(), y_pred[0].clone()))
            .reduce(|v1, v2| add(v1, v2))
            .unwrap();

        // Reset the gradients to zero
        model.zero_grad();
        // Backpropagate the loss
        loss.borrow_mut().backward();

        // Update parameters
        for param in model.parameters() {
            param.borrow_mut().data -= &(&param.borrow().grad * LEARNING_RATE);
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
