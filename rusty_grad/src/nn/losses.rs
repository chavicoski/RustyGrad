use std::{cell::RefCell, rc::Rc};

use crate::backend::{
    ops::{diff, pow},
    tensor::Tensor,
};

pub fn squared_error(
    y_true: Rc<RefCell<Tensor>>,
    y_pred: Rc<RefCell<Tensor>>,
) -> Rc<RefCell<Tensor>> {
    pow(diff(y_true, y_pred), 2.0)
}
