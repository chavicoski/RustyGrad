use crate::backend::{
    ops::{diff, pow},
    tensor::RTensor,
};

pub fn squared_error(y_true: &RTensor, y_pred: &RTensor) -> RTensor {
    pow(&diff(y_true, y_pred), 2.0)
}
