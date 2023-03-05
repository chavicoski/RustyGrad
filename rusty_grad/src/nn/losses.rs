use std::{cell::RefCell, rc::Rc};

use crate::backend::{
    ops::{diff, pow},
    value::Value,
};

pub fn mse(y_true: &Rc<RefCell<Value>>, y_pred: &Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    pow(&diff(y_true, y_pred), 2.0)
}
