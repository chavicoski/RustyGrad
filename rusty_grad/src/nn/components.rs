use crate::backend::tensor::Tensor;
use std::{cell::RefCell, rc::Rc};

pub trait Module {
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.borrow_mut().grad.fill(0.0);
        }
    }
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>>;
    fn forward(&self, x: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>>;
}
