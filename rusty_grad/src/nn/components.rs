use crate::backend::tensor::RTensor;

pub trait Module {
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.borrow_mut().grad.fill(0.0);
        }
    }
    fn parameters(&self) -> Vec<RTensor>;
    fn forward(&self, x: &RTensor) -> RTensor;
}
