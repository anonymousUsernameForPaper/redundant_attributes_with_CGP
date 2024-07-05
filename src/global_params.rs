use std::fmt::{Display, Formatter};
use crate::datasets::fitness_metric_type::FitnessFuncType;

#[derive(Clone)]
pub struct CgpParameters {
    pub graph_width: usize,
    pub mu: usize,
    pub lambda: usize,
    pub eval_after_iterations: usize,
    pub nbr_inputs: usize,
    pub nbr_outputs: usize,
    pub fitness_func_type: FitnessFuncType,
}


impl Display for CgpParameters {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "############ Parameters ############\n")?;
        write!(f, "graph_width: {}\n", self.graph_width)?;
        write!(f, "mu: {}\n", self.mu)?;
        write!(f, "lambda: {}\n", self.lambda)?;
        write!(f, "eval_after_iterations: {}\n", self.eval_after_iterations)?;
        write!(f, "nbr_inputs: {}\n", self.nbr_inputs)?;
        write!(f, "nbr_outputs: {}\n", self.nbr_outputs)?;
        write!(f, "#########################\n")
    }
}