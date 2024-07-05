use std::fmt::{Display, Formatter};
use rand::seq::SliceRandom;
use crate::global_params::CgpParameters as g_params;
use crate::utils::utility_funcs;

// #[cfg(feature = "vanilla")]
use crate::vanilla_cgp::chromosome::Chromosome;
// #[cfg(feature = "dag")]
// use crate::dag::chromosome_dag::Chromosome;
// #[cfg(feature = "reorder")]
// use crate::reorder::chromosome_reorder_equidistant::Chromosome;


pub struct Runner {
    params: g_params,
    data: Vec<Vec<f32>>,
    label: Vec<usize>,
    eval_data: Option<Vec<Vec<f32>>>,
    eval_label: Option<Vec<usize>>,
    population: Vec<Chromosome>,
    best_fitness: f32,
    fitness_vals: Vec<f32>,
    parent_id: usize,
}

impl Display for Runner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parent: {}", self.population[self.parent_id])?;
        writeln!(f, "Fitness: {}", self.best_fitness)
    }
}

impl Runner {
    pub fn new(params: g_params,
               data: Vec<Vec<f32>>,
               label: Vec<usize>,
               mut eval_data: Option<Vec<Vec<f32>>>,
               eval_label: Option<Vec<usize>>) -> Self {
        let mut chromosomes: Vec<Chromosome> = Vec::with_capacity(params.mu + params.lambda);
        let mut fitness_vals: Vec<f32> = Vec::with_capacity(params.mu + params.lambda);

        // transpose so a whole row of the dataset can be used as an array for calculation
        let data = utility_funcs::transpose(data);
        if eval_data.is_some() {
            eval_data = Some(utility_funcs::transpose(eval_data.unwrap()));
        }

        for _ in 0..(params.mu + params.lambda) {
            let mut chromosome = Chromosome::new(params.clone());
            let fitness = chromosome.evaluate(&data, &label);
            fitness_vals.push(fitness);

            chromosomes.push(chromosome);
        }

        let best_fitness = utility_funcs::get_min(&fitness_vals);
        let parent_id = utility_funcs::get_argmin(&fitness_vals);

        Self {
            params,
            data,
            label,
            eval_data,
            eval_label,
            population: chromosomes,
            best_fitness,
            fitness_vals,
            parent_id,
        }
    }

    pub fn learn_step(&mut self) {
        self.mutate_chromosomes();

        self.eval_chromosomes();

        self.new_parent_by_neutral_search();
    }

    fn new_parent_by_neutral_search(&mut self) {
        let mut min_keys: Vec<usize> = Vec::with_capacity(self.params.mu + self.params.lambda);

        utility_funcs::get_argmins_of_value(&self.fitness_vals, &mut min_keys, self.best_fitness);

        if min_keys.len() == 1 {
            self.parent_id = min_keys[0];
        } else {
            if min_keys.contains(&self.parent_id) {
                let index = min_keys.iter().position(|x| *x == self.parent_id).unwrap();
                min_keys.remove(index);
            }
            self.parent_id = *min_keys.choose(&mut rand::thread_rng()).unwrap();
        }
    }

    fn mutate_chromosomes(&mut self) {
        // mutate new chromosomes; do not mutate parent
        for i in 0..(self.params.mu + self.params.lambda) {
            if i == self.parent_id {
                continue;
            }
            self.population[i] = self.population[self.parent_id].clone();

            self.population[i].mutate_single();

        }
    }

    fn eval_chromosomes(&mut self) {
        for i in 0..(self.params.mu + self.params.lambda) {
            if i != self.parent_id {
                let fitness: f32 = self.population[i].evaluate(&self.data, &self.label);

                self.fitness_vals[i] = fitness;
            }
        }

        let best_fitness = utility_funcs::get_min(&self.fitness_vals);

        self.best_fitness = best_fitness;
    }

    pub fn get_test_fitness(&mut self) -> f32 {
        let mut best_fitness = f32::MAX;

        if self.eval_data.is_none() {
            return best_fitness
        }

        for individual in &mut self.population {
            let fitness = individual.evaluate(&self.eval_data.as_ref().unwrap(), &self.eval_label.as_ref().unwrap());

            if fitness < best_fitness {
                best_fitness = fitness;
            }
        }
        return best_fitness;

    }

    pub fn get_best_fitness(&self) -> f32 {
        return self.best_fitness;
    }

    pub fn get_parent(&self) -> Chromosome {
        return self.population[self.parent_id].clone();
    }
}

