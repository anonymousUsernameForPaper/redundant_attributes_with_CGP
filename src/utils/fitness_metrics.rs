use std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;
use itertools::Itertools;

pub fn fitness_regression(prediction: &Vec<usize>, labels: &Vec<usize>) -> f32 {
    let mut fitness: usize = 0;
    // prediction.iter().zip(labels.iter()).for_each(|(inner_pred, inner_label)|
    //     inner_pred.iter().zip(inner_label.iter()).for_each(|(x, y)| fitness += (x - y).abs())
    // );
    prediction.iter().zip(labels.iter()).for_each(|(x, y)|
        fitness += x.abs_diff(*y));

    let mut fitness: f32 = fitness as f32 / (prediction.len() as f32);

    if fitness.is_nan() {
        fitness = f32::MAX;
    } else if fitness.is_infinite() {
        fitness = f32::MAX;
    }

    return fitness;
}

/// Balanced Accuracy
pub fn fitness_categorical_multiclass(prediction: &Vec<usize>, labels: &Vec<usize>) -> f32 {
    let unique_labels: Vec<usize> = (0..=*labels.iter().max().unwrap()).collect();
    let mut true_positives: HashMap<usize, f32, BuildNoHashHasher<usize>> = HashMap::default();
    // init hashmap with zeros
    for label in &unique_labels {
        true_positives.insert(*label, 0.);
    }

    // get all true positives
    prediction.iter().zip(labels.iter()).for_each(|(x, y)| {
        if x == y {
            *true_positives.get_mut(y).unwrap() += 1.;
        }
    });

    // get the total amount of occurences of each class in the label
    let mut occurences_label: HashMap<usize, f32, BuildNoHashHasher<usize>> = HashMap::default();
    // init hashmap with zeros
    for label in &unique_labels {
        occurences_label.insert(*label, 0.);
    }
    // count occurences
    for elem in labels {
        *occurences_label.get_mut(elem).unwrap() += 1.;
    }

    let true_unique_labels: Vec<usize> = labels
        .clone()
        .into_iter()
        .unique()
        .collect();
    let number_classes = true_unique_labels.len() as f32;

    let mut balanced_accuracy = 0.;

    for class in true_unique_labels {
        balanced_accuracy += true_positives[&class] / occurences_label[&class];
    }

    let balanced_accuracy = balanced_accuracy / number_classes;


    return 1. - balanced_accuracy;
}

/// Matthews correlation coefficient (MCC) / Phi coefficient
pub fn fitness_categorical_binary(prediction: &Vec<usize>, labels: &Vec<usize>) -> f32 {
    let mut true_positive: f32 = 0.;
    let mut false_postive: f32 = 0.;
    let mut false_negative: f32 = 0.;
    let mut true_negative: f32 = 0.;

    // get confusion matrix
    prediction.iter().zip(labels.iter()).for_each(|(x, y)| {
        if *x == 1 && *y == 1 {
            true_positive += 1.;
        } else if *x == 1 {
            false_postive += 1.;
        } else if *x == 0 && *y == 0 {
            true_negative += 1.;
        } else {
            false_negative += 1.;
        }
    });

    let n = true_negative + true_positive + false_negative + false_postive;
    let s = (true_positive + false_negative) / n;
    let p = (true_positive + false_postive) / n;

    let denominator = (p * s) * (1.- s) * (1. - p);
    if denominator <= 1e-5 {
        return 1.;
    }

    let numerator = (true_positive / n) - (s * p);
    let denominator = denominator.sqrt();

    let mcc = numerator / denominator;

    return 1. - mcc.abs();
}