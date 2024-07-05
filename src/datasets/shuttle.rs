use std::fs;
use std::path::Path;
use crate::datasets::dataset_utils::standardize_dataset;
use crate::datasets::fitness_metric_type::FitnessFuncType;

pub fn get_dataset(dataset_path: String) -> (Vec<Vec<f32>>,
                         Vec<usize>,
                         Vec<Vec<f32>>,
                         Vec<usize>,
                         FitnessFuncType) {
    let contents_data_train = fs::read_to_string(Path::new(&dataset_path).join("attributes_train.csv"))
        .expect("Should have been able to read the file");
    let contents_label_train = fs::read_to_string(Path::new(&dataset_path).join("label_train.csv"))
        .expect("Should have been able to read the file");
    let contents_data_test = fs::read_to_string(Path::new(&dataset_path).join("attributes_test.csv"))
        .expect("Should have been able to read the file");
    let contents_label_test = fs::read_to_string(Path::new(&dataset_path).join("label_test.csv"))
        .expect("Should have been able to read the file");

    let contents_data_train = contents_data_train.lines();
    let contents_label_train = contents_label_train.lines();
    let contents_data_test = contents_data_test.lines();
    let contents_label_test = contents_label_test.lines();
    let mut train_data: Vec<Vec<f32>> = vec![];
    let mut train_label: Vec<usize> = vec![];

    for line in contents_data_train {
        let line: Vec<&str> = line.split(",").collect();
        let converted_data: Vec<f32> = line.iter().map(|val| val.parse::<f32>().unwrap()).collect();
        train_data.push(converted_data);
    }
    for line in contents_data_test {
        let line: Vec<&str> = line.split(",").collect();
        let converted_data: Vec<f32> = line.iter().map(|val| val.parse::<f32>().unwrap()).collect();
        train_data.push(converted_data);
    }
    for line in contents_label_train {
        let lbl = line.parse::<f32>().unwrap() as usize;
        train_label.push(lbl);
    }
    for line in contents_label_test {
        let lbl = line.parse::<f32>().unwrap() as usize;
        train_label.push(lbl);
    }

    let train_data = standardize_dataset(train_data);


    return (train_data, train_label, vec![], vec![], FitnessFuncType::ClassificationMultiClass);

}
