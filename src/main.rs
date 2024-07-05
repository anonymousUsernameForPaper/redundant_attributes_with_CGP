use std::io::{BufWriter, Write};
use cgp::global_params::CgpParameters;
use cgp::datasets::*;
use clap::Parser;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::exit;
use itertools::Itertools;
use cgp::datasets::crossvalidation::CrossValidation;
use cgp::datasets::dataset_utils::{add_redundancies, RedundancyType};
use cgp::datasets::fitness_metric_type::FitnessFuncType;
use cgp::utils::runner::Runner;


#[derive(Parser, Clone)]
#[clap(author, version, about, name = "testname")]
struct Args {
    #[arg(long, default_value_t = 0)]
    run_id: usize,

    #[arg(long, default_value_t = 3)]
    dataset: usize,

    #[arg(long, default_value_t = 500)]
    nbr_nodes: usize,

    #[arg(long, default_value_t = 0.2)]
    data_bloat: f32,

    #[arg(long, default_value_t = 2)]
    redundancy_type: usize,

    #[arg(long)]
    dataset_path: String,
}

fn main() {
    // ################################################################################
    // ############################ Arguments #########################################
    // ################################################################################
    let mut args = Args::parse();
    let (
        data,
        label,
        _,
        _,
        fitness_type
    ) = match args.dataset {
        0 => abalone::get_dataset(args.dataset_path),  // ~3h
        1 => credit::get_dataset(args.dataset_path),  // ~10 min?
        2 => shuttle::get_dataset(args.dataset_path),  // ~6h
        3 => breast_cancer::get_dataset(args.dataset_path),  // ~10 min
        4 => page_blocks::get_dataset(args.dataset_path),  // ~1h
        5 => waveform::get_dataset(args.dataset_path),  // ~10min
        _ => panic!("Wrong dataset"),
    };

    let redundancy_type = match args.redundancy_type {
        0 => RedundancyType::Copy,
        1 => RedundancyType::Noise,
        2 => RedundancyType::NoisyCopy,
        _ => { panic!("Wrong r-type") }
    };

    let (data, _, indice_insert_copy_mapping) = add_redundancies(data,
                                                                 None,
                                                                 args.data_bloat,
                                                                 redundancy_type);

    let nbr_outputs: usize = match fitness_type {
        FitnessFuncType::Regression => { 1 }
        FitnessFuncType::ClassificationBinary => { 1 }
        FitnessFuncType::ClassificationMultiClass => { *label.iter().max().unwrap() }
    };

    let params = CgpParameters {
        graph_width: args.nbr_nodes,
        mu: 1,
        lambda: 4,
        eval_after_iterations: 500,
        nbr_inputs: data[0].len(),
        nbr_outputs,
        fitness_func_type: fitness_type,
    };

    // let stdout = std::io::stdout();
    // let mut lock = stdout.lock();

    let mut cross_validate = CrossValidation::new(data.len(), 5);

    for fold in 0..5 {
        let run_id = args.run_id + fold;

        let (train_data, train_label, test_data, test_label) = cross_validate.split(data.clone(), label.clone());
        let test_data = Some(test_data);
        let test_label = Some(test_label);

        // ################################################################################
        // ############################ Logger ####### ####################################
        // ################################################################################
        let dataset_string = match args.dataset {
            0 => "abalone",
            1 => "credit",
            2 => "shuttle",
            3 => "breast_cancer",
            4 => "page_blocks",
            5 => "waveform",
            _ => panic!("wrong dataset number in string"),
        };

        let redundancy_string = match args.redundancy_type {
            0 => "redundancy_type_0",
            1 => "redundancy_type_1",
            2 => "redundancy_type_2",
            _ => panic!("wrong redundancy type"),
        };
        let databloat_string = match args.data_bloat {
            0.0 => "baseline",
            0.2 => "databloat_20",
            0.4 => "databloat_40",
            0.6 => "databloat_60",
            0.8 => "databloat_80",
            1.0 => "databloat_100",
            _ => panic!("wrong datablaot type"),
        };

        let save_path:PathBuf;
        if args.data_bloat == 0.0 {
            save_path = Path::new("")
                .join("Experiments_Output")
                .join(dataset_string)
                .join(databloat_string)
                .join(format!("number_nodes_{}", args.nbr_nodes));
        } else {
            save_path = Path::new("")
                .join("Experiments_Output")
                .join(dataset_string)
                .join(redundancy_string)
                .join(databloat_string)
                .join(format!("number_nodes_{}", args.nbr_nodes));
        }

        fs::create_dir_all(save_path.clone()).unwrap();

        let save_file_iteration = format!("run_{}_iteration.txt", run_id);
        let mut output_file = BufWriter::new(File::create(save_path.join(save_file_iteration))
            .expect("cannot create file"));
        // ################################################################################
        // ############################ Training ##########################################
        // ################################################################################


        let mut runner = Runner::new(params.clone(),
                                     train_data,
                                     train_label,
                                     test_data,
                                     test_label);

        let mut runtime_iterations: usize = 0;

        for _ in 0..100_000 {
            writeln!(output_file, "Iteration: {runtime_iterations}, Fitness: {:?}", runner.get_best_fitness()).expect("write not okay??");
            // if runtime_iterations % params.eval_after_iterations == 0 {
            //     writeln!(lock, "Iteration: {runtime_iterations}, Fitness: {:?}", runner.get_best_fitness()).expect("write not okay??");
            // }
            runtime_iterations += 1;
            runner.learn_step();  // lern step

            if runner.get_best_fitness() <= 0.01 {  // for single parent
                // println!("eval fitness: {:?}", runner.get_test_fitness());
                break;
            }
        }

        let fitness_eval = runner.get_test_fitness();
        let fitness_train = runner.get_best_fitness();

        // ################################################################################
        // ############################ Saving to text ####################################
        // ################################################################################
        println!("{fitness_train}");

        let save_file_redundancy = format!("run_{}_redundancy.txt", run_id);
        let mut output_redundancy = File::create(save_path.join(save_file_redundancy))
            .expect("cannot create file");
        write!(output_redundancy, "{:?}", indice_insert_copy_mapping).expect("cannot write");

        println!("{runtime_iterations}");
        writeln!(output_file, "End at iteration: {}", runtime_iterations).expect("cannot write");
        writeln!(output_file, "Fitness Eval: {}", fitness_eval).expect("cannot write");
        writeln!(output_file, "Fitness Train: {}", fitness_train).expect("cannot write");

        output_file.flush().unwrap();

        let save_file_active_node = format!("run_{}_active_node.txt", run_id);
        let mut output = File::create(save_path.join(save_file_active_node))
            .expect("cannot create file");

        let mut parent = runner.get_parent();
        parent.get_active_nodes_id();

        write!(output, "{:?}", parent.active_nodes).expect("cannot write");
    }
}
