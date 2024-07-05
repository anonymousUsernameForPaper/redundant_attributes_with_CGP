use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::PathBuf;
use std::io::prelude::*;
use flate2::Compression;
use flate2::write;
use std::error::Error;
use std::ffi::OsStr;

// #[cfg(feature = "vanilla")]
// use crate::vanilla_cgp::chromosome::Chromosome;
// #[cfg(feature = "dag")]
use crate::dag::chromosome_dag::Chromosome;


fn write_compr(path: &PathBuf) -> Box<dyn Write> {
    let file = match File::create(path) {
        Err(why) => panic!("couldn't open {}: {}", path.display(), why.description()),
        Ok(file) => file,
    };

    if path.extension() == Some(OsStr::new("gz")) {
        Box::new(BufWriter::with_capacity(
            128 * 1024,
            write::GzEncoder::new(file, Compression::default()),
        ))
    } else {
        Box::new(BufWriter::with_capacity(128 * 1024, file))
    }
}


pub struct LoggingStuff {
    // pub buffered_writer: BufWriter<File>,
    pub buffered_compressed_writer: Box<dyn Write>,

}


impl LoggingStuff {
    pub fn new(path: PathBuf) -> Self {
        // let file = File::create(path).unwrap();
        // let buffered_writer = BufWriter::with_capacity(100, file);
        let writer = write_compr(&path);
        Self {
            buffered_compressed_writer: writer,
        }
    }

    pub fn execute(&mut self, it: usize, chromosome: Chromosome) {
        let mut write_string = String::new();

        write_string.push_str("Iteration: ");
        write_string.push_str(format!("{};", it).as_str());

        write_string.push_str("Active_Nodes: ");
        write_string.push_str(format!("{:?};", chromosome.active_nodes.unwrap()).as_str());

        write_string.push_str("Genes: ");
        for node in chromosome.nodes_grid {
            let con1: i64;
            let con2: i64;
            if node.connection0 == usize::MAX {
                con1 = -1;
            } else {
                con1 = node.connection0 as i64;
            }

            if node.connection1 == usize::MAX {
                con2 = -1;
            } else {
                con2 = node.connection1 as i64;
            }

            write_string.push_str(format!("({},{},{})-", node.function_id, con1, con2).as_str());
        }

        write_string.push_str("\n");
        // self.buffered_writer.write(write_string.as_ref());
        self.buffered_compressed_writer.write_all(write_string.as_ref()).unwrap();
    }

    pub fn close_writer(&mut self) {
        // self.buffered_writer.flush().unwrap();
        self.buffered_compressed_writer.flush().unwrap();
    }
}
