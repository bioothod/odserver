#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;
extern crate clap;
extern crate image;
extern crate tensorflow;

use clap::{Arg, App};
use image::GenericImage;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use tensorflow::Code;
//use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

struct Graph {
    graph: tensorflow::Graph,
    session: tensorflow::Session,
}

#[derive(Debug)]
struct Class {
    class: i32,
    score: f32,
}

impl Class {
    fn new(c: i32, s: f32) -> Class {
        Class {
            class: c,
            score: s,
        }
    }
}

impl Graph {
    fn new(filename: &str) -> Result<Graph, Box<Error>> {
        if !Path::new(filename).exists() {
            return Err(Box::new(Status::new_set(Code::NotFound,
                                                &format!("Run 'python addition.py' to generate {} \
                                                          and try again.",
                                                         filename))
                .unwrap()));
        }

        let mut graph = tensorflow::Graph::new();
        
        let mut proto = Vec::new();
        File::open(filename)?.read_to_end(&mut proto)?;
        graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;

        let session = Session::new(&SessionOptions::new(), &graph)?;

        return Ok(Graph{graph, session});
    }

    fn step<T>(&mut self, img: &Tensor<T>) -> Result<(Vec<Class>), Box<Error>>
        where T: tensorflow::TensorType
    {
        let mut step = StepWithGraph::new();

        step.add_input(&self.graph.operation_by_name_required("image_tensor")?, 0, img);
        let scores = step.request_output(&self.graph.operation_by_name_required("detection_scores")?, 0);
        let classes = step.request_output(&self.graph.operation_by_name_required("detection_classes")?, 0);
        self.session.run(&mut step)?;

        let scores = step.take_output(scores)?;
        let classes = step.take_output(classes)?;

        return Ok(classes.iter().zip(scores.iter()).map(|(&s, &c)| Class::new(c, s)).collect());
    }
}

fn main() {
    let matches = App::new("Tensorflow object detection server")
        .arg(Arg::with_name("model")
                .short("m")
                .long("model")
                .required(true)
                .takes_value(true)
                .help("tensorflow graph file"))
        .arg(Arg::with_name("image_dir")
                .long("image_dir")
                .takes_value(true)
                .help("image directory to scan and detect"))
        .arg(Arg::with_name("threshold")
                .long("threshold")
                .takes_value(true)
                .help("image directory to scan and detect"))
        .get_matches();

    let model_filename = matches.value_of("model").unwrap();
    let image_dir = matches.value_of("image_dir").unwrap();
    let threshold = matches.value_of("threshold").unwrap_or("0.9").parse::<f32>();

    let mut gr = Graph::new(model_filename).unwrap();

    if let Ok(entries) = fs::read_dir(image_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        let img = image::open(entry.path());
                        if let Ok(img) = img {
                            let rgb = img.to_rgb();
                            let sz = rgb.height() * rgb.width() * 3;
                            let mut t = Tensor::new(&[sz as u64]);
                            t.copy_from_slice(&rgb.into_raw());

                            println!("path: {}, color_type: {:?}, dimensions: {:?}",
                                     entry.path().display(), img.color(), img.dimensions());

                            match gr.step(&t) {
                                Err(err) => println!("step failed: {}", err),
                                Ok(res) => println!("result: {:?}", res[0]),
                            }
                        }
                    }
                }
            }
        }
    }
}
