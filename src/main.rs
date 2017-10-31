#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;
extern crate image;
extern crate tensorflow;

use image::GenericImage;
use std::env;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use std::process;
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
    let filename = "export_model/frozen_inference_graph.pb";
    //let filename = "examples/addition-model/model.pb";
    
    let mut gr = Graph::new(filename).unwrap_or_else(|err| {
        eprintln!("could not create new graph from {}: {}", filename, err);
        process::exit(1);
    });

    let mut args = env::args();
    args.next();
    for arg in args {
        if let Ok(entries) = fs::read_dir(arg) {
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
}
