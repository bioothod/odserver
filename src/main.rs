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
    fn new(c: f32, s: f32) -> Class {
        Class {
            class: c as i32,
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

        let scores = step.take_output::<f32>(scores)?;
        let classes = step.take_output::<f32>(classes)?;

        return Ok(classes.iter().zip(scores.iter()).map(|(&c, &s)| Class::new(c, s)).collect());
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
    let threshold = matches.value_of("threshold").unwrap_or("0.8").parse::<f32>().unwrap();

    let mut gr = Graph::new(model_filename).unwrap();

    if let Ok(entries) = fs::read_dir(image_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        let img = image::open(entry.path());
                        if let Ok(img) = img {
                            if let Some(rgb) = img.as_rgb8() {
                                println!("path: {}, color_type: {:?}, dimensions: {:?}",
                                         entry.path().display(), img.color(), img.dimensions());

                                let mut t = Tensor::<u8>::new(&[1, rgb.height() as u64, rgb.width() as u64, 3]);
                                for (x, y, &p) in rgb.enumerate_pixels() {
                                    let idx = (y*rgb.width() + x)*3;
                                    let idx: usize = idx as usize;

                                    t[idx + 0] = p.data[0];
                                    t[idx + 1] = p.data[1];
                                    t[idx + 2] = p.data[2];
                                }

                                match gr.step(&t) {
                                    Err(err) => println!("step failed: {}", err),
                                    Ok(res) => {
                                        let m: Vec<&Class> = res.iter().filter(|c| c.score >= threshold).collect();
                                        let is_selfie = m.iter().filter(|c| c.class == 6).count() > 0;
                                        println!("result: {:?}, selfie: {}", m, is_selfie)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
