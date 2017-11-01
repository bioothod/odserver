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
use std::io;
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

    fn process_image(&mut self, threshold: f32, img: &image::DynamicImage) -> Result<Vec<Class>, Box<Error>> {
        match img.as_rgb8() {
            None => {
                Ok(vec!())
            },
            Some(rgb) => {
                let mut t = Tensor::<u8>::new(&[1, rgb.height() as u64, rgb.width() as u64, 3]);
                for (x, y, &p) in rgb.enumerate_pixels() {
                    let idx = (y*rgb.width() + x)*3;
                    let idx: usize = idx as usize;

                    t[idx + 0] = p.data[0];
                    t[idx + 1] = p.data[1];
                    t[idx + 2] = p.data[2];
                }

                let res = self.step(&t)?;
                let m: Vec<Class> = res.into_iter().filter(|c| c.score >= threshold).collect();
                Ok(m)
            }
        }
    }
}

fn parse_file(gr: &mut Graph, threshold: f32, entry: io::Result<fs::DirEntry>) -> Result<(), Box<Error>> {
    let entry = entry?;
    let file_type = entry.file_type()?;
    if !file_type.is_file() {
        return Ok(());
    }

    let img = image::open(entry.path())?;
    let m = gr.process_image(threshold, &img)?;
    let is_selfie = m.iter().filter(|c| c.class == 6).count() > 0;
    println!("path: {}, dimensions: {:?}, matches: {:?}, is_selfie: {}",
             entry.path().display(), img.dimensions(), m, is_selfie);

    Ok(())
}

fn parse_dir(gr: &mut Graph, threshold: f32, image_dir: &str) -> Result<(), Box<Error>> {
    for entry in fs::read_dir(image_dir)? {
        let _ = parse_file(gr, threshold, entry);
    }

    Ok(())
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
    let threshold = matches.value_of("threshold").unwrap_or("0.8").parse::<f32>().unwrap();

    let mut gr = Graph::new(model_filename).unwrap();
    if let Some(image_dir) = matches.value_of("image_dir") {
        parse_dir(&mut gr, threshold, image_dir).unwrap();
    }
}
