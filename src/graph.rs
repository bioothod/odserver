#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;
extern crate image;
extern crate tensorflow;

use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use self::tensorflow::Code;
use self::tensorflow::ImportGraphDefOptions;
use self::tensorflow::Session;
use self::tensorflow::SessionOptions;
use self::tensorflow::Status;
use self::tensorflow::StepWithGraph;
use self::tensorflow::Tensor;

pub struct Graph {
    graph: tensorflow::Graph,
    session: tensorflow::Session,
}

#[derive(Debug)]
pub struct Class {
    pub class: i32,
    pub score: f32,
}

impl Class {
    pub fn new(c: f32, s: f32) -> Class {
        Class {
            class: c as i32,
            score: s,
        }
    }
}

impl Graph {
    pub fn new(filename: &str) -> Result<Graph, Box<Error>> {
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

    pub fn step<T>(&mut self, img: &Tensor<T>) -> Result<(Vec<Class>), Box<Error>>
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

    pub fn process_image(&mut self, threshold: f32, img: &image::DynamicImage) -> Result<Vec<Class>, Box<Error>> {
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

