extern crate clap;
extern crate hyper;
extern crate image;

use clap::{Arg, App};
use hyper::Chunk;
use image::GenericImage;
use std::error::Error;
use std::fs;
use std::io;

mod graph;
use graph::Graph;

mod server;

fn parse_file(gr: &Graph, threshold: f32, entry: io::Result<fs::DirEntry>) -> Result<(), Box<Error>> {
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

fn parse_dir(gr: &Graph, threshold: f32, image_dir: &str) -> Result<(), Box<Error>> {
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
        .arg(Arg::with_name("address")
                .short("a")
                .long("addr")
                .takes_value(true)
                .help("server listen address, format: addr:port"))
        .arg(Arg::with_name("threshold")
                .long("threshold")
                .takes_value(true)
                .help("image directory to scan and detect"))
        .get_matches();

    let model_filename = matches.value_of("model").unwrap();
    let threshold = matches.value_of("threshold").unwrap_or("0.8").parse::<f32>().unwrap();

    let mut gr = Graph::new(model_filename).unwrap();
    if let Some(image_dir) = matches.value_of("image_dir") {
        parse_dir(&gr, threshold, image_dir).unwrap();
    }

    if let Some(addr) = matches.value_of("address") {
        let srv = server::Server::new(move |chunk: Chunk| -> Result<Chunk, Box<Error>> {
            let data = chunk.to_vec();
            let img = image::load_from_memory(&data)?;
            println!("uploaded image, dimensions: {:?}", img.dimensions());

            let _m = (&gr).process_image(threshold, &img)?;

            Ok(Chunk::from("this is a response\n"))
        });
        srv.start(addr).unwrap();
    }
}
