extern crate clap;
extern crate futures_util;
extern crate hyper;
extern crate image;
extern crate serde;
extern crate serde_json;

use crate::image::GenericImageView;

#[macro_use]
extern crate serde_derive;

use clap::{Arg, App};
use futures_util::TryStreamExt;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server};
use hyper::body::Bytes;

use std::error::Error;
use std::fs;
use std::io;
use std::sync::Arc;

mod graph;
use graph::Graph;

fn parse_file(gr: &Graph, threshold: f32, entry: io::Result<fs::DirEntry>) -> Result<(), Box<dyn Error>> {
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

fn parse_dir(gr: &Graph, threshold: f32, image_dir: &str) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(image_dir)? {
        let _ = parse_file(gr, threshold, entry);
    }

    Ok(())
}

fn process_image(gr: Arc<Graph>, threshold: f32, chunk: &Bytes) -> Result<String, Box<dyn Error>> {
    let img = image::load_from_memory(chunk)?;

    let m = (&gr).process_image(threshold, &img)?;
    let is_selfie = m.iter().filter(|c| c.class == 6).count() > 0;

    #[derive(Serialize, Deserialize)]
    struct ReplyDim {
        width: u32,
        height: u32,
    }
    #[derive(Serialize, Deserialize)]
    struct Reply {
        dimensions: ReplyDim,
        matches: Vec<graph::Class>,
        is_selfie: bool,
    }

    let r = Reply {
        dimensions: ReplyDim {
            width: img.width(),
            height: img.height(),
        },
        matches: m,
        is_selfie: is_selfie,
    };
    let res = serde_json::to_string(&r)?;
    Ok(res)
}

async fn process_request(gr: Arc<Graph>, threshold: f32, req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => Ok(Response::new(Body::from("Try POSTing data to /image\n"))),
        (&Method::POST, "/image") => {
            let chunk_stream = req.into_body().map_ok(move |chunk| {
                match process_image(gr.clone(), threshold, &chunk) {
                    Ok(data) => data,
                    Err(error) => {
                        #[derive(Serialize, Deserialize)]
                        struct ReplyErr {
                            error: String,
                        }
                        let re = ReplyErr { error: format!("{}", error) };
                        let r = serde_json::to_string(&re);
                        let s: String;
                        match r {
                            Err(_) => s = re.error,
                            Ok(x) => s = x,
                        };

                        s
                    }
                }
            });
            Ok(Response::new(Body::wrap_stream(chunk_stream)))
        },
        _ => Ok(Response::new(Body::from("Try POSTing data to /image\n"))),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

    let gr = Graph::new(model_filename).unwrap();
    if let Some(image_dir) = matches.value_of("image_dir") {
        parse_dir(&gr, threshold, image_dir).unwrap();
    }

    let graph_wrapper = Arc::new(gr);

    if let Some(addr) = matches.value_of("address") {
        let make_service = make_service_fn(|_| {
            let gr = graph_wrapper.clone();

            async move { Ok::<_, hyper::Error>(service_fn(move |req| process_request(gr.clone(), threshold, req))) }
        });

        let saddr = addr.parse()?;
        let server = Server::bind(&saddr).serve(make_service);

        println!("Listening on http://{}", addr);

        server.await?;
    }

    Ok(())
}
