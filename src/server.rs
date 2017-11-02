extern crate hyper;
extern crate futures;

use self::futures::future::Future;
use self::futures::Stream;
use self::hyper::Chunk;
use self::hyper::{Method, StatusCode};
use self::hyper::server::{Http, Request, Response, Service};
use std::error::Error;
use std::sync::Arc;

pub struct Server<T>
{
    cb: T
}

struct ODServer<T>
{
    server: Arc<Server<T>>,
}

impl<T> Service for ODServer<T>
    where T: 'static + Fn(Chunk) -> Result<Chunk, Box<Error>>
{
    type Request = Request;
    type Response = Response;
    type Error = hyper::Error;

    type Future = Box<Future<Item=Self::Response, Error=Self::Error>>;

    fn call(&self, req: Request) -> Self::Future {
        match (req.method(), req.path()) {
            (&Method::Get, "/") => {
                Box::new(futures::future::ok(
                    Response::new().with_body("Try POSTing data to /image\n")
                ))
            },
            (&Method::Post, "/image") => {
                let c = self.server.clone();
                Box::new(req.body().concat2().map(move |chunk| {
                        match (c.cb)(chunk) {
                            Err(error) => Response::new().with_body(format!("there was an error: {}", error)).with_status(StatusCode::InternalServerError),
                            Ok(data) => Response::new().with_body(data),
                        }
                    })
                )
            },
            _ => {
                Box::new(futures::future::ok(
                    Response::new().with_status(StatusCode::NotFound)
                ))
            },
        }
    }
}

impl<T> Server<T>
    where T: 'static + Fn(Chunk) -> Result<Chunk, Box<Error>>
{
    pub fn new(cb: T) -> Server<T> {
        Server {
            cb: cb,
        }
    }

    pub fn start(self, saddr: &str) -> Result<(), Box<Error>> {
        let addr = saddr.parse()?;
        let arc = Arc::new(self);
        let server = Http::new().bind(&addr, move || Ok(ODServer{server: arc.clone()}))?;
        server.run()?;
        Ok(())
    }
}
