#[macro_use]
extern crate lazy_static;

extern crate num;
extern crate seahash;

mod noun;
use noun::{ Noun, nock };

mod parse;
use parse::parse;

use std::env;

//Run this:
// cls; $env:RUST_BACKTRACE=1; cargo run NOCK_HERE
// [42 [8 [4 0 1] [4 0 3]]]]]               43
// [42 [8 [4 0 1] [0 1]]]                   [43 42]
// [42 [6 [1 1] [4 0 1] [1 233]]]           233

//Goldbach (large test program): https://urbit.org/~~/fora/posts/~2017.3.12..05.37.44..7d5e~/

fn main() {
    let mut args: Vec<_> = env::args().collect();
    if args.len() > 1 {
        args.remove(0);
        let noun = parse(args.join(" "));

        let start = PreciseTime::now();
        let result = nock(noun);
        let end = PreciseTime::now();
        let elapsed = start.to(end);
        println!("Elapsed {}ns {}ms {}s", elapsed.num_nanoseconds().unwrap(), elapsed.num_milliseconds(), elapsed.num_seconds());

        match result {
            None => println!("None"),
            Some(r) => println!("{}", r)
        }
    } 
}

extern crate time;
use time::PreciseTime;
fn benchmark() {
    
    // Tends to run in ~11500ns (release)
    let input = parse(String::from("[42 [8 [4 0 1] [0 1]]]"));

    let start = PreciseTime::now();
    let result = nock(input);
    let end = PreciseTime::now();

    println!("result = {}", result.unwrap());
    println!("{}ns", start.to(end).num_nanoseconds().unwrap());

}