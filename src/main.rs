#![feature(path_add_extension)]

pub mod ast;
pub mod code_gen;
pub mod errors;
pub mod lang;

use std::{collections::HashMap, fs::File, io::Write, path::PathBuf};

use code_gen::{byte_code::ByteCode, cpp::Cpp};
use lang::Rule;
use pest::{
    pratt_parser::{Assoc, Op, PrattParser},
    Parser,
};

use crate::{ast::AST, code_gen::java::Java, lang::LangParser};

use clap::Parser as cParser;

#[derive(cParser, Debug)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    let pratt: PrattParser<Rule> = PrattParser::new()
        .op(Op::infix(Rule::or, Assoc::Left) | Op::infix(Rule::and, Assoc::Left))
        .op(Op::infix(Rule::le, Assoc::Left)
            | Op::infix(Rule::lt, Assoc::Left)
            | Op::infix(Rule::ge, Assoc::Left)
            | Op::infix(Rule::gt, Assoc::Left)
            | Op::infix(Rule::ne, Assoc::Left)
            | Op::infix(Rule::eq, Assoc::Left))
        .op(Op::infix(Rule::sum, Assoc::Left) | Op::infix(Rule::sub, Assoc::Left))
        .op(Op::infix(Rule::mul, Assoc::Left) | Op::infix(Rule::div, Assoc::Left))
        .op(Op::prefix(Rule::negation) | Op::prefix(Rule::negative));

    let unparsed_file = std::fs::read_to_string(args.input).expect("File not found");

    let pair = LangParser::parse(Rule::main, &unparsed_file);
    let mut pair = match pair {
        Ok(p) => p,
        Err(err) => {
            println!("Parsing error: {err}");
            return;
        }
    };

    let pair = pair.next().unwrap();

    let cpp = AST::from(pair.clone()).validate_generate::<Cpp>(&pratt);
    let java = AST::from(pair.clone()).validate_generate::<Java>(&pratt);
    let bc = AST::from(pair).validate_generate::<ByteCode>(&pratt);

    match cpp {
        Ok(cpp) => {
            println!("Writing C++ file...");
            let mut file = args.output.clone();
            file.add_extension("cpp");

            let mut file = match File::create(&file) {
                Ok(file) => file,
                Err(_) => {
                    println!("Error while creating file: {file:?}");
                    return;
                },
            };

            let content = cpp.into_inner();
            file.write_all(content.as_bytes()).unwrap();
        }
        Err(err) => {
            println!("Error while generating C++ code:\n{err}");
            return;
        },
    }

    match java {
        Ok(java) => {
            println!("Writing Java file...");
            let mut file = args.output.clone();
            file.add_extension("java");

            let mut file = match File::create(&file) {
                Ok(file) => file,
                Err(_) => {
                    println!("Error while creating file: {file:?}");
                    return;
                },
            };

            let content = java.into_inner();
            file.write_all(content.as_bytes()).unwrap();
        }
        Err(err) => {
            println!("Error while generating Java code:\n{err}");
            return;
        },
    }

    match bc {
        Ok(bc) => {
            println!("Starting Interpreter...");
            let mut mem = HashMap::new();
            bc.eval(&pratt, &mut mem);
        }
        Err(err) => {
            println!("Error while generating Byte Code:\n{err}");
            return;
        },
    }
}
