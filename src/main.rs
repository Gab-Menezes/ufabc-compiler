pub mod ast;
pub mod code_gen;
pub mod errors;
pub mod lang;

use lang::Rule;
use pest::{
    pratt_parser::{Assoc, Op, PrattParser},
    Parser,
};

use crate::{ast::AST, code_gen::java::Java, lang::LangParser};

fn main() {
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

    let unparsed_file = std::fs::read_to_string("test1.isi").expect("cannot read file");

    let pair = LangParser::parse(Rule::main, &unparsed_file)
        .expect("unsuccessful parse")
        .next()
        .unwrap();

    let gen = AST::from(pair).validate_generate::<Java>(&pratt);

    match gen {
        Ok(gen) => {
            // let mut mem = HashMap::new();
            // gen.eval(&pratt, &mut mem)
            println!("{}", gen.into_inner());
        }
        Err(err) => println!("{err}"),
    }
}
