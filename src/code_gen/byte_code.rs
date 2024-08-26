use std::collections::HashMap;

use pest::{iterators::Pair, pratt_parser::PrattParser};

use crate::lang::{Rule, Types, Values};

use super::CodeGen;

#[derive(Debug, Clone)]
pub enum ByteCode<'a> {
    MainBlock(Vec<ByteCode<'a>>),
    CodeBlock(Vec<ByteCode<'a>>),
    DeclareVar(String, Pair<'a, Rule>),
    AssignVar(String, Pair<'a, Rule>),
    ChangeAssignVar(String, Pair<'a, Rule>, Pair<'a, Rule>),
    CmdIf(Pair<'a, Rule>, Box<ByteCode<'a>>, Option<Box<ByteCode<'a>>>),
    CmdFor(Pair<'a, Rule>, Box<ByteCode<'a>>, Box<ByteCode<'a>>),
    CmdWhile(Pair<'a, Rule>, Box<ByteCode<'a>>),
    CmdWrite(Pair<'a, Rule>),
    CmdRead(String),
}

impl<'a> CodeGen<'a> for ByteCode<'a> {
    fn main_block(acc: Vec<Self>) -> Self {
        Self::MainBlock(acc)
    }

    fn code_block(acc: Vec<Self>) -> Self {
        Self::CodeBlock(acc)
    }

    fn declare_var(
        ident: &'a str,
        _: Types,
        expr: Pair<'a, Rule>,
        _: &PrattParser<Rule>,
        _: u32,
    ) -> Self {
        Self::DeclareVar(ident.to_string(), expr)
    }

    fn assign_var(
        ident: &'a str,
        _: Types,
        expr: Pair<'a, Rule>,
        _: &PrattParser<Rule>,
        _: u32,
    ) -> Self {
        Self::AssignVar(ident.to_string(), expr)
    }

    fn change_assign_var(
        ident: &'a str,
        _: Types,
        op: Pair<'a, Rule>,
        expr: Pair<'a, Rule>,
        _: &PrattParser<Rule>,
        _: u32,
    ) -> Self {
        Self::ChangeAssignVar(ident.to_string(), op, expr)
    }

    fn cmd_if(
        expr: Pair<'a, Rule>,
        true_branch: Self,
        false_branch: Option<Self>,
        _: &PrattParser<Rule>,
        _: u32,
    ) -> Self {
        Self::CmdIf(
            expr,
            Box::new(true_branch),
            false_branch.map(|false_branch| Box::new(false_branch)),
        )
    }

    fn cmd_for(
        expr: Pair<'a, Rule>,
        change_assign: Self,
        block: Self,
        _: &PrattParser<Rule>,
        _: u32,
    ) -> Self {
        Self::CmdFor(expr, Box::new(change_assign), Box::new(block))
    }

    fn cmd_while(expr: Pair<'a, Rule>, block: Self, _: &PrattParser<Rule>, _: u32) -> Self {
        Self::CmdWhile(expr, Box::new(block))
    }

    fn cmd_write(content: Pair<'a, Rule>, _: Types, _: u32) -> Self {
        Self::CmdWrite(content)
    }

    fn cmd_read(ident: &'a str, _: Types, _: u32) -> Self {
        Self::CmdRead(ident.to_owned())
    }
}

impl<'a> ByteCode<'a> {
    pub fn eval(self, pratt: &PrattParser<Rule>, mem: &mut HashMap<String, Values>) {
        fn apply_op<'a>(lhs: Values, op: Pair<'a, Rule>, rhs: Values) -> Values {
            match (lhs, rhs) {
                (Values::Int(lhs), Values::Int(rhs)) => match op.as_rule() {
                    Rule::sum => Values::Int(lhs + rhs),
                    Rule::sub => Values::Int(lhs - rhs),
                    Rule::mul => Values::Int(lhs * rhs),
                    Rule::div => Values::Int(lhs / rhs),
                    Rule::le => Values::Bool(lhs <= rhs),
                    Rule::lt => Values::Bool(lhs < rhs),
                    Rule::ge => Values::Bool(lhs >= rhs),
                    Rule::gt => Values::Bool(lhs > rhs),
                    Rule::ne => Values::Bool(lhs != rhs),
                    Rule::eq => Values::Bool(lhs == rhs),
                    Rule::or => unreachable!(),
                    Rule::and => unreachable!(),
                    _ => unreachable!(),
                },
                (Values::Float(lhs), Values::Float(rhs)) => match op.as_rule() {
                    Rule::sum => Values::Float(lhs + rhs),
                    Rule::sub => Values::Float(lhs - rhs),
                    Rule::mul => Values::Float(lhs * rhs),
                    Rule::div => Values::Float(lhs / rhs),
                    Rule::le => Values::Bool(lhs <= rhs),
                    Rule::lt => Values::Bool(lhs < rhs),
                    Rule::ge => Values::Bool(lhs >= rhs),
                    Rule::gt => Values::Bool(lhs > rhs),
                    Rule::ne => Values::Bool(lhs != rhs),
                    Rule::eq => Values::Bool(lhs == rhs),
                    Rule::or => unreachable!(),
                    Rule::and => unreachable!(),
                    _ => unreachable!(),
                },
                (Values::String(lhs), Values::String(rhs)) => match op.as_rule() {
                    Rule::sum => Values::String(format!("{lhs}{rhs}")),
                    Rule::sub => unreachable!(),
                    Rule::mul => unreachable!(),
                    Rule::div => unreachable!(),
                    Rule::le => Values::Bool(lhs <= rhs),
                    Rule::lt => Values::Bool(lhs < rhs),
                    Rule::ge => Values::Bool(lhs >= rhs),
                    Rule::gt => Values::Bool(lhs > rhs),
                    Rule::ne => Values::Bool(lhs != rhs),
                    Rule::eq => Values::Bool(lhs == rhs),
                    Rule::or => unreachable!(),
                    Rule::and => unreachable!(),
                    _ => unreachable!(),
                },
                (Values::Bool(lhs), Values::Bool(rhs)) => match op.as_rule() {
                    Rule::sum => unreachable!(),
                    Rule::sub => unreachable!(),
                    Rule::mul => unreachable!(),
                    Rule::div => unreachable!(),
                    Rule::le => unreachable!(),
                    Rule::lt => unreachable!(),
                    Rule::ge => unreachable!(),
                    Rule::gt => unreachable!(),
                    Rule::ne => Values::Bool(lhs != rhs),
                    Rule::eq => Values::Bool(lhs == rhs),
                    Rule::or => Values::Bool(lhs || rhs),
                    Rule::and => Values::Bool(lhs && rhs),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }

        fn eval_expr<'a>(
            expr: Pair<'a, Rule>,
            pratt: &PrattParser<Rule>,
            mem: &mut HashMap<String, Values>,
        ) -> Values {
            let pairs = expr.into_inner();
            pratt
                .map_primary(|atom| match atom.as_rule() {
                    Rule::ident => mem.get(atom.as_str()).unwrap().clone(),
                    Rule::expr => eval_expr(atom, pratt, mem),
                    Rule::int_val => Values::Int(atom.as_str().parse().unwrap()),
                    Rule::float_val => Values::Float(atom.as_str().parse().unwrap()),
                    Rule::r#true => Values::Bool(atom.as_str().parse().unwrap()),
                    Rule::r#false => Values::Bool(atom.as_str().parse().unwrap()),
                    Rule::string_raw => Values::String(atom.as_str().to_owned()),
                    _ => unreachable!(),
                })
                .map_prefix(|op, val| match val {
                    Values::Int(val) => match op.as_rule() {
                        Rule::negative => Values::Int(-val),
                        Rule::negation => unreachable!(),
                        _ => unreachable!(),
                    },
                    Values::Float(val) => match op.as_rule() {
                        Rule::negative => Values::Float(-val),
                        Rule::negation => unreachable!(),
                        _ => unreachable!(),
                    },
                    Values::String(_) => unreachable!(),
                    Values::Bool(val) => match op.as_rule() {
                        Rule::negative => unreachable!(),
                        Rule::negation => Values::Bool(!val),
                        _ => unreachable!(),
                    },
                })
                .map_infix(|lhs, op, rhs| apply_op(lhs, op, rhs))
                .parse(pairs)
        }

        match self {
            ByteCode::DeclareVar(ident, expr) | ByteCode::AssignVar(ident, expr) => {
                let v = eval_expr(expr, pratt, mem);
                mem.insert(ident, v);
            }
            ByteCode::CodeBlock(bcs) | ByteCode::MainBlock(bcs) => {
                for bc in bcs {
                    bc.eval(pratt, mem);
                }
            }
            ByteCode::ChangeAssignVar(ident, op, expr) => {
                let lhs = mem.get(&ident).unwrap().clone();
                let rhs = eval_expr(expr, pratt, mem);
                let v = apply_op(lhs, op, rhs);
                mem.insert(ident, v);
            }
            ByteCode::CmdIf(expr, true_branch, false_branch) => {
                let cond = eval_expr(expr, pratt, mem);
                let Values::Bool(cond) = cond else {
                    unreachable!();
                };
                if cond {
                    true_branch.eval(pratt, mem);
                } else {
                    match false_branch {
                        Some(false_branch) => false_branch.eval(pratt, mem),
                        None => {}
                    }
                }
            }
            ByteCode::CmdFor(expr, change_assing, block) => loop {
                let cond = eval_expr(expr.clone(), pratt, mem);
                let Values::Bool(cond) = cond else {
                    unreachable!();
                };

                if !cond {
                    break;
                }

                block.clone().eval(pratt, mem);
                change_assing.clone().eval(pratt, mem);
            },
            ByteCode::CmdWhile(expr, block) => loop {
                let cond = eval_expr(expr.clone(), pratt, mem);
                let Values::Bool(cond) = cond else {
                    unreachable!();
                };

                if !cond {
                    break;
                }

                block.clone().eval(pratt, mem);
            },
            ByteCode::CmdWrite(content) => match content.as_rule() {
                Rule::string_raw => {
                    let content = content.as_str();
                    println!("{content}");
                }
                Rule::int_val => {
                    let content: i64 = content.as_str().parse().unwrap();
                    println!("{content}");
                }
                Rule::float_val => {
                    let content: f64 = content.as_str().parse().unwrap();
                    println!("{content}");
                }
                Rule::r#true | Rule::r#false => {
                    let content: bool = content.as_str().parse().unwrap();
                    println!("{content}");
                }
                Rule::ident => {
                    let content = mem.get(content.as_str()).unwrap().clone();
                    match content {
                        Values::Int(content) => println!("{content}"),
                        Values::Float(content) => println!("{content}"),
                        Values::String(content) => println!("{content}"),
                        Values::Bool(content) => println!("{content}"),
                    }
                }
                _ => unreachable!(),
            },
            ByteCode::CmdRead(ident) => {
                let mut buffer = String::new();
                std::io::stdin().read_line(&mut buffer).unwrap();
                let buffer = &buffer[..buffer.len() - 1];
                let val = mem.get_mut(&ident).unwrap();
                match val {
                    Values::Int(val) => *val = buffer.parse().unwrap(),
                    Values::Float(val) => *val = buffer.parse().unwrap(),
                    Values::String(val) => *val = buffer.to_owned(),
                    Values::Bool(val) => *val = buffer.parse().unwrap(),
                }
            }
        }
    }
}
