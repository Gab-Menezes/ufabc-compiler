pub mod byte_code;
pub mod cpp;
pub mod java;

use pest::{iterators::Pair, pratt_parser::PrattParser};

use crate::lang::{Rule, Types};

pub trait CodeGen<'a>: Sized {
    fn main_block(acc: Vec<Self>) -> Self;
    fn code_block(acc: Vec<Self>) -> Self;
    fn declare_var(
        ident: &'a str,
        t: Types,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self;
    fn assign_var(
        ident: &'a str,
        t: Types,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self;
    fn change_assign_var(
        ident: &'a str,
        t: Types,
        op: Pair<'a, Rule>,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self;
    fn cmd_if(
        expr: Pair<'a, Rule>,
        true_branch: Self,
        false_branch: Option<Self>,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self;
    fn cmd_for(
        expr: Pair<'a, Rule>,
        change_assign: Self,
        block: Self,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self;
    fn cmd_while(expr: Pair<'a, Rule>, block: Self, pratt: &PrattParser<Rule>, depth: u32) -> Self;
    fn cmd_write(content: Pair<'a, Rule>, t: Types, depth: u32) -> Self;
    fn cmd_read(ident: &'a str, t: Types, depth: u32) -> Self;
}
