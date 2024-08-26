use pest_derive::Parser;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Types {
    Int,
    Float,
    String,
    Bool,
}

#[derive(Debug, Clone)]
pub enum Values {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl Types {
    pub fn can_do_op(self, op: Rule) -> bool {
        match self {
            Types::Int | Types::Float => matches!(
                op,
                Rule::cmd_write
                    | Rule::cmd_read
                    | Rule::negative
                    | Rule::sum
                    | Rule::sub
                    | Rule::mul
                    | Rule::div
                    | Rule::le
                    | Rule::lt
                    | Rule::ge
                    | Rule::gt
                    | Rule::ne
                    | Rule::eq
            ),
            Types::String => matches!(
                op,
                Rule::cmd_write
                    | Rule::cmd_read
                    | Rule::sum
                    | Rule::le
                    | Rule::lt
                    | Rule::ge
                    | Rule::gt
                    | Rule::ne
                    | Rule::eq
            ),
            Types::Bool => matches!(
                op,
                Rule::cmd_write | Rule::negation | Rule::ne | Rule::eq | Rule::or | Rule::and
            ),
        }
    }

    pub fn op_result_type(self, op: Rule) -> Self {
        match op {
            Rule::le
            | Rule::lt
            | Rule::ge
            | Rule::gt
            | Rule::ne
            | Rule::eq
            | Rule::or
            | Rule::and => Self::Bool,

            Rule::sum | Rule::sub | Rule::mul | Rule::div => self,

            _ => unreachable!(),
        }
    }
}

#[derive(Parser)]
#[grammar = "lang.pest"]
pub struct LangParser;