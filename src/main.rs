#![feature(iter_array_chunks)]

use std::{
    collections::{hash_map::Entry, HashMap},
    marker::PhantomData,
};

use pest::{
    iterators::Pair,
    pratt_parser::{Assoc, Op, PrattParser},
    Parser, Span,
};
use pest_derive::Parser;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Types {
    Int,
    Float,
    String,
    Bool,
}

impl Types {
    fn can_do_op(self, op: Rule) -> bool {
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
            // TODO: Add comparisons
            Types::String => matches!(op, Rule::cmd_write | Rule::cmd_read),
            Types::Bool => matches!(
                op,
                Rule::cmd_write | Rule::negation | Rule::ne | Rule::eq | Rule::or | Rule::and
            ),
        }
    }

    fn op_result_type(self, op: Rule) -> Self {
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
struct LangParser;

#[derive(Debug)]
struct TypeMismatchError<'a> {
    stmt_span: Span<'a>,
    expected: Types,
    expected_span: Span<'a>,
    provided: Types,
    provided_span: Span<'a>,
}

#[derive(Debug)]
struct VariableAlreadyDeclaredError<'a> {
    original_span: Span<'a>,
    conflict_span: Span<'a>,
}

#[derive(Debug)]
struct InvalidOperationError<'a> {
    stmt: Span<'a>,
    op_span: Span<'a>,
    operand_span: Span<'a>,
}

#[derive(Debug)]
enum ValidationError<'a> {
    MissingTokens,
    MissingType(Span<'a>),
    TypeMismatch(TypeMismatchError<'a>),
    VariableAlreadyDeclared(VariableAlreadyDeclaredError<'a>),
    VariableNotDeclared(Span<'a>),
    InvalidOperation(InvalidOperationError<'a>),
}

struct AST<'a>(Pair<'a, Rule>);

impl<'a> From<Pair<'a, Rule>> for AST<'a> {
    fn from(value: Pair<'a, Rule>) -> Self {
        Self(value)
    }
}

impl<'a> AST<'a> {
    fn validate_generate(self, pratt: &PrattParser<Rule>) -> Result<Option<Types>, ValidationError<'a>> {
        let mut ident_types = HashMap::new();
        // let mut code_gen = T::new();
        self.inner_validate_generate(&mut ident_types, pratt)
    }

    fn inner_validate_generate(
        self,
        ident_types: &mut HashMap<&'a str, (Types, Span<'a>)>,
        pratt: &PrattParser<Rule>
    ) -> Result<Option<Types>, ValidationError<'a>> {
        let self_span = self.0.as_span();
        match self.0.as_rule() {
            Rule::EOI => Ok(None),

            Rule::WHITESPACE
            | Rule::endl
            | Rule::cmd_op
            | Rule::cmd_st
            | Rule::atom
            | Rule::op_bin
            | Rule::types
            | Rule::bool_val
            | Rule::string_val
            | Rule::reserved
            | Rule::op_una => unreachable!(),

            Rule::sum
            | Rule::sub
            | Rule::mul
            | Rule::div
            | Rule::le
            | Rule::lt
            | Rule::ge
            | Rule::gt
            | Rule::ne
            | Rule::eq
            | Rule::or
            | Rule::and => unreachable!(),

            Rule::negation | Rule::negative => unreachable!(),

            Rule::int => Ok(Some(Types::Int)),
            Rule::float => Ok(Some(Types::Float)),
            Rule::string => Ok(Some(Types::String)),
            Rule::bool => Ok(Some(Types::Bool)),

            Rule::r#true => Ok(Some(Types::Bool)),
            Rule::r#false => Ok(Some(Types::Bool)),
            Rule::int_val => Ok(Some(Types::Int)),
            Rule::float_val => Ok(Some(Types::Float)),
            Rule::string_raw => Ok(Some(Types::String)),
            Rule::string_construct => Ok(Some(Types::String)),

            Rule::ident => match ident_types.get(self.0.as_str()) {
                Some((t, _)) => Ok(Some(*t)),
                None => Err(ValidationError::VariableNotDeclared(self_span)),
            },
            Rule::main => {
                for p in self.0.into_inner() {
                    AST::from(p).inner_validate_generate(ident_types, pratt)?;
                }
                Ok(None)
            }
            Rule::var_dec => {
                let mut pairs = self.0.into_inner();
                let ident = pairs.next().ok_or(ValidationError::MissingTokens)?;
                debug_assert_eq!(ident.as_rule(), Rule::ident);

                let lhs = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let lhs_span = lhs.as_span();
                debug_assert!(
                    matches!(
                        lhs.as_rule(),
                        Rule::int | Rule::float | Rule::string | Rule::bool
                    ),
                    "{lhs_span:?}"
                );
                let lhs_type = AST::from(lhs)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(lhs_span))?;

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr.as_rule(), Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr.clone())
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if lhs_type == expr_type {
                    match ident_types.entry(ident.as_str()) {
                        Entry::Occupied(e) => Err(ValidationError::VariableAlreadyDeclared(
                            VariableAlreadyDeclaredError {
                                original_span: e.get().1,
                                conflict_span: self_span,
                            },
                        )),
                        Entry::Vacant(e) => {
                            let compiled_expr =
                                ByteCode::declare_variable(ident.as_str(), lhs_type, expr);
                            println!("{compiled_expr:#?}");
                            let mem = HashMap::new();
                            let r = match compiled_expr {
                                ByteCode::DeclareVar(_, _, e) => e.eval(&mem),
                            };
                            println!("{r:#?}");
                            e.insert((lhs_type, self_span));
                            Ok(None)
                        }
                    }
                } else {
                    Err(ValidationError::TypeMismatch(TypeMismatchError {
                        stmt_span: self_span,
                        expected: lhs_type,
                        expected_span: lhs_span,
                        provided: expr_type,
                        provided_span: expr_span,
                    }))
                }
            }
            Rule::cmd => {
                for p in self.0.into_inner() {
                    AST::from(p).inner_validate_generate(ident_types, pratt)?;
                }
                Ok(None)
            }
            Rule::cmd_read => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 1);
                let ident = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let ident_rule = ident.as_rule();
                let ident_span = ident.as_span();
                debug_assert!(matches!(ident_rule, Rule::ident), "{ident_span:?}");
                let atom_type = AST::from(ident)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(ident_span))?;

                if atom_type.can_do_op(Rule::cmd_read) {
                    Ok(None)
                } else {
                    Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt: self_span,
                        op_span: self_span,
                        operand_span: ident_span,
                    }))
                }
            }
            Rule::cmd_write => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 1);
                let val = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let val_rule = val.as_rule();
                let val_span = val.as_span();
                debug_assert!(matches!(val_rule, Rule::ident), "{val_span:?}");
                let atom_type = AST::from(val)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(val_span))?;

                if atom_type.can_do_op(Rule::cmd_write) {
                    Ok(None)
                } else {
                    Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt: self_span,
                        op_span: self_span,
                        operand_span: val_span,
                    }))
                }
            }
            Rule::cmd_assign => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 2);

                let ident = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let ident_span = ident.as_span();
                debug_assert!(matches!(ident.as_rule(), Rule::ident), "{ident_span:?}");
                let ident_type = AST::from(ident)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(ident_span))?;

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr.as_rule(), Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if ident_type == expr_type {
                    Ok(None)
                } else {
                    Err(ValidationError::TypeMismatch(TypeMismatchError {
                        stmt_span: self_span,
                        expected: ident_type,
                        expected_span: ident_span,
                        provided: expr_type,
                        provided_span: expr_span,
                    }))
                }
            }
            Rule::cmd_change_assign => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 3);

                let ident = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let ident_span = ident.as_span();
                debug_assert!(matches!(ident.as_rule(), Rule::ident), "{ident_span:?}");
                let ident_type = AST::from(ident)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(ident_span))?;

                let op = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let op_rule = op.as_rule();
                let op_span = op.as_span();
                debug_assert!(
                    matches!(op_rule, Rule::sum | Rule::sub | Rule::mul | Rule::div),
                    "{op_span:?}"
                );

                if !ident_type.can_do_op(op_rule) {
                    return Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt: self_span,
                        op_span,
                        operand_span: ident_span,
                    }));
                }

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_rule = expr.as_rule();
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr_rule, Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(ident_span))?;

                if ident_type != expr_type {
                    Err(ValidationError::TypeMismatch(TypeMismatchError {
                        stmt_span: self_span,
                        expected: ident_type,
                        expected_span: ident_span,
                        provided: expr_type,
                        provided_span: expr_span,
                    }))
                } else {
                    Ok(None)
                }
            }
            Rule::cmd_if => {
                let mut pairs = self.0.into_inner();
                debug_assert!(pairs.len() <= 3);

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr.as_rule(), Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if expr_type != Types::Bool {
                    return Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt: self_span,
                        op_span: expr_span,
                        operand_span: expr_span,
                    }));
                }

                let cmd_true = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_true_span = cmd_true.as_span();
                debug_assert!(matches!(cmd_true.as_rule(), Rule::cmd), "{cmd_true_span:?}");
                let cmd_true_type = AST::from(cmd_true).inner_validate_generate(ident_types, pratt)?;
                debug_assert!(matches!(cmd_true_type, None), "{cmd_true_span:?}");

                let cmd_false =
                    pairs
                        .next()
                        .map(|cmd_false| -> Result<Option<Types>, ValidationError> {
                            let cmd_false_span = cmd_false.as_span();
                            debug_assert!(
                                matches!(cmd_false.as_rule(), Rule::cmd),
                                "{cmd_false_span:?}"
                            );
                            let cmd_false_type =
                                AST::from(cmd_false).inner_validate_generate(ident_types, pratt)?;
                            debug_assert!(matches!(cmd_false_type, None), "{cmd_false_span:?}");
                            Ok(None)
                        });

                match cmd_false {
                    Some(cmd_false) => cmd_false,
                    None => Ok(None),
                }
            }
            Rule::cmd_for => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 3);

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_rule = expr.as_rule();
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr_rule, Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if expr_type != Types::Bool {
                    return Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt: self_span,
                        op_span: expr_span,
                        operand_span: expr_span,
                    }));
                }

                let cmd_change_assign = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_change_assign_rule = cmd_change_assign.as_rule();
                let cmd_change_assign_span = cmd_change_assign.as_span();
                debug_assert!(
                    matches!(cmd_change_assign_rule, Rule::cmd_change_assign),
                    "{cmd_change_assign_span:?}"
                );
                let cmd_change_assign_type =
                    AST::from(cmd_change_assign).inner_validate_generate(ident_types, pratt)?;
                debug_assert!(
                    matches!(cmd_change_assign_type, None),
                    "{cmd_change_assign_span:?}"
                );

                let cmd = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_rule = cmd.as_rule();
                let cmd_span = cmd.as_span();
                debug_assert!(matches!(cmd_rule, Rule::cmd), "{cmd_span:?}");
                let cmd_type = AST::from(cmd).inner_validate_generate(ident_types, pratt)?;
                debug_assert!(matches!(cmd_type, None), "{cmd_span:?}");

                Ok(None)
            }
            Rule::cmd_while => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 2);

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_rule = expr.as_rule();
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr_rule, Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if expr_type != Types::Bool {
                    return Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt: self_span,
                        op_span: expr_span,
                        operand_span: expr_span,
                    }));
                }

                let cmd = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_rule = cmd.as_rule();
                let cmd_span = cmd.as_span();
                debug_assert!(matches!(cmd_rule, Rule::cmd), "{cmd_span:?}");
                let cmd_type = AST::from(cmd).inner_validate_generate(ident_types, pratt)?;
                debug_assert!(matches!(cmd_type, None), "{cmd_span:?}");

                Ok(None)
            }
            Rule::expr => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len() % 2, 1);
                let atom = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let atom_span = atom.as_span();
                debug_assert!(
                    matches!(
                        atom.as_rule(),
                        Rule::expr_una
                            | Rule::r#true
                            | Rule::r#false
                            | Rule::float_val
                            | Rule::int_val
                            | Rule::string_raw
                            | Rule::string_construct
                            | Rule::ident
                            | Rule::expr
                    ),
                    "{atom_span:?}"
                );
                let mut result_type = AST::from(atom)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(atom_span))?;

                for [op, rhs] in pairs.array_chunks::<2>() {
                    let op_rule = op.as_rule();
                    let op_span = op.as_span();
                    debug_assert!(
                        matches!(
                            op_rule,
                            Rule::sum
                                | Rule::sub
                                | Rule::mul
                                | Rule::div
                                | Rule::le
                                | Rule::lt
                                | Rule::ge
                                | Rule::gt
                                | Rule::ne
                                | Rule::eq
                                | Rule::or
                                | Rule::and
                        ),
                        "{op_span:?}"
                    );

                    if !result_type.can_do_op(op_rule) {
                        return Err(ValidationError::InvalidOperation(InvalidOperationError {
                            stmt: self_span,
                            op_span,
                            operand_span: atom_span,
                        }));
                    }

                    let rhs_span = rhs.as_span();
                    debug_assert!(
                        matches!(
                            rhs.as_rule(),
                            Rule::expr_una
                                | Rule::r#true
                                | Rule::r#false
                                | Rule::float_val
                                | Rule::int_val
                                | Rule::string_raw
                                | Rule::string_construct
                                | Rule::ident
                                | Rule::expr
                        ),
                        "{rhs_span:?}"
                    );
                    let rhs_type = AST::from(rhs)
                        .inner_validate_generate(ident_types, pratt)?
                        .ok_or(ValidationError::MissingType(rhs_span))?;

                    if result_type != rhs_type {
                        return Err(ValidationError::TypeMismatch(TypeMismatchError {
                            stmt_span: self_span,
                            expected: result_type,
                            expected_span: atom_span,
                            provided: rhs_type,
                            provided_span: rhs_span,
                        }));
                    }

                    result_type = result_type.op_result_type(op_rule);
                }
                return Ok(Some(result_type));
            }
            Rule::expr_una => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 2);

                let op = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let op_rule = op.as_rule();
                let op_span = op.as_span();
                debug_assert!(
                    matches!(op_rule, Rule::negation | Rule::negative),
                    "{op_span:?}"
                );

                let operand = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let operand_rule = operand.as_rule();
                let operand_span = operand.as_span();
                debug_assert!(
                    matches!(
                        operand_rule,
                        Rule::r#true | Rule::r#false | Rule::ident | Rule::expr
                    ),
                    "{operand_span:?}"
                );
                let operand_type = AST::from(operand)
                    .inner_validate_generate(ident_types, pratt)?
                    .ok_or(ValidationError::MissingType(operand_span))?;

                if operand_type.can_do_op(op_rule) {
                    Ok(Some(operand_type))
                } else {
                    Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt: self_span,
                        op_span,
                        operand_span,
                    }))
                }
            }
        }
    }
}

#[derive(Debug)]
enum UnaryOp {
    Negation,
    Negative,
}

impl UnaryOp {
    fn apply(self, val: Values) -> Values {
        match val {
            Values::Int(val) => match self {
                UnaryOp::Negative => Values::Int(-val),
                UnaryOp::Negation => unreachable!(),
            },
            Values::Float(val) => match self {
                UnaryOp::Negative => Values::Float(-val),
                UnaryOp::Negation => unreachable!(),
            },
            Values::String(_) => unreachable!(),
            Values::Bool(val) => match self {
                UnaryOp::Negation => Values::Bool(!val),
                UnaryOp::Negative => unreachable!(),
            },
        }
    }
}

#[derive(Debug)]
enum BinOp {
    Sum,
    Sub,
    Mul,
    Div,
    Le,
    Lt,
    Ge,
    Gt,
    Ne,
    Eq,
    Or,
    And,
}

impl BinOp {
    fn apply(self, lhs: Values, rhs: Values) -> Values {
        match (lhs, rhs) {
            (Values::Int(lhs), Values::Int(rhs)) => match self {
                BinOp::Sum => Values::Int(lhs + rhs),
                BinOp::Sub => Values::Int(lhs - rhs),
                BinOp::Mul => Values::Int(lhs * rhs),
                BinOp::Div => Values::Int(lhs / rhs),
                BinOp::Le => Values::Bool(lhs <= rhs),
                BinOp::Lt => Values::Bool(lhs < rhs),
                BinOp::Ge => Values::Bool(lhs >= rhs),
                BinOp::Gt => Values::Bool(lhs > rhs),
                BinOp::Ne => Values::Bool(lhs != rhs),
                BinOp::Eq => Values::Bool(lhs == rhs),
                BinOp::Or => unreachable!(),
                BinOp::And => unreachable!(),
            },
            (Values::Float(lhs), Values::Float(rhs)) => match self {
                BinOp::Sum => Values::Float(lhs + rhs),
                BinOp::Sub => Values::Float(lhs - rhs),
                BinOp::Mul => Values::Float(lhs * rhs),
                BinOp::Div => Values::Float(lhs / rhs),
                BinOp::Le => Values::Bool(lhs <= rhs),
                BinOp::Lt => Values::Bool(lhs < rhs),
                BinOp::Ge => Values::Bool(lhs >= rhs),
                BinOp::Gt => Values::Bool(lhs > rhs),
                BinOp::Ne => Values::Bool(lhs != rhs),
                BinOp::Eq => Values::Bool(lhs == rhs),
                BinOp::Or => unreachable!(),
                BinOp::And => unreachable!(),
            },
            (Values::String(lhs), Values::String(rhs)) => match self {
                BinOp::Le => Values::Bool(lhs <= rhs),
                BinOp::Lt => Values::Bool(lhs < rhs),
                BinOp::Ge => Values::Bool(lhs >= rhs),
                BinOp::Gt => Values::Bool(lhs > rhs),
                BinOp::Ne => Values::Bool(lhs != rhs),
                BinOp::Eq => Values::Bool(lhs == rhs),
                BinOp::Sum => unreachable!(),
                BinOp::Sub => unreachable!(),
                BinOp::Mul => unreachable!(),
                BinOp::Div => unreachable!(),
                BinOp::Or => unreachable!(),
                BinOp::And => unreachable!(),
            },
            (Values::Bool(lhs), Values::Bool(rhs)) => match self {
                BinOp::Ne => Values::Bool(lhs != rhs),
                BinOp::Eq => Values::Bool(lhs == rhs),
                BinOp::Or => Values::Bool(lhs || rhs),
                BinOp::And => Values::Bool(lhs && rhs),
                BinOp::Le => unreachable!(),
                BinOp::Lt => unreachable!(),
                BinOp::Ge => unreachable!(),
                BinOp::Gt => unreachable!(),
                BinOp::Sum => unreachable!(),
                BinOp::Sub => unreachable!(),
                BinOp::Mul => unreachable!(),
                BinOp::Div => unreachable!(),
            },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct UnaryExpr {
    op: UnaryOp,
    atom: Box<Atom>,
}

impl UnaryExpr {
    pub fn eval(self, mem: &HashMap<String, Values>) -> Values {
        self.op.apply(self.atom.eval(mem))
    }

    fn build<'a>(pair: Pair<'a, Rule>) -> Self {
        let mut pairs = pair.into_inner();

        let op = pairs.next().unwrap();
        let op = match op.as_rule() {
            Rule::negation => UnaryOp::Negation,
            Rule::negative => UnaryOp::Negative,
            _ => unreachable!(),
        };

        let atom = pairs.next().unwrap();
        Self {
            op,
            atom: Box::new(Atom::build(atom)),
        }
    }
}

#[derive(Debug)]
enum Atom {
    UnaryExpr(Box<UnaryExpr>),
    Raw(Values),
    Ident(String),
    Expr(Box<Expr>),
}

impl Atom {
    fn eval(self, mem: &HashMap<String, Values>) -> Values {
        match self {
            Atom::UnaryExpr(u) => u.eval(mem),
            Atom::Raw(t) => t.into(),
            Atom::Ident(ident) => mem.get(&ident).unwrap().clone(),
            Atom::Expr(e) => e.eval(mem),
        }
    }

    fn build<'a>(atom: Pair<'a, Rule>) -> Self {
        match atom.as_rule() {
            Rule::ident => Self::Ident(atom.as_str().to_owned()),

            Rule::expr_una => Self::UnaryExpr(Box::new(UnaryExpr::build(atom))),
            Rule::expr => Self::Expr(Box::new(Expr::build(atom))),

            Rule::int_val => Self::Raw(Values::Int(atom.as_str().parse().unwrap())),
            Rule::float_val => Self::Raw(Values::Float(atom.as_str().parse().unwrap())),
            Rule::r#true => Self::Raw(Values::Bool(atom.as_str().parse().unwrap())),
            Rule::r#false => Self::Raw(Values::Bool(atom.as_str().parse().unwrap())),
            Rule::string_raw => Self::Raw(Values::String(atom.as_str().to_owned())),
            //TODO: Fix this
            Rule::string_construct => Self::Raw(Values::String(atom.as_str().parse().unwrap())),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct Expr {
    atom: Box<Atom>,
    bin_op_atom: Vec<(BinOp, Atom)>,
}

impl Expr {
    fn eval(self, mem: &HashMap<String, Values>) -> Values {
        let mut result = self.atom.eval(mem);
        for (bin_op, atom) in self.bin_op_atom {
            let atom = atom.eval(mem);
            result = bin_op.apply(result, atom);
        }
        result
    }

    fn build<'a>(expr: Pair<'a, Rule>) -> Self {
        let mut pairs = expr.into_inner();
        let atom = pairs.next().unwrap();
        let atom = Atom::build(atom);

        let bin_op_atom = pairs
            .array_chunks::<2>()
            .map(|[bin_op, atom]| {
                let bin_op = match bin_op.as_rule() {
                    Rule::sum => BinOp::Sum,
                    Rule::sub => BinOp::Sub,
                    Rule::mul => BinOp::Mul,
                    Rule::div => BinOp::Div,
                    Rule::le => BinOp::Le,
                    Rule::lt => BinOp::Lt,
                    Rule::ge => BinOp::Ge,
                    Rule::gt => BinOp::Gt,
                    Rule::ne => BinOp::Ne,
                    Rule::eq => BinOp::Eq,
                    Rule::or => BinOp::Or,
                    Rule::and => BinOp::And,
                    _ => unreachable!(),
                };
                let atom = Atom::build(atom);
                (bin_op, atom)
            })
            .collect();

        Self {
            atom: Box::new(atom),
            bin_op_atom,
        }
    }
}

#[derive(Debug, Clone)]
enum Values {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl From<i64> for Values {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for Values {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<String> for Values {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<bool> for Values {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

#[derive(Debug)]
enum ByteCode {
    DeclareVar(String, Types, Expr),
}

trait CodeGen {
    fn declare_variable<'a>(ident: &'a str, t: Types, expr: Pair<'a, Rule>) -> Self;
}

impl CodeGen for ByteCode {
    fn declare_variable<'a>(ident: &'a str, t: Types, expr: Pair<'a, Rule>) -> Self {
        Self::DeclareVar(ident.to_string(), t, Expr::build(expr))
    }
}

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
        .op(Op::infix(Rule::mul, Assoc::Left) | Op::infix(Rule::div, Assoc::Left));

    // let expr_type_parser = pratt
    // .map_primary(|atom| match atom.as_rule() {

    // });

    let unparsed_file = std::fs::read_to_string("test1.isi").expect("cannot read file");

    let pair = LangParser::parse(Rule::main, &unparsed_file)
        .expect("unsuccessful parse") // unwrap the parse result
        .next()
        .unwrap(); // get and unwrap the `file` rule; never fails

    // println!("{pair:#?}");

    AST::from(pair).validate_generate(&pratt).unwrap();
    // println!("{}", c_code.0);
}
