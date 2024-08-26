#![feature(iter_array_chunks)]

use std::{
    collections::{hash_map::Entry, HashMap, VecDeque},
    io::Read,
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
    provided: Types,
}

#[derive(Debug)]
struct VariableAlreadyDeclaredError<'a> {
    original_span: Span<'a>,
    conflict_span: Span<'a>,
}

#[derive(Debug)]
struct InvalidOperationError<'a> {
    stmt_span: Span<'a>,
    op_span: Span<'a>,
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
    fn validate_generate<C: CodeGen<'a>>(
        self,
        pratt: &PrattParser<Rule>,
    ) -> Result<C, ValidationError<'a>> {
        let mut ident_types = HashMap::new();
        self.inner_validate_generate::<C>(&mut ident_types, pratt, 0).map(|(_, c)| c.unwrap())
    }

    fn inner_validate_generate<C: CodeGen<'a>>(
        self,
        ident_types: &mut HashMap<&'a str, (Types, Span<'a>)>,
        pratt: &PrattParser<Rule>,
        mut depth: u32
    ) -> Result<(Option<Types>, Option<C>), ValidationError<'a>> {
        depth += 1;
        let stmt_span = self.0.as_span();
        match self.0.as_rule() {
            Rule::EOI => Ok((None, None)),

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

            Rule::int => Ok((Some(Types::Int), None)),
            Rule::float => Ok((Some(Types::Float), None)),
            Rule::string => Ok((Some(Types::String), None)),
            Rule::bool => Ok((Some(Types::Bool), None)),

            Rule::r#true => Ok((Some(Types::Bool), None)),
            Rule::r#false => Ok((Some(Types::Bool), None)),
            Rule::int_val => Ok((Some(Types::Int), None)),
            Rule::float_val => Ok((Some(Types::Float), None)),
            Rule::string_raw => Ok((Some(Types::String), None)),

            Rule::ident => match ident_types.get(self.0.as_str()) {
                Some((t, _)) => Ok((Some(*t), None)),
                None => Err(ValidationError::VariableNotDeclared(stmt_span)),
            },
            Rule::main => {
                let mut acc = Vec::new();
                for p in self.0.into_inner() {
                    let generated = AST::from(p)
                        .inner_validate_generate::<C>(ident_types, pratt, 0)?
                        .1;
                    match generated {
                        Some(generated) => acc.push(generated),
                        None => {}
                    }
                }
                let c = Some(C::main_block(acc));
                Ok((None, c))
            }
            Rule::cmd => {
                let mut acc = Vec::new();
                for p in self.0.into_inner() {
                    let generated = AST::from(p)
                        .inner_validate_generate::<C>(ident_types, pratt, depth - 1)?
                        .1;
                    match generated {
                        Some(generated) => acc.push(generated),
                        None => {}
                    }
                }
                let c = Some(C::code_block(acc));
                Ok((None, c))
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
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(lhs_span))?;

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr.as_rule(), Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if lhs_type == expr_type {
                    match ident_types.entry(ident.as_str()) {
                        Entry::Occupied(e) => Err(ValidationError::VariableAlreadyDeclared(
                            VariableAlreadyDeclaredError {
                                original_span: e.get().1,
                                conflict_span: stmt_span,
                            },
                        )),
                        Entry::Vacant(e) => {
                            let compiled_expr =
                                Some(C::declare_var(ident.as_str(), lhs_type, expr, pratt, depth));
                            e.insert((lhs_type, stmt_span));
                            Ok((None, compiled_expr))
                        }
                    }
                } else {
                    Err(ValidationError::TypeMismatch(TypeMismatchError {
                        stmt_span,
                        expected: lhs_type,
                        provided: expr_type,
                    }))
                }
            }
            Rule::cmd_read => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 1);
                let ident = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let ident_rule = ident.as_rule();
                let ident_span = ident.as_span();
                debug_assert!(matches!(ident_rule, Rule::ident), "{ident_span:?}");
                let atom_type = AST::from(ident.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(ident_span))?;

                if atom_type.can_do_op(Rule::cmd_read) {
                    let compiled_expr = Some(C::cmd_read(ident.as_str(), atom_type, depth));
                    Ok((None, compiled_expr))
                } else {
                    Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt_span,
                        op_span: stmt_span,
                    }))
                }
            }
            Rule::cmd_write => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 1);
                let val = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let val_rule = val.as_rule();
                let val_span = val.as_span();
                debug_assert!(
                    matches!(
                        val_rule,
                        Rule::string_raw
                            | Rule::int_val
                            | Rule::r#true
                            | Rule::r#false
                            | Rule::float_val
                            | Rule::ident
                    ),
                    "{val_span:?}"
                );
                let val_type = AST::from(val.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(val_span))?;

                if val_type.can_do_op(Rule::cmd_write) {
                    let compiled_expr = Some(C::cmd_write(val, val_type, depth));
                    Ok((None, compiled_expr))
                } else {
                    Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt_span,
                        op_span: stmt_span,
                    }))
                }
            }
            Rule::cmd_assign => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 2);

                let ident = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let ident_span = ident.as_span();
                debug_assert!(matches!(ident.as_rule(), Rule::ident), "{ident_span:?}");
                let ident_type = AST::from(ident.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(ident_span))?;

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr.as_rule(), Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if ident_type == expr_type {
                    let compiled_expr =
                        Some(C::assign_var(ident.as_str(), ident_type, expr, pratt, depth));
                    Ok((None, compiled_expr))
                } else {
                    Err(ValidationError::TypeMismatch(TypeMismatchError {
                        stmt_span,
                        expected: ident_type,
                        provided: expr_type,
                    }))
                }
            }
            Rule::cmd_change_assign => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 3);

                let ident = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let ident_span = ident.as_span();
                debug_assert!(matches!(ident.as_rule(), Rule::ident), "{ident_span:?}");
                let ident_type = AST::from(ident.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
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
                        stmt_span,
                        op_span,
                    }));
                }

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_rule = expr.as_rule();
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr_rule, Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(ident_span))?;

                if ident_type != expr_type {
                    Err(ValidationError::TypeMismatch(TypeMismatchError {
                        stmt_span,
                        expected: ident_type,
                        provided: expr_type,
                    }))
                } else {
                    let compiled_expr = Some(C::change_assign_var(
                        ident.as_str(),
                        ident_type,
                        op,
                        expr,
                        pratt,
                        depth
                    ));
                    Ok((None, compiled_expr))
                }
            }
            Rule::cmd_if => {
                let mut pairs = self.0.into_inner();
                debug_assert!(pairs.len() <= 3);

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr.as_rule(), Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if expr_type != Types::Bool {
                    return Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt_span,
                        op_span: expr_span,
                    }));
                }

                let cmd_true = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_true_span = cmd_true.as_span();
                debug_assert!(matches!(cmd_true.as_rule(), Rule::cmd), "{cmd_true_span:?}");
                let (cmd_true_type, cmd_true_code_gen) =
                    AST::from(cmd_true).inner_validate_generate::<C>(ident_types, pratt, depth)?;
                debug_assert!(matches!(cmd_true_type, None), "{cmd_true_span:?}");
                debug_assert!(cmd_true_code_gen.is_some(), "{cmd_true_span:?}");

                let cmd_false = pairs.next().map(
                    |cmd_false| -> Result<(Option<Types>, Option<C>), ValidationError> {
                        let cmd_false_span = cmd_false.as_span();
                        debug_assert!(
                            matches!(cmd_false.as_rule(), Rule::cmd),
                            "{cmd_false_span:?}"
                        );
                        let (cmd_false_type, cmd_false_code_gen) = AST::from(cmd_false)
                            .inner_validate_generate::<C>(ident_types, pratt, depth)?;
                        debug_assert!(matches!(cmd_false_type, None), "{cmd_false_span:?}");
                        debug_assert!(cmd_false_code_gen.is_some(), "{cmd_false_span:?}");
                        Ok((None, cmd_false_code_gen))
                    },
                );

                match cmd_false {
                    Some(cmd_false) => {
                        let (_, cmd_false_code_gen) = cmd_false?;
                        let compiled_expr = Some(C::cmd_if(
                            expr,
                            cmd_true_code_gen.unwrap(),
                            cmd_false_code_gen,
                            pratt,
                            depth
                        ));
                        Ok((None, compiled_expr))
                    }
                    None => {
                        let compiled_expr =
                            Some(C::cmd_if(expr, cmd_true_code_gen.unwrap(), None, pratt, depth));
                        Ok((None, compiled_expr))
                    }
                }
            }
            Rule::cmd_for => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 3);

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_rule = expr.as_rule();
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr_rule, Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if expr_type != Types::Bool {
                    return Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt_span,
                        op_span: expr_span,
                    }));
                }

                let cmd_change_assign = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_change_assign_rule = cmd_change_assign.as_rule();
                let cmd_change_assign_span = cmd_change_assign.as_span();
                debug_assert!(
                    matches!(cmd_change_assign_rule, Rule::cmd_change_assign),
                    "{cmd_change_assign_span:?}"
                );
                let (cmd_change_assign_type, cmd_change_assign_code_gen) =
                    AST::from(cmd_change_assign)
                        .inner_validate_generate::<C>(ident_types, pratt, depth)?;
                debug_assert!(
                    matches!(cmd_change_assign_type, None),
                    "{cmd_change_assign_span:?}"
                );
                debug_assert!(
                    cmd_change_assign_code_gen.is_some(),
                    "{cmd_change_assign_span:?}"
                );

                let cmd = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_rule = cmd.as_rule();
                let cmd_span = cmd.as_span();
                debug_assert!(matches!(cmd_rule, Rule::cmd), "{cmd_span:?}");
                let (cmd_type, cmd_code_gen) =
                    AST::from(cmd).inner_validate_generate::<C>(ident_types, pratt, depth)?;
                debug_assert!(matches!(cmd_type, None), "{cmd_span:?}");
                debug_assert!(cmd_code_gen.is_some(), "{cmd_span:?}");

                let compiled_expr = Some(C::cmd_for(
                    expr,
                    cmd_change_assign_code_gen.unwrap(),
                    cmd_code_gen.unwrap(),
                    pratt,
                    depth
                ));
                Ok((None, compiled_expr))
            }
            Rule::cmd_while => {
                let mut pairs = self.0.into_inner();
                debug_assert_eq!(pairs.len(), 2);

                let expr = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let expr_rule = expr.as_rule();
                let expr_span = expr.as_span();
                debug_assert!(matches!(expr_rule, Rule::expr), "{expr_span:?}");
                let expr_type = AST::from(expr.clone())
                    .inner_validate_generate::<C>(ident_types, pratt, depth)?
                    .0
                    .ok_or(ValidationError::MissingType(expr_span))?;

                if expr_type != Types::Bool {
                    return Err(ValidationError::InvalidOperation(InvalidOperationError {
                        stmt_span,
                        op_span: expr_span,
                    }));
                }

                let cmd = pairs.next().ok_or(ValidationError::MissingTokens)?;
                let cmd_rule = cmd.as_rule();
                let cmd_span = cmd.as_span();
                debug_assert!(matches!(cmd_rule, Rule::cmd), "{cmd_span:?}");
                let (cmd_type, cmd_code_gen) =
                    AST::from(cmd).inner_validate_generate::<C>(ident_types, pratt, depth)?;
                debug_assert!(matches!(cmd_type, None), "{cmd_span:?}");
                debug_assert!(cmd_code_gen.is_some(), "{cmd_span:?}");

                let compiled_expr = Some(C::cmd_while(expr, cmd_code_gen.unwrap(), pratt, depth));
                Ok((None, compiled_expr))
            }
            Rule::expr => {
                let pairs = self.0.into_inner();
                pratt
                    .map_primary(|atom| match atom.as_rule() {
                        Rule::r#true
                        | Rule::r#false
                        | Rule::int_val
                        | Rule::float_val
                        | Rule::string_raw
                        | Rule::expr
                        | Rule::ident => {
                            AST::from(atom).inner_validate_generate::<C>(ident_types, pratt, depth)
                        }
                        _ => unreachable!(),
                    })
                    .map_prefix(|op, atom| {
                        let atom = atom?.0.ok_or(ValidationError::MissingType(stmt_span))?;
                        if atom.can_do_op(op.as_rule()) {
                            Ok((Some(atom), None))
                        } else {
                            Err(ValidationError::InvalidOperation(InvalidOperationError {
                                stmt_span,
                                op_span: op.as_span(),
                            }))
                        }
                    })
                    .map_infix(|lhs, op, rhs| {
                        let lhs = lhs?.0.ok_or(ValidationError::MissingType(stmt_span))?;
                        let rhs = rhs?.0.ok_or(ValidationError::MissingType(stmt_span))?;
                        if !lhs.can_do_op(op.as_rule()) {
                            return Err(ValidationError::InvalidOperation(InvalidOperationError {
                                stmt_span,
                                op_span: op.as_span(),
                            }));
                        }

                        if lhs != rhs {
                            return Err(ValidationError::TypeMismatch(TypeMismatchError {
                                stmt_span,
                                expected: lhs,
                                provided: rhs,
                            }));
                        }

                        Ok((Some(lhs.op_result_type(op.as_rule())), None))
                    })
                    .parse(pairs)
            }
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

trait CodeGen<'a>: Sized {
    fn main_block(acc: Vec<Self>) -> Self;
    fn code_block(acc: Vec<Self>) -> Self;
    fn declare_var(
        ident: &'a str,
        t: Types,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self;
    fn assign_var(
        ident: &'a str,
        t: Types,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self;
    fn change_assign_var(
        ident: &'a str,
        t: Types,
        op: Pair<'a, Rule>,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self;
    fn cmd_if(
        expr: Pair<'a, Rule>,
        true_branch: Self,
        false_branch: Option<Self>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self;
    fn cmd_for(
        expr: Pair<'a, Rule>,
        change_assign: Self,
        block: Self,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self;
    fn cmd_while(expr: Pair<'a, Rule>, block: Self, pratt: &PrattParser<Rule>, depth: u32) -> Self;
    fn cmd_write(content: Pair<'a, Rule>, t: Types, depth: u32) -> Self;
    fn cmd_read(ident: &'a str, t: Types, depth: u32) -> Self;
}

#[derive(Debug, Clone)]
enum ByteCode<'a> {
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

    fn declare_var(ident: &'a str, _: Types, expr: Pair<'a, Rule>, _: &PrattParser<Rule>, _: u32) -> Self {
        Self::DeclareVar(ident.to_string(), expr)
    }

    fn assign_var(ident: &'a str, _: Types, expr: Pair<'a, Rule>, _: &PrattParser<Rule>, _: u32) -> Self {
        Self::AssignVar(ident.to_string(), expr)
    }

    fn change_assign_var(
        ident: &'a str,
        _: Types,
        op: Pair<'a, Rule>,
        expr: Pair<'a, Rule>,
        _: &PrattParser<Rule>,
        _: u32
    ) -> Self {
        Self::ChangeAssignVar(ident.to_string(), op, expr)
    }

    fn cmd_if(
        expr: Pair<'a, Rule>,
        true_branch: Self,
        false_branch: Option<Self>,
        _: &PrattParser<Rule>,
        _: u32
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
        _: u32
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
    fn eval(self, pratt: &PrattParser<Rule>, mem: &mut HashMap<String, Values>) {
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

struct CppCode(String);

impl CppCode {
    fn gen_expr<'a>(expr: Pair<'a, Rule>, pratt: &PrattParser<Rule>) -> String {
        let pairs = expr.into_inner();
        pratt
            .map_primary(|atom| match atom.as_rule() {
                Rule::ident | Rule::int_val | Rule::float_val | Rule::r#true | Rule::r#false => {
                    format!("({})", atom.as_str())
                }
                Rule::expr => format!("({})", Self::gen_expr(atom, pratt)),
                Rule::string_raw => format!("std::string(\"{}\")", atom.as_str()),
                _ => unreachable!(),
            })
            .map_prefix(|op, val| match op.as_rule() {
                Rule::negative => format!("(-{val})"),
                Rule::negation => format!("(!{val})"),
                _ => unreachable!(),
            })
            .map_infix(|lhs, op, rhs| match op.as_rule() {
                Rule::sum => format!("({lhs} + {rhs})"),
                Rule::sub => format!("({lhs} - {rhs})"),
                Rule::mul => format!("({lhs} * {rhs})"),
                Rule::div => format!("({lhs} / {rhs})"),
                Rule::le => format!("({lhs} <= {rhs})"),
                Rule::lt => format!("({lhs} < {rhs})"),
                Rule::ge => format!("({lhs} >= {rhs})"),
                Rule::gt => format!("({lhs} > {rhs})"),
                Rule::ne => format!("({lhs} != {rhs})"),
                Rule::eq => format!("({lhs} == {rhs})"),
                Rule::or => format!("({lhs} || {rhs})"),
                Rule::and => format!("({lhs} && {rhs})"),
                _ => unreachable!(),
            })
            .parse(pairs)
    }
    
    fn gen_indentation(depth: u32) -> String {
        "    ".repeat(depth as usize)
    }
}

impl<'a> CodeGen<'a> for CppCode {
    fn main_block(acc: Vec<Self>) -> Self {
        let mut output = String::from(
            r#"#include <iostream>
#include <string>

int main() {
"#,
        );

        for code in acc {
            output.push_str(&code.0);
            output.push('\n');
        }

        output.push_str("    return 0;\n");
        output.push('}');

        Self(output)
    }

    fn code_block(acc: Vec<Self>) -> Self {
        let mut output = String::new();
        for code in acc {
            output.push_str(&code.0);
            output.push('\n');
        }
        Self(output)
    }

    fn declare_var(
        ident: &'a str,
        t: Types,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self {
        let indentation = Self::gen_indentation(depth);
        let t = match t {
            Types::Int => "long long int",
            Types::Float => "double",
            Types::String => "std::string",
            Types::Bool => "bool",
        };
        Self(format!("{indentation}{t} {ident} = {};", Self::gen_expr(expr, pratt)))
    }

    fn assign_var(
        ident: &'a str,
        _: Types,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self {
        let indentation = Self::gen_indentation(depth);
        Self(format!("{indentation}{ident} = {};", Self::gen_expr(expr, pratt)))
    }

    fn change_assign_var(
        ident: &'a str,
        _: Types,
        op: Pair<'a, Rule>,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self {
        let op = match op.as_rule() {
            Rule::sum => "+",
            Rule::sub => "-",
            Rule::mul => "*",
            Rule::div => "/",
            _ => unreachable!(),
        };

        let indentation = Self::gen_indentation(depth);
        Self(format!("{indentation}{ident} {op}= {};", Self::gen_expr(expr, pratt)))
    }

    fn cmd_if(
        expr: Pair<'a, Rule>,
        true_branch: Self,
        false_branch: Option<Self>,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self {
        let indentation = Self::gen_indentation(depth);
        let output = format!(
            "{indentation}if ({}) {{\n{}{indentation}}}",
            Self::gen_expr(expr, pratt),
            true_branch.0
        );
        match false_branch {
            Some(false_branch) => Self(format!("{output} else {{\n{}{indentation}}}", false_branch.0)),
            None => Self(output),
        }
    }

    fn cmd_for(
        expr: Pair<'a, Rule>,
        mut change_assign: Self,
        block: Self,
        pratt: &PrattParser<Rule>,
        depth: u32
    ) -> Self {
        change_assign.0.pop();
        let indentation = Self::gen_indentation(depth);
        Self(format!(
            "{indentation}for (; {}; {}) {{\n{}{indentation}}}",
            Self::gen_expr(expr, pratt),
            change_assign.0.trim_start(),
            block.0
        ))
    }

    fn cmd_while(expr: Pair<'a, Rule>, block: Self, pratt: &PrattParser<Rule>, depth: u32) -> Self {
        let indentation = Self::gen_indentation(depth);
        Self(format!(
            "{indentation}while ({}) {{\n{}\n{indentation}}}",
            Self::gen_expr(expr, pratt),
            block.0
        ))
    }

    fn cmd_write(content: Pair<'a, Rule>, _: Types, depth: u32) -> Self {
        let indentation = Self::gen_indentation(depth);
        match content.as_rule() {
            Rule::string_raw => Self(format!("{indentation}std::cout << \"{}\" << '\\n';", content.as_str())),
            Rule::int_val | Rule::float_val | Rule::r#true | Rule::r#false | Rule::ident => {
                Self(format!("{indentation}std::cout << {} << '\\n';", content.as_str()))
            }
            _ => unreachable!(),
        }
    }

    fn cmd_read(ident: &'a str, _: Types, depth: u32) -> Self {
        let indentation = Self::gen_indentation(depth);
        Self(format!("{indentation}std::cin >> {ident};"))
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
        .op(Op::infix(Rule::mul, Assoc::Left) | Op::infix(Rule::div, Assoc::Left))
        .op(Op::prefix(Rule::negation) | Op::prefix(Rule::negative));

    let unparsed_file = std::fs::read_to_string("test1.isi").expect("cannot read file");

    let pair = LangParser::parse(Rule::main, &unparsed_file)
        .expect("unsuccessful parse")
        .next()
        .unwrap();

    let gen = AST::from(pair)
        .validate_generate::<CppCode>(&pratt)
        .unwrap();

    // let mut mem = HashMap::new();
    // gen.eval(&pratt, &mut mem);
    println!("{}", gen.0);
}
