use crate::lang::Types;
use pest::Span;
use std::fmt::Debug;
use thiserror::Error;

pub struct TypeMismatchError<'a> {
    pub stmt_span: Span<'a>,
    pub expected: Types,
    pub provided: Types,
}

impl<'a> Debug for TypeMismatchError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (line, col) = self.stmt_span.start_pos().line_col();
        f.write_fmt(format_args!(
            "({line}:{col}) Type mismatch: expected `{}`, provided `{}`\n",
            self.expected, self.provided
        ))?;
        f.write_fmt(format_args!("\t`{}`", self.stmt_span.as_str().trim()))
    }
}

pub struct VariableAlreadyDeclaredError<'a> {
    pub original_span: Span<'a>,
    pub conflict_span: Span<'a>,
}

impl<'a> Debug for VariableAlreadyDeclaredError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (line, col) = self.conflict_span.start_pos().line_col();
        f.write_fmt(format_args!("({line}:{col}) Variable already declared\n"))?;
        f.write_fmt(format_args!(
            "\tOriginal declaration: `{}`\n",
            self.original_span.as_str().trim()
        ))?;
        f.write_fmt(format_args!(
            "\tAnd declared again: `{}`",
            self.conflict_span.as_str().trim()
        ))
    }
}

pub struct InvalidOperationError<'a> {
    pub stmt_span: Span<'a>,
    pub op_span: Span<'a>,
}

impl<'a> Debug for InvalidOperationError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (line, col) = self.op_span.start_pos().line_col();
        f.write_fmt(format_args!(
            "({line}:{col}) Invalid operation: `{}`",
            self.op_span.as_str().trim()
        ))
    }
}

#[derive(Error, Debug)]
pub enum ValidationError<'a> {
    #[error("Internal error: Missing tokens")]
    MissingTokens,

    #[error("Internal error: Missing type `{}`", .0.as_str())]
    MissingType(Span<'a>),

    #[error("{0:?}")]
    TypeMismatch(TypeMismatchError<'a>),

    #[error("{0:?}")]
    VariableAlreadyDeclared(VariableAlreadyDeclaredError<'a>),

    #[error("({}:{}) Variable not declared: `{}`", .0.start_pos().line_col().0, .0.start_pos().line_col().1, .0.as_str().trim())]
    VariableNotDeclared(Span<'a>),

    #[error("{0:?}")]
    InvalidOperation(InvalidOperationError<'a>),
}
