use pest::Span;

use crate::lang::Types;

#[derive(Debug)]
pub struct TypeMismatchError<'a> {
    pub stmt_span: Span<'a>,
    pub expected: Types,
    pub provided: Types,
}

#[derive(Debug)]
pub struct VariableAlreadyDeclaredError<'a> {
    pub original_span: Span<'a>,
    pub conflict_span: Span<'a>,
}

#[derive(Debug)]
pub struct InvalidOperationError<'a> {
    pub stmt_span: Span<'a>,
    pub op_span: Span<'a>,
}

#[derive(Debug)]
pub enum ValidationError<'a> {
    MissingTokens,
    MissingType(Span<'a>),
    TypeMismatch(TypeMismatchError<'a>),
    VariableAlreadyDeclared(VariableAlreadyDeclaredError<'a>),
    VariableNotDeclared(Span<'a>),
    InvalidOperation(InvalidOperationError<'a>),
}
