use std::collections::{hash_map::Entry, HashMap};

use pest::{iterators::Pair, pratt_parser::PrattParser, Span};

use crate::{
    code_gen::CodeGen,
    errors::{
        InvalidOperationError, TypeMismatchError, ValidationError, VariableAlreadyDeclaredError,
    },
    lang::{Rule, Types},
};

pub struct AST<'a>(Pair<'a, Rule>);

impl<'a> From<Pair<'a, Rule>> for AST<'a> {
    fn from(value: Pair<'a, Rule>) -> Self {
        Self(value)
    }
}

impl<'a> AST<'a> {
    pub fn validate_generate<C: CodeGen<'a>>(
        self,
        pratt: &PrattParser<Rule>,
    ) -> Result<C, ValidationError<'a>> {
        let mut ident_types = HashMap::new();
        self.inner_validate_generate::<C>(&mut ident_types, pratt, 0)
            .map(|(_, c)| c.unwrap())
    }

    fn inner_validate_generate<C: CodeGen<'a>>(
        self,
        ident_types: &mut HashMap<&'a str, (Types, Span<'a>)>,
        pratt: &PrattParser<Rule>,
        mut depth: u32,
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
                    let compiled_expr = Some(C::assign_var(
                        ident.as_str(),
                        ident_type,
                        expr,
                        pratt,
                        depth,
                    ));
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
                        depth,
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
                            depth,
                        ));
                        Ok((None, compiled_expr))
                    }
                    None => {
                        let compiled_expr = Some(C::cmd_if(
                            expr,
                            cmd_true_code_gen.unwrap(),
                            None,
                            pratt,
                            depth,
                        ));
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
                let (cmd_change_assign_type, cmd_change_assign_code_gen) = AST::from(
                    cmd_change_assign,
                )
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
                    depth,
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
