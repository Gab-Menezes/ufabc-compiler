use pest::{iterators::Pair, pratt_parser::PrattParser};

use crate::lang::{Rule, Types};

use super::CodeGen;

pub struct Java(String);

impl Java {
    pub fn into_inner(self) -> String {
        self.0
    }

    fn gen_expr(expr: Pair<'_, Rule>, pratt: &PrattParser<Rule>) -> String {
        let pairs = expr.into_inner();
        pratt
            .map_primary(|atom| match atom.as_rule() {
                Rule::ident | Rule::int_val | Rule::float_val | Rule::r#true | Rule::r#false => {
                    format!("({})", atom.as_str())
                }
                Rule::expr => format!("({})", Self::gen_expr(atom, pratt)),
                Rule::string_raw => format!("(new String(\"{}\"))", atom.as_str()),
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
                Rule::le => format!("compareLe({lhs}, {rhs})"),
                Rule::lt => format!("compareLt({lhs}, {rhs})"),
                Rule::ge => format!("compareGe({lhs}, {rhs})"),
                Rule::gt => format!("compareGt({lhs}, {rhs})"),
                Rule::ne => format!("compareNe({lhs}, {rhs})"),
                Rule::eq => format!("compareEq({lhs}, {rhs})"),
                Rule::or => format!("({lhs} || {rhs})"),
                Rule::and => format!("({lhs} && {rhs})"),
                _ => unreachable!(),
            })
            .parse(pairs)
    }

    fn gen_indentation(depth: u32) -> String {
        "    ".repeat((depth + 1) as usize)
    }
}

impl<'a> CodeGen<'a> for Java {
    fn main_block(acc: Vec<Self>) -> Self {
        let mut output = String::from(
            r#"import java.util.Scanner;

public class Main {
    public static boolean compareLe(int a, int b) {
        return a <= b;
    }
    public static boolean compareLe(double a, double b) {
        return a <= b;
    }
    public static boolean compareLe(String a, String b) {
        return a.compareTo(b) <= 0;
    }

    public static boolean compareLt(int a, int b) {
        return a < b;
    }
    public static boolean compareLt(double a, double b) {
        return a < b;
    }
    public static boolean compareLt(String a, String b) {
        return a.compareTo(b) < 0;
    }

    public static boolean compareGe(int a, int b) {
        return a >= b;
    }
    public static boolean compareGe(double a, double b) {
        return a >= b;
    }
    public static boolean compareGe(String a, String b) {
        return a.compareTo(b) >= 0;
    }

    public static boolean compareGt(int a, int b) {
        return a > b;
    }
    public static boolean compareGt(double a, double b) {
        return a > b;
    }
    public static boolean compareGt(String a, String b) {
        return a.compareTo(b) > 0;
    }

    public static boolean compareNe(int a, int b) {
        return a != b;
    }
    public static boolean compareNe(double a, double b) {
        return a != b;
    }
    public static boolean compareNe(String a, String b) {
        return !a.equals(b);
    }
    public static boolean compareNe(boolean a, boolean b) {
        return a != b;
    }

    public static boolean compareEq(int a, int b) {
        return a == b;
    }
    public static boolean compareEq(double a, double b) {
        return a == b;
    }
    public static boolean compareEq(String a, String b) {
        return a.equals(b);
    }
    public static boolean compareEq(boolean a, boolean b) {
        return a == b;
    }

    public static void main(String[] args) {
        Scanner _scanner = new Scanner(System.in);
"#,
        );

        for code in acc {
            output.push_str(&code.0);
            output.push('\n');
        }

        output.push_str("    }\n");
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
        depth: u32,
    ) -> Self {
        let indentation = Self::gen_indentation(depth);
        let t = match t {
            Types::Int => "long",
            Types::Float => "double",
            Types::String => "String",
            Types::Bool => "boolean",
        };
        Self(format!(
            "{indentation}{t} {ident} = {};",
            Self::gen_expr(expr, pratt)
        ))
    }

    fn assign_var(
        ident: &'a str,
        _: Types,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self {
        let indentation = Self::gen_indentation(depth);
        Self(format!(
            "{indentation}{ident} = {};",
            Self::gen_expr(expr, pratt)
        ))
    }

    fn change_assign_var(
        ident: &'a str,
        _: Types,
        op: Pair<'a, Rule>,
        expr: Pair<'a, Rule>,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self {
        let op = match op.as_rule() {
            Rule::sum => "+",
            Rule::sub => "-",
            Rule::mul => "*",
            Rule::div => "/",
            _ => unreachable!(),
        };

        let indentation = Self::gen_indentation(depth);
        Self(format!(
            "{indentation}{ident} {op}= {};",
            Self::gen_expr(expr, pratt)
        ))
    }

    fn cmd_if(
        expr: Pair<'a, Rule>,
        true_branch: Self,
        false_branch: Option<Self>,
        pratt: &PrattParser<Rule>,
        depth: u32,
    ) -> Self {
        let indentation = Self::gen_indentation(depth);
        let output = format!(
            "{indentation}if ({}) {{\n{}{indentation}}}",
            Self::gen_expr(expr, pratt),
            true_branch.0
        );
        match false_branch {
            Some(false_branch) => Self(format!(
                "{output} else {{\n{}{indentation}}}",
                false_branch.0
            )),
            None => Self(output),
        }
    }

    fn cmd_for(
        expr: Pair<'a, Rule>,
        mut change_assign: Self,
        block: Self,
        pratt: &PrattParser<Rule>,
        depth: u32,
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
            Rule::string_raw => Self(format!(
                "{indentation}System.out.println(\"{}\");",
                content.as_str()
            )),
            Rule::int_val | Rule::float_val | Rule::r#true | Rule::r#false | Rule::ident => Self(
                format!("{indentation}System.out.println({});", content.as_str()),
            ),
            _ => unreachable!(),
        }
    }

    fn cmd_read(ident: &'a str, t: Types, depth: u32) -> Self {
        let indentation = Self::gen_indentation(depth);
        let r = match t {
            Types::Int => "Long",
            Types::Float => "Double",
            Types::String => "Line",
            Types::Bool => "Boolean",
        };
        // Self(format!("{indentation}std::cin >> {ident};"))
        Self(format!("{indentation}{ident} = _scanner.next{r}();"))
    }
}
