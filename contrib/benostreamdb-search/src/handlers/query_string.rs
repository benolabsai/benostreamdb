// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Lucene-compatible `query_string` and `simple_query_string` parser.
//!
//! Parses query strings (with boolean operators `AND`/`OR`/`NOT`, `+`/`-`,
//! field prefixes `field:value`, quoted phrases `"phrase"`, wildcards `*`,
//! numeric/text ranges `[from TO to]`, and parentheses) and translates them
//! into OpenSearch/ES JSON query ASTs (`bool`, `match`, `multi_match`,
//! `match_phrase`, `range`, `prefix`, `wildcard`).

use benostreamdb::BenoStreamError;
use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
pub struct QueryStringOptions {
    pub default_field: Option<String>,
    pub fields: Vec<String>,
    pub default_operator: String,
}

impl Default for QueryStringOptions {
    fn default() -> Self {
        Self {
            default_field: None,
            fields: Vec::new(),
            default_operator: "OR".to_string(),
        }
    }
}

impl QueryStringOptions {
    pub fn from_value(spec: &Value) -> Result<(String, Self), BenoStreamError> {
        let obj = spec
            .as_object()
            .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                reason: "query_string: expected an object".into(),
            })?;

        let query = obj
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                reason: "query_string: 'query' (string) is required".into(),
            })?
            .to_string();

        let default_field = obj
            .get("default_field")
            .and_then(Value::as_str)
            .map(String::from);

        let fields = obj
            .get("fields")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();

        let default_operator = obj
            .get("default_operator")
            .and_then(Value::as_str)
            .map(|s| s.to_ascii_uppercase())
            .unwrap_or_else(|| "OR".to_string());

        Ok((
            query,
            Self {
                default_field,
                fields,
                default_operator,
            },
        ))
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Token {
    LParen,
    RParen,
    And,
    Or,
    Not,
    Term {
        field: Option<String>,
        value: String,
        prefix: Option<char>,
    },
    Phrase {
        field: Option<String>,
        phrase: String,
        prefix: Option<char>,
    },
    Range {
        field: String,
        from: Option<String>,
        to: Option<String>,
        include_lower: bool,
        include_upper: bool,
        prefix: Option<char>,
    },
}

fn tokenize(input: &str, is_simple: bool) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c == '(' {
            tokens.push(Token::LParen);
            i += 1;
            continue;
        }
        if c == ')' {
            tokens.push(Token::RParen);
            i += 1;
            continue;
        }

        if is_simple && c == '|' {
            tokens.push(Token::Or);
            i += 1;
            continue;
        }

        let mut prefix = None;
        if (c == '+' || c == '-') && i + 1 < len && !chars[i + 1].is_whitespace() {
            prefix = Some(c);
            i += 1;
            if i < len && chars[i] == '(' {
                if prefix == Some('-') {
                    tokens.push(Token::Not);
                }
                tokens.push(Token::LParen);
                i += 1;
                continue;
            }
        }

        // Quoted phrase without field: "hello world"
        if chars[i] == '"' {
            i += 1;
            let mut phrase = String::new();
            while i < len && chars[i] != '"' {
                phrase.push(chars[i]);
                i += 1;
            }
            if i < len && chars[i] == '"' {
                i += 1;
            }
            tokens.push(Token::Phrase {
                field: None,
                phrase,
                prefix,
            });
            continue;
        }

        // Read continuous word, respecting brackets for ranges
        let mut in_bracket = false;
        let mut word = String::new();
        while i < len {
            let ch = chars[i];
            if !in_bracket && (ch.is_whitespace() || ch == '(' || ch == ')') {
                break;
            }
            if !in_bracket && is_simple && ch == '|' {
                break;
            }
            if ch == '[' || ch == '{' {
                in_bracket = true;
            } else if ch == ']' || ch == '}' {
                in_bracket = false;
            }
            word.push(ch);
            i += 1;
        }

        if word.is_empty() {
            continue;
        }

        if !is_simple && prefix.is_none() {
            if word == "AND" || word == "&&" {
                tokens.push(Token::And);
                continue;
            }
            if word == "OR" || word == "||" {
                tokens.push(Token::Or);
                continue;
            }
            if word == "NOT" || word == "!" {
                tokens.push(Token::Not);
                continue;
            }
        }

        // Check if word is field:value or field:"phrase" or field:[...]
        if let Some(colon_pos) = word.find(':') {
            let field_part = word[..colon_pos].to_string();
            let remainder = &word[colon_pos + 1..];

            // Quoted phrase after colon: field:"hello world"
            if remainder.starts_with('"') {
                let mut phrase = remainder.trim_start_matches('"').to_string();
                if phrase.ends_with('"') && phrase.len() > 1 {
                    phrase.pop();
                } else {
                    // Continue reading until closing quote
                    while i < len && chars[i] != '"' {
                        phrase.push(chars[i]);
                        i += 1;
                    }
                    if i < len && chars[i] == '"' {
                        i += 1;
                    }
                }
                tokens.push(Token::Phrase {
                    field: Some(field_part),
                    phrase,
                    prefix,
                });
                continue;
            }

            // Range after colon: field:[from TO to] or field:{from TO to}
            if (remainder.starts_with('[') || remainder.starts_with('{'))
                && (remainder.ends_with(']') || remainder.ends_with('}'))
            {
                let include_lower = remainder.starts_with('[');
                let include_upper = remainder.ends_with(']');
                let inner = &remainder[1..remainder.len().saturating_sub(1)];
                let (from, to) = parse_range_bounds(inner);
                tokens.push(Token::Range {
                    field: field_part,
                    from,
                    to,
                    include_lower,
                    include_upper,
                    prefix,
                });
                continue;
            }

            // Ordinary field:value
            tokens.push(Token::Term {
                field: Some(field_part),
                value: remainder.to_string(),
                prefix,
            });
            continue;
        }

        tokens.push(Token::Term {
            field: None,
            value: word,
            prefix,
        });
    }

    tokens
}

fn parse_range_bounds(content: &str) -> (Option<String>, Option<String>) {
    let parts: Vec<&str> = content.split(" TO ").collect();
    if parts.len() == 2 {
        let from = parts[0].trim();
        let to = parts[1].trim();
        let from_val = if from == "*" {
            None
        } else {
            Some(from.to_string())
        };
        let to_val = if to == "*" {
            None
        } else {
            Some(to.to_string())
        };
        (from_val, to_val)
    } else {
        (None, None)
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Leaf(Value),
    Not(Box<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    opts: &'a QueryStringOptions,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], opts: &'a QueryStringOptions) -> Self {
        Self {
            tokens,
            pos: 0,
            opts,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_token(&mut self) -> Option<&Token> {
        if self.pos < self.tokens.len() {
            let t = &self.tokens[self.pos];
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn parse_expression(&mut self) -> Option<Expr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Option<Expr> {
        let mut clauses = Vec::new();
        let first = self.parse_and()?;
        clauses.push(first);

        while let Some(Token::Or) = self.peek() {
            self.next_token();
            if let Some(next) = self.parse_and() {
                clauses.push(next);
            }
        }

        if clauses.len() == 1 {
            Some(clauses.remove(0))
        } else {
            Some(Expr::Or(clauses))
        }
    }

    fn parse_and(&mut self) -> Option<Expr> {
        let mut clauses = Vec::new();
        let first = self.parse_unary()?;
        clauses.push(first);

        while let Some(t) = self.peek() {
            if *t == Token::RParen || *t == Token::Or {
                break;
            }
            if *t == Token::And {
                self.next_token();
            } else if self.opts.default_operator != "AND" {
                // If default operator is OR, juxtaposition should be treated as OR
                break;
            }
            if let Some(next) = self.parse_unary() {
                clauses.push(next);
            } else {
                break;
            }
        }

        if clauses.len() == 1 {
            Some(clauses.remove(0))
        } else {
            Some(Expr::And(clauses))
        }
    }

    fn parse_unary(&mut self) -> Option<Expr> {
        if let Some(Token::Not) = self.peek() {
            self.next_token();
            return self.parse_primary().map(|p| Expr::Not(Box::new(p)));
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Option<Expr> {
        match self.next_token().cloned() {
            Some(Token::LParen) => {
                let inner = self.parse_expression();
                if let Some(Token::RParen) = self.peek() {
                    self.next_token();
                }
                inner
            }
            Some(Token::Term {
                field,
                value,
                prefix,
            }) => {
                let leaf = term_to_value(field.as_deref(), &value, self.opts);
                let expr = Expr::Leaf(leaf);
                Some(apply_prefix(expr, prefix))
            }
            Some(Token::Phrase {
                field,
                phrase,
                prefix,
            }) => {
                let leaf = phrase_to_value(field.as_deref(), &phrase, self.opts);
                let expr = Expr::Leaf(leaf);
                Some(apply_prefix(expr, prefix))
            }
            Some(Token::Range {
                field,
                from,
                to,
                include_lower,
                include_upper,
                prefix,
            }) => {
                let leaf = range_to_value(
                    &field,
                    from.as_deref(),
                    to.as_deref(),
                    include_lower,
                    include_upper,
                );
                let expr = Expr::Leaf(leaf);
                Some(apply_prefix(expr, prefix))
            }
            _ => None,
        }
    }
}

fn apply_prefix(expr: Expr, prefix: Option<char>) -> Expr {
    match prefix {
        Some('-') => Expr::Not(Box::new(expr)),
        _ => expr,
    }
}

fn term_to_value(field: Option<&str>, value: &str, opts: &QueryStringOptions) -> Value {
    if value.ends_with('*') && !value.contains('?') && value.matches('*').count() == 1 {
        let prefix = value.trim_end_matches('*');
        if let Some(f) = field {
            json!({ "prefix": { f: prefix } })
        } else if let Some(df) = &opts.default_field {
            json!({ "prefix": { df: prefix } })
        } else if !opts.fields.is_empty() {
            json!({ "prefix": { &opts.fields[0]: prefix } })
        } else {
            json!({ "prefix": { "content": prefix } })
        }
    } else if value.contains('*') || value.contains('?') {
        if let Some(f) = field {
            json!({ "wildcard": { f: value } })
        } else if let Some(df) = &opts.default_field {
            json!({ "wildcard": { df: value } })
        } else if !opts.fields.is_empty() {
            json!({ "wildcard": { &opts.fields[0]: value } })
        } else {
            json!({ "wildcard": { "content": value } })
        }
    } else if let Some(f) = field {
        json!({ "match": { f: value } })
    } else if opts.fields.len() > 1 {
        json!({
            "multi_match": {
                "query": value,
                "fields": opts.fields
            }
        })
    } else if opts.fields.len() == 1 {
        json!({ "match": { &opts.fields[0]: value } })
    } else if let Some(df) = &opts.default_field {
        json!({ "match": { df: value } })
    } else {
        json!({ "match": { "content": value } })
    }
}

fn phrase_to_value(field: Option<&str>, phrase: &str, opts: &QueryStringOptions) -> Value {
    if let Some(f) = field {
        json!({ "match_phrase": { f: phrase } })
    } else if opts.fields.len() > 1 {
        let should: Vec<Value> = opts
            .fields
            .iter()
            .map(|f| json!({ "match_phrase": { f: phrase } }))
            .collect();
        json!({ "bool": { "should": should } })
    } else if opts.fields.len() == 1 {
        json!({ "match_phrase": { &opts.fields[0]: phrase } })
    } else if let Some(df) = &opts.default_field {
        json!({ "match_phrase": { df: phrase } })
    } else {
        json!({ "match_phrase": { "content": phrase } })
    }
}

fn range_to_value(
    field: &str,
    from: Option<&str>,
    to: Option<&str>,
    include_lower: bool,
    include_upper: bool,
) -> Value {
    let mut r = Map::new();
    if let Some(f) = from {
        let key = if include_lower { "gte" } else { "gt" };
        r.insert(key.to_string(), parse_range_val(f));
    }
    if let Some(t) = to {
        let key = if include_upper { "lte" } else { "lt" };
        r.insert(key.to_string(), parse_range_val(t));
    }
    json!({ "range": { field: Value::Object(r) } })
}

fn parse_range_val(s: &str) -> Value {
    if let Ok(i) = s.parse::<i64>() {
        json!(i)
    } else if let Ok(f) = s.parse::<f64>() {
        json!(f)
    } else {
        json!(s)
    }
}

fn expr_to_value(expr: Expr) -> Value {
    match expr {
        Expr::Leaf(v) => v,
        Expr::Not(sub) => json!({
            "bool": {
                "must_not": [expr_to_value(*sub)]
            }
        }),
        Expr::Or(list) => {
            let mut should = Vec::new();
            let mut must_not = Vec::new();
            for item in list {
                match item {
                    Expr::Not(sub) => must_not.push(expr_to_value(*sub)),
                    other => should.push(expr_to_value(other)),
                }
            }
            if must_not.is_empty() {
                if should.len() == 1 {
                    should.remove(0)
                } else {
                    json!({
                        "bool": {
                            "should": should
                        }
                    })
                }
            } else {
                let mut bool_map = Map::new();
                if !should.is_empty() {
                    if should.len() == 1 {
                        bool_map.insert("must".to_string(), Value::Array(vec![should.remove(0)]));
                    } else {
                        bool_map.insert("should".to_string(), Value::Array(should));
                    }
                }
                bool_map.insert("must_not".to_string(), Value::Array(must_not));
                json!({ "bool": Value::Object(bool_map) })
            }
        }
        Expr::And(list) => {
            let mut must = Vec::new();
            let mut must_not = Vec::new();
            for item in list {
                match item {
                    Expr::Not(sub) => must_not.push(expr_to_value(*sub)),
                    other => must.push(expr_to_value(other)),
                }
            }
            let mut bool_map = Map::new();
            if !must.is_empty() {
                bool_map.insert("must".to_string(), Value::Array(must));
            }
            if !must_not.is_empty() {
                bool_map.insert("must_not".to_string(), Value::Array(must_not));
            }

            if bool_map.len() == 1 {
                if let Some(Value::Array(mut arr)) = bool_map.remove("must") {
                    if arr.len() == 1 {
                        if let Some(single) = arr.pop() {
                            return single;
                        }
                    }
                    bool_map.insert("must".to_string(), Value::Array(arr));
                }
            }
            json!({ "bool": Value::Object(bool_map) })
        }
    }
}

/// Parse a query string into an ES query DSL AST.
pub fn parse_query_string(
    input: &str,
    opts: &QueryStringOptions,
    is_simple: bool,
) -> Result<Value, BenoStreamError> {
    let tokens = tokenize(input, is_simple);
    if tokens.is_empty() {
        return Ok(json!({ "match_all": {} }));
    }

    let mut parser = Parser::new(&tokens, opts);
    let mut top_clauses = Vec::new();
    while let Some(expr) = parser.parse_expression() {
        top_clauses.push(expr);
    }

    if top_clauses.is_empty() {
        return Ok(json!({ "match_all": {} }));
    }

    let final_expr = if top_clauses.len() == 1 {
        top_clauses.remove(0)
    } else if opts.default_operator == "AND" {
        Expr::And(top_clauses)
    } else {
        Expr::Or(top_clauses)
    };

    Ok(expr_to_value(final_expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_term_and_phrase() {
        let opts = QueryStringOptions {
            default_field: Some("title".into()),
            ..Default::default()
        };
        let v = parse_query_string("rust", &opts, false).unwrap();
        assert_eq!(v, json!({"match": {"title": "rust"}}));

        let v2 = parse_query_string("\"hello world\"", &opts, false).unwrap();
        assert_eq!(v2, json!({"match_phrase": {"title": "hello world"}}));
    }

    #[test]
    fn test_field_prefix_and_operators() {
        let opts = QueryStringOptions::default();
        let v = parse_query_string("title:rust AND status:active", &opts, false).unwrap();
        assert_eq!(
            v,
            json!({
                "bool": {
                    "must": [
                        {"match": {"title": "rust"}},
                        {"match": {"status": "active"}}
                    ]
                }
            })
        );

        let v_or = parse_query_string("tag:novel OR tag:animation", &opts, false).unwrap();
        assert_eq!(
            v_or,
            json!({
                "bool": {
                    "should": [
                        {"match": {"tag": "novel"}},
                        {"match": {"tag": "animation"}}
                    ]
                }
            })
        );
    }

    #[test]
    fn test_not_and_prefix_modifiers() {
        let opts = QueryStringOptions {
            default_field: Some("content".into()),
            ..Default::default()
        };
        let v = parse_query_string("+apple -banana", &opts, false).unwrap();
        assert_eq!(
            v,
            json!({
                "bool": {
                    "must": [{"match": {"content": "apple"}}],
                    "must_not": [{"match": {"content": "banana"}}]
                }
            })
        );
    }

    #[test]
    fn test_range_and_wildcard() {
        let opts = QueryStringOptions::default();
        let v_range = parse_query_string("age:[10 TO 20]", &opts, false).unwrap();
        assert_eq!(
            v_range,
            json!({
                "range": {
                    "age": {
                        "gte": 10,
                        "lte": 20
                    }
                }
            })
        );

        let v_prefix = parse_query_string("name:ali*", &opts, false).unwrap();
        assert_eq!(v_prefix, json!({"prefix": {"name": "ali"}}));

        let v_wildcard = parse_query_string("code:a?c*", &opts, false).unwrap();
        assert_eq!(v_wildcard, json!({"wildcard": {"code": "a?c*"}}));
    }

    #[test]
    fn test_grouping_with_parens() {
        let opts = QueryStringOptions {
            default_field: Some("text".into()),
            ..Default::default()
        };
        let v = parse_query_string("(cat OR dog) AND animal", &opts, false).unwrap();
        assert_eq!(
            v,
            json!({
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"match": {"text": "cat"}},
                                    {"match": {"text": "dog"}}
                                ]
                            }
                        },
                        {"match": {"text": "animal"}}
                    ]
                }
            })
        );
    }
}
