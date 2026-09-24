use once_cell::sync::Lazy;
use regex::Regex;

/// The `PARTITIONED BY (...)` pattern, compiled once.
///
/// A literal pattern cannot fail to compile; `None` exists only so this stays
/// total (the SQL is passed through unchanged) instead of unwrapping.
static PARTITIONED_BY_RE: Lazy<Option<Regex>> =
    Lazy::new(|| Regex::new(r"(?i)PARTITIONED\s+BY\s*\([^)]*\)").ok());

/// Strips the `PARTITIONED BY (...)` clause from a DDL statement
/// because DataFusion's logical planner currently does not support it for local tables.
pub fn strip_partitioned_by(sql: &str) -> String {
    // Matches "PARTITIONED BY ( col1, col2 )" ignoring case and whitespace
    match PARTITIONED_BY_RE.as_ref() {
        Some(re) => re.replace(sql, "").to_string(),
        None => sql.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_partitioned_by() {
        let sql = "create table test PARTITIONED BY (pt) as (select 1)";
        let rewritten = strip_partitioned_by(sql);
        assert_eq!(rewritten, "create table test  as (select 1)");
    }

    #[test]
    fn test_strip_partitioned_by_multiline() {
        let sql = "create table test \n  PARTITIONED BY (\n pt\n )\nas (select 1)";
        let rewritten = strip_partitioned_by(sql);
        assert_eq!(rewritten, "create table test \n  \nas (select 1)");
    }
}
