use regex::Regex;

fn rewrite_sql_string(query: &str) -> String {
    let mut q = query.to_string();
    let re_cast = Regex::new(r"::vector(?:\(\d+\))?").unwrap();
    q = re_cast.replace_all(&q, "").to_string();
    
    let operators = [
        ("<->", "dist_l2"),
        ("<=>", "dist_cosine"),
        ("<#>", "dist_ip"),
        ("<+>", "dist_l1"),
        ("<~>", "dist_hamming"),
        ("<%>", "dist_jaccard"),
    ];
    
    for (op, func) in operators {
        let pattern = format!(r"(?P<lhs>[a-zA-Z0-9_.]+(?:\([^)]*\))?)\s*{}\s*(?P<rhs>'[^']+'|ARRAY\[[^\]]+\]|[a-zA-Z0-9_.]+)", regex::escape(op));
        if let Ok(re) = Regex::new(&pattern) {
            let replacement = format!("{}(${{lhs}}, ${{rhs}})", func);
            q = re.replace_all(&q, replacement.as_str()).to_string();
        }
    }
    q
}

fn main() {
    let q1 = "SELECT id, embedding <-> '[0.0, 1.0, 0.0]'::vector AS dist FROM news";
    println!("{}", rewrite_sql_string(q1));
}
