fn main() {
    let q = 0.05_f32;
    for i in 0..1000 {
        let v = i as f32 / 1000.0;
        let dist = ((q - v) * (q - v)).sqrt();
        if (dist - 0.40899998).abs() < 1e-6 {
            println!("MATCH: v = {}", v);
        }
    }
}
