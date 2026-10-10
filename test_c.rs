fn main() {
    let q_min = -350.0f32;
    let q_max = 350.0f32;
    let quantizer_min = -8.2f32;
    let quantizer_max = 8.2f32;
    
    let mut c = f32::MAX;
    if q_min < -1e-6 && quantizer_min <= 0.0 {
        c = c.min(quantizer_min / q_min);
    }
    if q_max > 1e-6 && quantizer_max >= 0.0 {
        c = c.min(quantizer_max / q_max);
    }
    println!("c = {}", c);
}
