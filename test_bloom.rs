use bloomfilter::Bloom;
fn main() {
    let mut bloom = Bloom::new_for_fp_rate(100, 0.01);
    bloom.set(&"test".to_string());
    assert!(bloom.check(&"test".to_string()));
}
