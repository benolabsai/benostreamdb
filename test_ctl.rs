use tikv_jemalloc_ctl;
fn main() {
    let _ = tikv_jemalloc_ctl::arena::decay::write(tikv_jemalloc_ctl::arena::ARENA_ALL);
}
