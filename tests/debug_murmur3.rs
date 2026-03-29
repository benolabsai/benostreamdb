// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::convert::TryInto;

fn murmur3_32_x86_trace(data: &[u8], seed: u32) -> u32 {
    let c1 = 0xcc9e2d51;
    let c2 = 0x1b873593;
    let r1 = 15;
    let r2 = 13;
    let m = 5;
    let n = 0xe6546b64;

    let mut hash = seed;
    let len = data.len();
    let nblocks = len / 4;

    println!("Seed: {}, Len: {}", seed, len);

    // Body
    for i in 0..nblocks {
        let idx = i * 4;
        let k_bytes = &data[idx..idx+4];
        let mut k = u32::from_le_bytes(k_bytes.try_into().unwrap());
        println!("Block {}: {:?} -> {:x}", i, k_bytes, k);

        k = k.wrapping_mul(c1);
        k = k.rotate_left(r1);
        k = k.wrapping_mul(c2);
        
        println!("  k transformed: {:x}", k);

        hash ^= k;
        hash = hash.rotate_left(r2);
        hash = hash.wrapping_mul(m).wrapping_add(n);
        
        println!("  hash after block {}: {:x} ({})", i, hash, hash);
    }

    // Tail
    let idx = nblocks * 4;
    let tail = &data[idx..];
    let mut k1 = 0;
    println!("Tail: {:?}", tail);
    
    match tail.len() {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(r1);
            k1 = k1.wrapping_mul(c2);
            hash ^= k1;
            println!("  k1 (3 bytes): {:x}, hash: {:x}", k1, hash);
        },
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(r1);
            k1 = k1.wrapping_mul(c2);
            hash ^= k1;
            println!("  k1 (2 bytes): {:x}, hash: {:x}", k1, hash);
        },
        1 => {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(r1);
            k1 = k1.wrapping_mul(c2);
            hash ^= k1;
            println!("  k1 (1 byte): {:x}, hash: {:x}", k1, hash);
        },
        _ => {}
    }

    // Finalization
    hash ^= len as u32;
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85ebca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2ae35);
    hash ^= hash >> 16;
    
    println!("Final hash: {} ({:x})", hash, hash);
    hash
}

fn main() {
    let res_test = murmur3_32_x86_trace(b"test", 0);
    println!("Res test: {}", res_test);
    
    let res_null = murmur3_32_x86_trace(b"iceberg\0", 0);
    println!("Res iceberg\\0: {}", res_null);

    // assert_eq!(res, 1216061395);
}
