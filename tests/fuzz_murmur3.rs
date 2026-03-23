use std::convert::TryInto;

fn murmur3_msg(data: &[u8], seed: u32, big_endian: bool) -> u32 {
    let c1 = 0xcc9e2d51;
    let c2 = 0x1b873593;
    let r1 = 15;
    let r2 = 13;
    let m = 5;
    let n = 0xe6546b64;

    let mut hash = seed;
    let len = data.len();
    let nblocks = len / 4;

    for i in 0..nblocks {
        let idx = i * 4;
        let k_bytes = &data[idx..idx+4];
        let mut k = if big_endian {
             u32::from_be_bytes(k_bytes.try_into().unwrap())
        } else {
             u32::from_le_bytes(k_bytes.try_into().unwrap())
        };

        k = k.wrapping_mul(c1);
        k = k.rotate_left(r1);
        k = k.wrapping_mul(c2);

        hash ^= k;
        hash = hash.rotate_left(r2);
        hash = hash.wrapping_mul(m).wrapping_add(n);
    }

    let idx = nblocks * 4;
    let tail = &data[idx..];
    let mut k1 = 0;
    
    match tail.len() {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(r1);
            k1 = k1.wrapping_mul(c2);
            hash ^= k1;
        },
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(r1);
            k1 = k1.wrapping_mul(c2);
            hash ^= k1;
        },
        1 => {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(r1);
            k1 = k1.wrapping_mul(c2);
            hash ^= k1;
        },
        _ => {}
    }

    hash ^= len as u32;
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85ebca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2ae35);
    hash ^= hash >> 16;
    
    hash
}

fn main() {
    let target = 1216061395;
    
    // Fuzz Seeds
    for s in 0..1000 {
        if murmur3_msg(b"iceberg", s, false) == target {
            println!("FOUND MATCH! Seed: {}, LE: true", s);
            return;
        }
    }
    
    // Fuzz BE
    if murmur3_msg(b"iceberg", 0, true) == target {
         println!("FOUND MATCH! Seed: 0, LE: false");
    }
    
    println!("No match found in basic fuzz.");
}
