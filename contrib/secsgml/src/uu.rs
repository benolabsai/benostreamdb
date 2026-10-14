//! UUEncoding and decoding routines ported from secsgml2.
//!
//! Provides line-oriented UU body detection, wrapper stripping,
//! and high-throughput UU-decoding.

use memchr::memchr;

/// Maximum decoded size limit to protect against out-of-memory attacks (256 MB).
pub const UU_DECODE_MAX: usize = 256 * 1024 * 1024;

/// Finds the body bounds of a uuencoded stream (between `begin 644 ...\n` and `end\n`).
///
/// Returns `Some((body_start, end_line_start))` if detected, otherwise `None`.
pub fn find_uu_body_bounds(data: &[u8]) -> Option<(usize, usize)> {
    let n = data.len();
    let mut i = 0;
    let mut lines_checked = 0;
    let mut begin_body_start = None;

    while i < n && lines_checked < 3 {
        let line_start = i;
        let line_end = match memchr(b'\n', &data[i..]) {
            Some(pos) => {
                let end = i + pos;
                if end > line_start && data[end - 1] == b'\r' {
                    end - 1
                } else {
                    end
                }
            }
            None => n,
        };

        let line = &data[line_start..line_end];
        if line.len() >= 9 && &line[..9] == b"begin 644" {
            if line.len() == 9 || line[9] == b' ' || line[9] == b'\t' {
                // Advance past newline
                let mut next_line = line_end;
                if next_line < n && data[next_line] == b'\r' {
                    next_line += 1;
                }
                if next_line < n && data[next_line] == b'\n' {
                    next_line += 1;
                }
                begin_body_start = Some(next_line);
                i = next_line;
                break;
            }
        }

        // Advance to next line
        i = match memchr(b'\n', &data[i..]) {
            Some(pos) => i + pos + 1,
            None => n,
        };
        lines_checked += 1;
    }

    let start = begin_body_start?;

    // Now find "end" line
    let mut end_line_start = None;
    while i < n {
        let line_start = i;
        let line_end = match memchr(b'\n', &data[i..]) {
            Some(pos) => {
                let end = i + pos;
                if end > line_start && data[end - 1] == b'\r' {
                    end - 1
                } else {
                    end
                }
            }
            None => n,
        };

        let line = &data[line_start..line_end];
        if line.len() >= 3 && &line[..3] == b"end" {
            if line.len() == 3 || line[3] == b' ' || line[3] == b'\t' {
                end_line_start = Some(line_start);
                break;
            }
        }

        i = match memchr(b'\n', &data[i..]) {
            Some(pos) => i + pos + 1,
            None => n,
        };
    }

    let end = end_line_start?;
    Some((start, end))
}

/// Checks if data begins with a uuencoded block.
#[inline]
pub fn is_uuencoded(data: &[u8]) -> bool {
    find_uu_body_bounds(data).is_some()
}

/// Strips the `begin 644 ...` and `end` wrappers from a uuencoded block.
pub fn strip_uu_wrappers(data: &[u8]) -> &[u8] {
    if let Some((start, end)) = find_uu_body_bounds(data) {
        &data[start..end]
    } else {
        data
    }
}

/// Decodes uuencoded bytes into raw binary.
pub fn uudecode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 3 / 4);
    let mut i = 0;
    let n = data.len();

    while i < n {
        let len_char = data[i];

        // Skip blank lines / CRLF
        if len_char == b'\n' || len_char == b'\r' {
            i += 1;
            continue;
        }

        // Zero-length line signifies the end of uuencoded block (e.g. '`' or space)
        let nbytes = (len_char.wrapping_sub(32)) & 0x3f;
        if nbytes == 0 {
            break;
        }

        i += 1; // consume length char

        // Find end of line
        let line_start = i;
        let (line_end, next_i) = match memchr(b'\n', &data[i..]) {
            Some(pos) => {
                let mut end = i + pos;
                let next = end + 1;
                if end > line_start && data[end - 1] == b'\r' {
                    end -= 1;
                }
                (end, next)
            }
            None => (n, n),
        };

        i = next_i;
        let line = &data[line_start..line_end];
        let mut remaining = nbytes as usize;

        if out.len() + remaining > UU_DECODE_MAX {
            break;
        }

        // Fast path: standard full line (45 bytes encoded as 60 chars)
        if len_char == b'M' && line.len() >= 60 && remaining == 45 {
            let mut g = 0;
            while g < 15 {
                let base = g * 4;
                let da = line[base].wrapping_sub(32) & 0x3f;
                let db = line[base + 1].wrapping_sub(32) & 0x3f;
                let dc = line[base + 2].wrapping_sub(32) & 0x3f;
                let dd = line[base + 3].wrapping_sub(32) & 0x3f;

                out.push((da << 2) | (db >> 4));
                out.push((db << 4) | (dc >> 2));
                out.push((dc << 6) | dd);
                g += 1;
            }
            continue;
        }

        // Slow / general path: partial or short lines
        let mut p = 0;
        let mut leftchar: u32 = 0;
        let mut leftbits: i32 = 0;

        while remaining > 0 {
            let ch = if p < line.len() {
                let c = line[p].wrapping_sub(32) & 0x3f;
                p += 1;
                c as u32
            } else {
                0
            };

            leftchar = (leftchar << 6) | ch;
            leftbits += 6;

            if leftbits >= 8 {
                leftbits -= 8;
                out.push(((leftchar >> leftbits) & 0xff) as u8);
                leftchar &= (1 << leftbits) - 1;
                remaining -= 1;
            }
        }
    }

    out
}

/// Decodes uuencoded content, stripping `begin 644` and `end` if present.
pub fn decode_uuencoded_content(data: &[u8]) -> Vec<u8> {
    if let Some((start, end)) = find_uu_body_bounds(data) {
        uudecode(&data[start..end])
    } else {
        data.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uu_body_bounds_and_decode() {
        let enc = b"begin 644 test.bin\n#0V%T\n`\nend\n";
        assert!(is_uuencoded(enc));
        let stripped = strip_uu_wrappers(enc);
        assert_eq!(stripped, b"#0V%T\n`\n");
        let decoded = decode_uuencoded_content(enc);
        assert_eq!(decoded, b"Cat");
    }

    #[test]
    fn test_uu_fast_path_45_bytes() {
        // 'M' line produces 45 bytes
        let raw: Vec<u8> = (0..45).map(|x| (x * 7 + 13) as u8).collect();
        // encode it
        let mut encoded = vec![b'M'];
        for chunk in raw.chunks(3) {
            let b0 = chunk[0];
            let b1 = chunk[1];
            let b2 = chunk[2];
            let c0 = (b0 >> 2) + 32;
            let c1 = (((b0 & 0x3) << 4) | (b1 >> 4)) + 32;
            let c2 = (((b1 & 0xf) << 2) | (b2 >> 6)) + 32;
            let c3 = (b2 & 0x3f) + 32;
            encoded.extend_from_slice(&[c0, c1, c2, c3]);
        }
        encoded.push(b'\n');
        encoded.extend_from_slice(b"`\n");

        let mut full = b"begin 644 sample.bin\n".to_vec();
        full.extend_from_slice(&encoded);
        full.extend_from_slice(b"end\n");

        let decoded = decode_uuencoded_content(&full);
        assert_eq!(decoded, raw);
    }
}
