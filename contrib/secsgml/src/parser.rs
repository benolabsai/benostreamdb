//! SEC SGML document parser ported from secsgml2.
//!
//! Provides high-throughput parsing of multi-document SGML filings,
//! automatic UU-decode of binary attachments, wrapper stripping,
//! and metadata integration.

use memchr::memchr;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::SecSgmlError;
use crate::metadata::{
    build_metadata_tree, parse_submission_metadata, standardize_submission_metadata,
};
use crate::uu::{find_uu_body_bounds, uudecode};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentMeta {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub sequence: String,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secsgml_size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secsgml_start_byte: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secsgml_end_byte: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub meta: DocumentMeta,
    pub content: Vec<u8>,
    pub is_uuencoded: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum ScanState {
    Between,
    InDocMeta,
    InText,
}

#[inline]
fn find_eol(data: &[u8]) -> usize {
    match memchr(b'\n', data) {
        Some(pos) => {
            if pos > 0 && data[pos - 1] == b'\r' {
                pos - 1
            } else {
                pos
            }
        }
        None => data.len(),
    }
}

#[inline]
fn value_to_eol_str(data: &[u8]) -> String {
    let eol = find_eol(data);
    let slice = &data[..eol];
    let trimmed = trim_ascii(slice);
    String::from_utf8_lossy(trimmed).into_owned()
}

#[inline]
fn trim_ascii(mut s: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = s.split_first() {
        if first.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    while let Some((&last, rest)) = s.split_last() {
        if last.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    s
}

fn strip_wrappers(mut data: &[u8]) -> &[u8] {
    data = trim_ascii(data);

    let wrappers: &[(&[u8], &[u8])] = &[
        (b"<PDF>", b"</PDF>"),
        (b"<XBRL>", b"</XBRL>"),
        (b"<XML>", b"</XML>"),
    ];

    let mut changed = true;
    while changed {
        changed = false;
        for &(open, close) in wrappers {
            if data.len() >= open.len() + close.len()
                && data.starts_with(open)
                && data.ends_with(close)
            {
                data = trim_ascii(&data[open.len()..data.len() - close.len()]);
                changed = true;
            }
        }
    }

    data
}

/// Parses an SEC SGML filing into structured metadata and extracted `Document` objects.
pub fn parse_sgml(
    buf: &[u8],
    filter_document_types: Option<&[&str]>,
) -> Result<(Value, Vec<Document>), SecSgmlError> {
    let n = buf.len();

    // 1. Find submission metadata header (everything prior to the first <DOCUMENT>)
    let doc_open_needle = b"<DOCUMENT>";
    let header_end = buf
        .windows(doc_open_needle.len())
        .position(|window| window == doc_open_needle)
        .unwrap_or(n);

    let header_str = String::from_utf8_lossy(&buf[..header_end]);
    let raw_events = parse_submission_metadata(&header_str);
    let std_events = standardize_submission_metadata(raw_events);
    let mut meta_map = build_metadata_tree(&std_events);

    // 2. Scan documents using single-pass '<' search
    let mut docs: Vec<Document> = Vec::with_capacity(32);
    let mut state = ScanState::Between;

    let mut cur_meta = DocumentMeta::default();
    let mut text_start = 0;
    let mut cur_content = Vec::new();
    let mut is_uu = false;

    let mut p = 0;
    while p < n {
        let lt_offset = match memchr(b'<', &buf[p..]) {
            Some(pos) => pos,
            None => break,
        };
        let lt = p + lt_offset;
        let remain = n - lt;

        if remain < 2 {
            break;
        }

        let c1 = buf[lt + 1];

        if c1 == b'D' {
            if remain >= 10 && &buf[lt + 1..lt + 10] == b"DOCUMENT>" {
                if state == ScanState::Between {
                    cur_meta = DocumentMeta::default();
                    cur_content.clear();
                    is_uu = false;
                    state = ScanState::InDocMeta;
                }
                p = lt + 10;
            } else if remain >= 13 && &buf[lt + 1..lt + 13] == b"DESCRIPTION>" {
                if state == ScanState::InDocMeta {
                    cur_meta.description = value_to_eol_str(&buf[lt + 13..]);
                }
                p = lt + 13;
            } else {
                p = lt + 1;
            }
        } else if c1 == b'/' {
            if remain < 3 {
                p = lt + 1;
                continue;
            }
            let c2 = buf[lt + 2];
            if c2 == b'D' && remain >= 11 && &buf[lt + 1..lt + 11] == b"/DOCUMENT>" {
                if state == ScanState::InDocMeta || state == ScanState::InText {
                    docs.push(Document {
                        meta: cur_meta.clone(),
                        content: std::mem::take(&mut cur_content),
                        is_uuencoded: is_uu,
                    });
                    cur_meta = DocumentMeta::default();
                    is_uu = false;
                    state = ScanState::Between;
                    p = lt + 11;
                    continue;
                }
                p = lt + 11;
            } else if c2 == b'T' && remain >= 7 && &buf[lt + 1..lt + 7] == b"/TEXT>" {
                if state == ScanState::InText {
                    let text_end = lt;
                    let payload = &buf[text_start..text_end];

                    if let Some((start, end)) = find_uu_body_bounds(payload) {
                        is_uu = true;
                        cur_content = uudecode(&payload[start..end]);
                    } else {
                        is_uu = false;
                        cur_content = strip_wrappers(payload).to_vec();
                    }

                    state = ScanState::InDocMeta;
                    p = lt + 7;
                    continue;
                }
                p = lt + 7;
            } else {
                p = lt + 1;
            }
        } else if c1 == b'T' {
            if remain >= 6 && &buf[lt + 1..lt + 6] == b"TEXT>" {
                if state == ScanState::InDocMeta {
                    text_start = lt + 6;
                    state = ScanState::InText;
                }
                p = lt + 6;
            } else if remain >= 6 && &buf[lt + 1..lt + 6] == b"TYPE>" {
                if state == ScanState::InDocMeta {
                    cur_meta.r#type = value_to_eol_str(&buf[lt + 6..]);
                }
                p = lt + 6;
            } else {
                p = lt + 1;
            }
        } else if c1 == b'S' && remain >= 10 && &buf[lt + 1..lt + 10] == b"SEQUENCE>" {
            if state == ScanState::InDocMeta {
                cur_meta.sequence = value_to_eol_str(&buf[lt + 10..]);
            }
            p = lt + 10;
        } else if c1 == b'F' && remain >= 10 && &buf[lt + 1..lt + 10] == b"FILENAME>" {
            if state == ScanState::InDocMeta {
                cur_meta.filename = value_to_eol_str(&buf[lt + 10..]);
            }
            p = lt + 10;
        } else {
            p = lt + 1;
        }
    }

    // 3. Optional filter by document type
    if let Some(types) = filter_document_types {
        docs.retain(|d| types.iter().any(|t| t.eq_ignore_ascii_case(&d.meta.r#type)));
    }

    // 4. Attach documents metadata array to submission metadata
    let mut docs_meta_val = Vec::with_capacity(docs.len());
    for d in &mut docs {
        d.meta.secsgml_size_bytes = Some(d.content.len());
        let meta_obj = serde_json::to_value(&d.meta)?;
        docs_meta_val.push(meta_obj);
    }

    meta_map.insert("documents".to_string(), Value::Array(docs_meta_val));

    Ok((Value::Object(meta_map), docs))
}

/// Parses SGML filing into `(metadata_json, documents_bytes)` matching `secsgml2.parse_sgml_content_into_memory`.
pub fn parse_sgml_content_into_memory(
    buf: &[u8],
    filter_document_types: Option<&[&str]>,
) -> Result<(Value, Vec<Vec<u8>>), SecSgmlError> {
    let (meta, docs) = parse_sgml(buf, filter_document_types)?;
    let doc_bytes = docs.into_iter().map(|d| d.content).collect();
    Ok((meta, doc_bytes))
}
