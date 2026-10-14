//! SEC EDGAR submission metadata parsing and standardization.
//!
//! Handles header formats (privacy-enhanced, tab-delimited, and archive-style SGML)
//! and standardizes tag names and values (SIC extraction, Act extraction, key normalization).

use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionEventType {
    SectionStart,
    SectionEnd,
    KeyVal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionEvent {
    pub event_type: SubmissionEventType,
    pub key: String,
    pub value: String,
    pub depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegexKind {
    None,
    SecAct,
    Sic,
}

struct MapEntry {
    from: &'static str,
    to: &'static str,
    rx: RegexKind,
}

static MAPPING: &[MapEntry] = &[
    MapEntry {
        from: "paper",
        to: "paper",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "accession number",
        to: "accession-number",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "conformed submission type",
        to: "type",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "public document count",
        to: "public-document-count",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "public document_count",
        to: "public-document-count",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "conformed period of report",
        to: "period",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "filed as of date",
        to: "filing-date",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "date as of change",
        to: "date-of-filing-date-change",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "effectiveness date",
        to: "effectiveness-date",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "filer",
        to: "filer",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "company data",
        to: "company-data",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "company conformed name",
        to: "conformed-name",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "central index key",
        to: "cik",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "state of incorporation",
        to: "state-of-incorporation",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "fiscal year end",
        to: "fiscal-year-end",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "filing values",
        to: "filing-values",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "form type",
        to: "form-type",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "sec act",
        to: "act",
        rx: RegexKind::SecAct,
    },
    MapEntry {
        from: "sec file number",
        to: "file-number",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "film number",
        to: "film-number",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "business address",
        to: "business-address",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "street 1",
        to: "street1",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "city",
        to: "city",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "state",
        to: "state",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "zip",
        to: "zip",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "business phone",
        to: "phone",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "mail address",
        to: "mail-address",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "former company",
        to: "former-company",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "former conformed name",
        to: "former-conformed-name",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "date of name change",
        to: "date-changed",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "sros",
        to: "sros",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "subject company",
        to: "subject-company",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "standard industrial classification",
        to: "assigned-sic",
        rx: RegexKind::Sic,
    },
    MapEntry {
        from: "irs number",
        to: "irs-number",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "filed by",
        to: "filed-by",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "street 2",
        to: "street2",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "items",
        to: "items",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "group members",
        to: "group-members",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "organization name",
        to: "organization-name",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "recieved date",
        to: "recieved-date",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "action date",
        to: "action-date",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "non us state territory",
        to: "non-us-state-territory",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "address is a non us location",
        to: "address-is-a-non-us-location",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "ein",
        to: "ein",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "class-contract-ticker-symbol",
        to: "class-contract-ticker-symbol",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "class-contract-name",
        to: "class-contract-name",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "class-contract-id",
        to: "class-contract-id",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "sec-document",
        to: "sec-document",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "sec-header",
        to: "sec-header",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "acceptance-datetime",
        to: "acceptance-datetime",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "series-and-classes-contracts-data",
        to: "series-and-classes-contracts-data",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "existing-series-and-classes-contracts",
        to: "existing-series-and-classes-contracts",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "merger-series-and-classes-contracts",
        to: "merger-series-and-classes-contracts",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "new-series-and-classes-contracts",
        to: "new-series-and-classes-contracts",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "series",
        to: "series",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "owner-cik",
        to: "owner-cik",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "series-id",
        to: "series-id",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "series-name",
        to: "series-name",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "acquiring-data",
        to: "acquiring-data",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "target-data",
        to: "target-data",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "new-classes-contracts",
        to: "new-classes-contracts",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "new-series",
        to: "new-series",
        rx: RegexKind::None,
    },
    MapEntry {
        from: "relationship",
        to: "relationship",
        rx: RegexKind::None,
    },
];

fn extract_sec_act(val: &str) -> Option<String> {
    let bytes = val.as_bytes();
    if bytes.len() < 7 {
        return None;
    }
    for i in 0..bytes.len().saturating_sub(6) {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            let mut j = i + 4;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j + 2 < bytes.len() && &bytes[j..j + 3] == b"Act" {
                return Some(val[i + 2..i + 4].to_string());
            }
        }
    }
    None
}

fn extract_sic(val: &str) -> Option<String> {
    let bytes = val.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        if bytes[i] == b'[' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b']' && j > i + 1 {
                return Some(val[i + 1..j].to_string());
            }
        }
    }
    None
}

fn normalize_key_fallback(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut in_ws = false;
    for c in key.chars() {
        if c.is_ascii_whitespace() {
            if !in_ws {
                out.push('-');
                in_ws = true;
            }
        } else {
            out.push(c.to_ascii_lowercase());
            in_ws = false;
        }
    }
    out
}

/// Standardizes submission metadata events matching secsgml2.
pub fn standardize_submission_metadata(events: Vec<SubmissionEvent>) -> Vec<SubmissionEvent> {
    let mut out = Vec::with_capacity(events.len());

    for ev in events {
        let has_slash = ev.key.starts_with('/');
        let key_trimmed = if has_slash { &ev.key[1..] } else { &ev.key };
        let lower = key_trimmed.to_ascii_lowercase();

        let (mapped_key, regex_kind) = match MAPPING.iter().find(|m| m.from == lower) {
            Some(entry) => (entry.to.to_string(), entry.rx),
            None => (normalize_key_fallback(key_trimmed), RegexKind::None),
        };

        let final_key = if has_slash {
            format!("/{}", mapped_key)
        } else {
            mapped_key
        };

        let final_val = if ev.event_type == SubmissionEventType::KeyVal {
            match regex_kind {
                RegexKind::SecAct => extract_sec_act(&ev.value).unwrap_or(ev.value),
                RegexKind::Sic => extract_sic(&ev.value).unwrap_or(ev.value),
                RegexKind::None => ev.value,
            }
        } else {
            ev.value
        };

        out.push(SubmissionEvent {
            event_type: ev.event_type,
            key: final_key,
            value: final_val,
            depth: ev.depth,
        });
    }

    out
}

fn parse_tab_metadata(data: &str, events: &mut Vec<SubmissionEvent>) {
    let mut depth: usize = 0;

    for raw_line in data.lines() {
        if raw_line.is_empty() {
            continue;
        }

        let indent = raw_line.chars().take_while(|&c| c == '\t').count();
        let content = raw_line[indent..].trim();
        if content.is_empty() {
            continue;
        }

        while depth > indent {
            depth -= 1;
            events.push(SubmissionEvent {
                event_type: SubmissionEventType::SectionEnd,
                key: String::new(),
                value: String::new(),
                depth,
            });
        }

        if let Some(colon_pos) = content.find(':') {
            let key = content[..colon_pos].trim();
            let value = content[colon_pos + 1..].trim();
            if value.is_empty() {
                events.push(SubmissionEvent {
                    event_type: SubmissionEventType::SectionStart,
                    key: key.to_string(),
                    value: String::new(),
                    depth,
                });
                depth += 1;
            } else {
                events.push(SubmissionEvent {
                    event_type: SubmissionEventType::KeyVal,
                    key: key.to_string(),
                    value: value.to_string(),
                    depth,
                });
            }
        } else if content.starts_with('<') {
            if let Some(gt_pos) = content.find('>') {
                let tag = content[1..gt_pos].trim();
                let value = content[gt_pos + 1..].trim();
                if let Some(stripped_closing) = tag.strip_prefix('/') {
                    if depth > 0 {
                        depth -= 1;
                    }
                    events.push(SubmissionEvent {
                        event_type: SubmissionEventType::SectionEnd,
                        key: format!("/{}", stripped_closing),
                        value: String::new(),
                        depth,
                    });
                } else if value.is_empty() {
                    events.push(SubmissionEvent {
                        event_type: SubmissionEventType::SectionStart,
                        key: tag.to_string(),
                        value: String::new(),
                        depth,
                    });
                    depth += 1;
                } else {
                    events.push(SubmissionEvent {
                        event_type: SubmissionEventType::KeyVal,
                        key: tag.to_string(),
                        value: value.to_string(),
                        depth,
                    });
                }
            }
        }
    }

    while depth > 0 {
        depth -= 1;
        events.push(SubmissionEvent {
            event_type: SubmissionEventType::SectionEnd,
            key: String::new(),
            value: String::new(),
            depth,
        });
    }
}

fn parse_archive_metadata(data: &str, events: &mut Vec<SubmissionEvent>) {
    let mut depth: usize = 0;

    for raw_line in data.lines() {
        let line = raw_line.trim_start();
        if line.starts_with('<') {
            if let Some(gt_pos) = line.find('>') {
                let tag = line[1..gt_pos].trim();
                let value = line[gt_pos + 1..].trim();

                if tag.eq_ignore_ascii_case("SUBMISSION") {
                    // Skip SUBMISSION wrapper to mirror legacy output
                    continue;
                }

                if let Some(closing) = tag.strip_prefix('/') {
                    if depth > 0 {
                        depth -= 1;
                    }
                    events.push(SubmissionEvent {
                        event_type: SubmissionEventType::SectionEnd,
                        key: format!("/{}", closing),
                        value: String::new(),
                        depth,
                    });
                } else if value.is_empty() {
                    events.push(SubmissionEvent {
                        event_type: SubmissionEventType::SectionStart,
                        key: tag.to_string(),
                        value: String::new(),
                        depth,
                    });
                    depth += 1;
                } else {
                    events.push(SubmissionEvent {
                        event_type: SubmissionEventType::KeyVal,
                        key: tag.to_string(),
                        value: value.to_string(),
                        depth,
                    });
                }
            }
        }
    }

    while depth > 0 {
        depth -= 1;
        events.push(SubmissionEvent {
            event_type: SubmissionEventType::SectionEnd,
            key: String::new(),
            value: String::new(),
            depth,
        });
    }
}

/// Parses raw submission metadata before the first `<DOCUMENT>` tag.
pub fn parse_submission_metadata(data: &str) -> Vec<SubmissionEvent> {
    let mut events = Vec::new();
    let trimmed = data.trim();
    if trimmed.is_empty() {
        return events;
    }

    if trimmed.starts_with('-') {
        // Privacy-enhanced message
        let dbl_newline = if let Some(pos) = trimmed.find("\n\n") {
            pos + 2
        } else if let Some(pos) = trimmed.find("\r\n\r\n") {
            pos + 4
        } else {
            0
        };

        if dbl_newline > 0 && dbl_newline < trimmed.len() {
            let pem_val = &trimmed[..dbl_newline];
            events.push(SubmissionEvent {
                event_type: SubmissionEventType::KeyVal,
                key: "privacy-enhanced-message".to_string(),
                value: pem_val.to_string(),
                depth: 0,
            });
            let rem = trimmed[dbl_newline..].trim_start();
            parse_tab_metadata(rem, &mut events);
        } else {
            parse_tab_metadata(trimmed, &mut events);
        }
    } else if trimmed.to_ascii_uppercase().starts_with("<SE") {
        parse_tab_metadata(trimmed, &mut events);
    } else {
        parse_archive_metadata(trimmed, &mut events);
    }

    events
}

/// Recursively builds a JSON object hierarchy from standardized submission events.
pub fn build_metadata_tree(events: &[SubmissionEvent]) -> Map<String, Value> {
    build_object_inner(events, 0, events.len(), -1).0
}

fn find_section_end(events: &[SubmissionEvent], start_idx: usize, end_idx: usize) -> usize {
    let target_depth = events[start_idx].depth;
    for i in (start_idx + 1)..end_idx {
        if events[i].event_type == SubmissionEventType::SectionEnd
            && events[i].depth == target_depth
        {
            return i;
        }
    }
    end_idx
}

fn insert_value(map: &mut Map<String, Value>, key: String, val: Value) {
    match map.remove(&key) {
        None => {
            map.insert(key, val);
        }
        Some(Value::Array(mut arr)) => {
            arr.push(val);
            map.insert(key, Value::Array(arr));
        }
        Some(existing) => {
            map.insert(key, Value::Array(vec![existing, val]));
        }
    }
}

fn build_object_inner(
    events: &[SubmissionEvent],
    start_idx: usize,
    end_idx: usize,
    depth: i32,
) -> (Map<String, Value>, usize) {
    let mut result = Map::new();
    let mut i = start_idx;

    while i < end_idx {
        if events[i].depth as i32 != depth + 1 {
            i += 1;
            continue;
        }

        match events[i].event_type {
            SubmissionEventType::KeyVal => {
                insert_value(
                    &mut result,
                    events[i].key.clone(),
                    Value::String(events[i].value.clone()),
                );
                i += 1;
            }
            SubmissionEventType::SectionStart => {
                let end = find_section_end(events, i, end_idx);
                let (sub_obj, _) = build_object_inner(events, i + 1, end, depth + 1);
                insert_value(&mut result, events[i].key.clone(), Value::Object(sub_obj));
                i = end + 1;
            }
            SubmissionEventType::SectionEnd => {
                i += 1;
            }
        }
    }

    (result, i)
}
