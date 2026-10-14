//! File format extractors for BenoStreamDB MCP ingest_file tool.
//!
//! Handles:
//! - Office Word (.docx)
//! - Office PowerPoint (.pptx)
//! - Financial XBRL (.xbrl)
//! - XML (.xml)
//! - SEC EDGAR SGML (.sgml, .nc, SEC .txt)
//! - Zstandard decompression (.zst, .zstd)
//! - Zip archive extraction (.zip)

use anyhow::{Context, Result};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde_json::{json, Map, Value};
use std::io::{Cursor, Read};
use std::path::Path;

/// Decompresses Zstandard-compressed bytes.
pub fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::decode_all(Cursor::new(data)).context("Failed to decompress zstd stream")
}

/// Extracts paragraphs from a Word document (.docx).
pub fn extract_docx_records(data: &[u8], file_path: &str, chunk_size: usize) -> Result<Vec<Value>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(data)).context("Invalid docx zip archive")?;

    let mut doc_xml = String::new();
    {
        let mut entry = zip
            .by_name("word/document.xml")
            .context("docx missing word/document.xml")?;
        entry.read_to_string(&mut doc_xml)?;
    }

    let mut reader = Reader::from_str(&doc_xml);
    reader.trim_text(true);

    let mut records = Vec::new();
    let mut in_p = false;
    let mut in_t = false;
    let mut cur_para = String::new();
    let mut para_idx = 0;
    let mut cur_chunk = String::new();
    let mut chunk_idx = 0;
    let mut start_para = 1;

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = e.name();
                if name.as_ref() == b"w:p" {
                    in_p = true;
                    cur_para.clear();
                } else if name.as_ref() == b"w:t" {
                    in_t = true;
                }
            }
            Ok(Event::End(ref e)) => {
                let name = e.name();
                if name.as_ref() == b"w:p" {
                    in_p = false;
                    let trimmed = cur_para.trim();
                    if !trimmed.is_empty() {
                        para_idx += 1;
                        if cur_chunk.len() + trimmed.len() > chunk_size && !cur_chunk.is_empty() {
                            records.push(json!({
                                "path": file_path,
                                "chunk_index": chunk_idx,
                                "start_paragraph": start_para,
                                "end_paragraph": para_idx - 1,
                                "content": cur_chunk.clone(),
                                "format": "docx"
                            }));
                            chunk_idx += 1;
                            cur_chunk.clear();
                            start_para = para_idx;
                        }
                        if !cur_chunk.is_empty() {
                            cur_chunk.push_str("\n\n");
                        }
                        cur_chunk.push_str(trimmed);
                    }
                } else if name.as_ref() == b"w:t" {
                    in_t = false;
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_p && in_t {
                    if let Ok(text) = e.unescape() {
                        cur_para.push_str(&text);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!("XML parse error in docx: {}", e)),
            _ => {}
        }
        buf.clear();
    }

    if !cur_chunk.is_empty() {
        records.push(json!({
            "path": file_path,
            "chunk_index": chunk_idx,
            "start_paragraph": start_para,
            "end_paragraph": para_idx,
            "content": cur_chunk,
            "format": "docx"
        }));
    }

    Ok(records)
}

/// Extracts slides from a PowerPoint presentation (.pptx).
pub fn extract_pptx_records(data: &[u8], file_path: &str) -> Result<Vec<Value>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(data)).context("Invalid pptx zip archive")?;

    // Find and sort slide entries: ppt/slides/slide{N}.xml
    let mut slide_names: Vec<String> = (0..zip.len())
        .filter_map(|i| {
            let file = zip.by_index(i).ok()?;
            let name = file.name();
            if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect();

    slide_names.sort_by_key(|name| {
        let stem = name
            .strip_prefix("ppt/slides/slide")
            .and_then(|s| s.strip_suffix(".xml"))
            .unwrap_or("0");
        stem.parse::<u32>().unwrap_or(0)
    });

    let mut records = Vec::new();

    for (slide_idx, name) in slide_names.iter().enumerate() {
        let mut slide_xml = String::new();
        {
            let mut entry = zip.by_name(name)?;
            entry.read_to_string(&mut slide_xml)?;
        }

        let mut reader = Reader::from_str(&slide_xml);
        reader.trim_text(true);

        let mut slide_text = String::new();
        let mut in_t = false;
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name().as_ref() == b"a:t" {
                        in_t = true;
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"a:t" {
                        in_t = false;
                    } else if e.name().as_ref() == b"a:p" {
                        if !slide_text.ends_with('\n') {
                            slide_text.push('\n');
                        }
                    }
                }
                Ok(Event::Text(ref e)) => {
                    if in_t {
                        if let Ok(t) = e.unescape() {
                            slide_text.push_str(&t);
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }

        let trimmed = slide_text.trim();
        records.push(json!({
            "path": file_path,
            "slide_index": slide_idx + 1,
            "slide_name": name,
            "content": trimmed,
            "format": "pptx"
        }));
    }

    Ok(records)
}

/// Extracts financial facts from an XBRL document (.xbrl).
pub fn extract_xbrl_records(data: &[u8], file_path: &str) -> Result<Vec<Value>> {
    let xml_str = String::from_utf8_lossy(data);
    let mut reader = Reader::from_str(&xml_str);
    reader.trim_text(true);

    let mut records = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let tag_bytes = e.name().as_ref().to_vec();
                let tag_name = String::from_utf8_lossy(&tag_bytes).into_owned();

                // Check for XBRL fact attributes
                let mut context_ref = None;
                let mut unit_ref = None;
                let mut decimals = None;

                for attr in e.attributes().flatten() {
                    let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                    let val = String::from_utf8_lossy(&attr.value).into_owned();
                    match key.as_str() {
                        "contextRef" => context_ref = Some(val),
                        "unitRef" => unit_ref = Some(val),
                        "decimals" => decimals = Some(val),
                        _ => {}
                    }
                }

                if context_ref.is_some() || unit_ref.is_some() {
                    // Read text content
                    let text = reader.read_text(e.name())?;
                    let mut rec = json!({
                        "path": file_path,
                        "concept": tag_name,
                        "value": text.trim(),
                        "format": "xbrl"
                    });
                    if let Some(ctx) = context_ref {
                        rec["context_ref"] = json!(ctx);
                    }
                    if let Some(unit) = unit_ref {
                        rec["unit_ref"] = json!(unit);
                    }
                    if let Some(dec) = decimals {
                        rec["decimals"] = json!(dec);
                    }
                    records.push(rec);
                }
            }
            Ok(Event::Empty(ref e)) => {
                let tag_bytes = e.name().as_ref().to_vec();
                let tag_name = String::from_utf8_lossy(&tag_bytes).into_owned();

                let mut context_ref = None;
                let mut unit_ref = None;

                for attr in e.attributes().flatten() {
                    let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                    let val = String::from_utf8_lossy(&attr.value).into_owned();
                    if key == "contextRef" {
                        context_ref = Some(val);
                    } else if key == "unitRef" {
                        unit_ref = Some(val);
                    }
                }

                if context_ref.is_some() || unit_ref.is_some() {
                    let mut rec = json!({
                        "path": file_path,
                        "concept": tag_name,
                        "value": "",
                        "format": "xbrl"
                    });
                    if let Some(ctx) = context_ref {
                        rec["context_ref"] = json!(ctx);
                    }
                    if let Some(unit) = unit_ref {
                        rec["unit_ref"] = json!(unit);
                    }
                    records.push(rec);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!("XBRL XML parse error: {}", e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(records)
}

/// Extracts records from general XML documents (.xml).
pub fn extract_xml_records(data: &[u8], file_path: &str) -> Result<Vec<Value>> {
    let xml_str = String::from_utf8_lossy(data);
    let mut reader = Reader::from_str(&xml_str);
    reader.trim_text(true);

    let mut records = Vec::new();
    let mut buf = Vec::new();
    let mut path_stack: Vec<String> = Vec::new();
    let mut cur_record_depth: Option<usize> = None;
    let mut cur_record_map: Option<Map<String, Value>> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let tag_name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                path_stack.push(tag_name.clone());

                if path_stack.len() == 2 && cur_record_map.is_none() {
                    cur_record_depth = Some(2);
                    let mut map = Map::new();
                    map.insert("path".to_string(), json!(file_path));
                    map.insert("record_type".to_string(), json!(tag_name));
                    cur_record_map = Some(map);
                }
            }
            Ok(Event::End(ref e)) => {
                let tag_name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                if let Some(depth) = cur_record_depth {
                    if path_stack.len() == depth {
                        if let Some(map) = cur_record_map.take() {
                            records.push(Value::Object(map));
                        }
                        cur_record_depth = None;
                    }
                }
                if let Some(last) = path_stack.last() {
                    if last == &tag_name {
                        path_stack.pop();
                    }
                }
            }
            Ok(Event::Text(ref e)) => {
                if let Some(ref mut map) = cur_record_map {
                    if let Some(tag) = path_stack.last() {
                        if let Ok(text) = e.unescape() {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                map.insert(tag.clone(), json!(trimmed));
                            }
                        }
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!("XML parse error: {}", e)),
            _ => {}
        }
        buf.clear();
    }

    if records.is_empty() {
        // Fallback: single narrative XML record
        records.push(json!({
            "path": file_path,
            "content": xml_str.into_owned(),
            "format": "xml"
        }));
    }

    Ok(records)
}

/// Extracts records from SEC EDGAR SGML filings (.sgml, .nc, SEC .txt) using secsgml.
pub fn extract_secsgml_records(data: &[u8], file_path: &str) -> Result<Vec<Value>> {
    use base64::Engine;

    let (meta, docs) = secsgml::parse_sgml(data, None)
        .map_err(|e| anyhow::anyhow!("SEC SGML parse error: {}", e))?;

    let mut records = Vec::with_capacity(docs.len().max(1));

    let accession_number = meta
        .get("accession-number")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let filing_type = meta
        .get("type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let period = meta
        .get("period")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let filing_date = meta
        .get("filing-date")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let cik = meta
        .pointer("/filer/company-data/cik")
        .or_else(|| meta.get("cik"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    for doc in docs {
        let content_str = if doc.is_uuencoded {
            base64::prelude::BASE64_STANDARD.encode(&doc.content)
        } else {
            String::from_utf8_lossy(&doc.content).into_owned()
        };

        let mut rec = json!({
            "path": file_path,
            "doc_type": doc.meta.r#type,
            "sequence": doc.meta.sequence,
            "filename": doc.meta.filename,
            "description": doc.meta.description,
            "content": content_str,
            "is_binary": doc.is_uuencoded,
            "size_bytes": doc.content.len(),
            "format": "secsgml"
        });

        if let Some(ref acc) = accession_number {
            rec["accession_number"] = json!(acc);
        }
        if let Some(ref ft) = filing_type {
            rec["submission_type"] = json!(ft);
        }
        if let Some(ref p) = period {
            rec["period"] = json!(p);
        }
        if let Some(ref fd) = filing_date {
            rec["filing_date"] = json!(fd);
        }
        if let Some(ref c) = cik {
            rec["cik"] = json!(c);
        }

        records.push(rec);
    }

    Ok(records)
}

/// Extracts entries from a ZIP archive (.zip) and parses supported formats within.
pub fn extract_zip_records(data: &[u8], file_path: &str, chunk_size: usize) -> Result<Vec<Value>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(data)).context("Invalid zip archive")?;

    let mut records = Vec::new();

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if entry.is_dir() {
            continue;
        }

        let name = entry.name().to_string();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;

        let inner_ext = Path::new(&name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let inner_records = parse_file_bytes(&buf, &name, &inner_ext, chunk_size)?;
        for mut rec in inner_records {
            if let Some(obj) = rec.as_object_mut() {
                obj.insert("archive_path".to_string(), json!(file_path));
                obj.insert("entry_name".to_string(), json!(name));
            }
            records.push(rec);
        }
    }

    Ok(records)
}

/// Dispatches raw file bytes and extension to the appropriate extractor.
pub fn parse_file_bytes(
    raw_bytes: &[u8],
    file_path: &str,
    ext: &str,
    chunk_size: usize,
) -> Result<Vec<Value>> {
    // 1. Transparent Zstandard decompression (.zst, .zstd)
    if ext == "zst" || ext == "zstd" {
        let decompressed = decompress_zstd(raw_bytes)?;
        let path = Path::new(file_path);
        let inner_ext = path
            .file_stem()
            .and_then(|s| Path::new(s).extension())
            .and_then(|e| e.to_str())
            .unwrap_or("txt")
            .to_lowercase();
        return parse_file_bytes(&decompressed, file_path, &inner_ext, chunk_size);
    }

    // 2. Specific formats
    match ext {
        "docx" => extract_docx_records(raw_bytes, file_path, chunk_size),
        "pptx" => extract_pptx_records(raw_bytes, file_path),
        "xbrl" => extract_xbrl_records(raw_bytes, file_path),
        "xml" => extract_xml_records(raw_bytes, file_path),
        "sgml" | "nc" => extract_secsgml_records(raw_bytes, file_path),
        "zip" => extract_zip_records(raw_bytes, file_path, chunk_size),
        _ => {
            // Check if text file contains SEC SGML markers
            if ext == "txt" || ext.is_empty() {
                if raw_bytes
                    .windows(10)
                    .any(|w| w == b"<DOCUMENT>" || w == b"<SEC-DOCUM")
                {
                    return extract_secsgml_records(raw_bytes, file_path);
                }
            }

            // Fallback plain text / code chunking
            let content = String::from_utf8_lossy(raw_bytes);
            let lines: Vec<&str> = content.lines().collect();
            let mut cur_chunk = String::new();
            let mut start_line = 1;
            let mut chunk_idx = 0;
            let mut records = Vec::new();

            for (line_idx, line) in lines.iter().enumerate() {
                if cur_chunk.len() + line.len() > chunk_size && !cur_chunk.is_empty() {
                    records.push(json!({
                        "path": file_path,
                        "chunk_index": chunk_idx,
                        "start_line": start_line,
                        "end_line": line_idx,
                        "content": cur_chunk.clone(),
                        "format": ext
                    }));
                    chunk_idx += 1;
                    cur_chunk.clear();
                    start_line = line_idx + 1;
                }
                cur_chunk.push_str(line);
                cur_chunk.push('\n');
            }

            if !cur_chunk.is_empty() {
                records.push(json!({
                    "path": file_path,
                    "chunk_index": chunk_idx,
                    "start_line": start_line,
                    "end_line": lines.len(),
                    "content": cur_chunk,
                    "format": ext
                }));
            }

            Ok(records)
        }
    }
}
