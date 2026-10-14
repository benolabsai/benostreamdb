//! File format extractors for BenoStreamDB MCP ingest_file tool.
//!
//! Handles general-purpose enterprise and document formats:
//! - Office Word (.docx)
//! - Office PowerPoint (.pptx)
//! - Structured XML (.xml)
//! - Zstandard decompression (.zst, .zstd)
//! - Zip archive extraction (.zip)
//! - Plain text / code chunking (.txt, .md, .rs, .py, etc.)
//! Domain-specific file formats are handled via pluggable `CustomFileExtractor`
//! functions configured on `McpServer`.

use anyhow::{Context, Result};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde_json::{json, Map, Value};
use std::io::{Cursor, Read};
use std::path::Path;

/// Function signature for domain-specific custom file extractors.
pub type CustomFileExtractor = fn(
    raw_bytes: &[u8],
    file_path: &str,
    ext: &str,
    chunk_size: usize,
) -> Option<Result<Vec<Value>>>;

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

/// Extracts text slides from a PowerPoint presentation (.pptx).
pub fn extract_pptx_records(data: &[u8], file_path: &str) -> Result<Vec<Value>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(data)).context("Invalid pptx zip archive")?;

    let mut slide_names = Vec::new();
    for i in 0..zip.len() {
        let entry = zip.by_index(i)?;
        let name = entry.name().to_string();
        if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
            slide_names.push(name);
        }
    }

    slide_names.sort_by(|a, b| {
        let extract_num = |s: &str| -> usize {
            s.trim_start_matches("ppt/slides/slide")
                .trim_end_matches(".xml")
                .parse::<usize>()
                .unwrap_or(0)
        };
        extract_num(a).cmp(&extract_num(b))
    });

    let mut records = Vec::new();

    for (slide_idx, name) in slide_names.iter().enumerate() {
        let mut slide_xml = String::new();
        {
            let mut entry = zip
                .by_name(name)
                .with_context(|| format!("Missing slide entry: {}", name))?;
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

/// Extracts entries from a ZIP archive (.zip) and parses supported formats within.
pub fn extract_zip_records(
    data: &[u8],
    file_path: &str,
    chunk_size: usize,
    custom_extractor: Option<CustomFileExtractor>,
) -> Result<Vec<Value>> {
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

        let inner_records =
            parse_file_bytes_with_custom(&buf, &name, &inner_ext, chunk_size, custom_extractor)?;
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
    parse_file_bytes_with_custom(raw_bytes, file_path, ext, chunk_size, None)
}

/// Extracts structured tabular records from delimited files (CSV, TSV, PSV).
/// Automatically handles comma, tab, and pipe delimiters with column type inference.
pub fn extract_delimited_records(data: &[u8], file_path: &str, ext: &str) -> Result<Vec<Value>> {
    let delimiter = match ext {
        "tsv" => b'\t',
        "psv" => b'|',
        _ => {
            // Sniff first non-empty line
            if let Some(first_line) = data.split(|&b| b == b'\n').find(|l| !l.is_empty()) {
                let pipes = first_line.iter().filter(|&&b| b == b'|').count();
                let tabs = first_line.iter().filter(|&&b| b == b'\t').count();
                let commas = first_line.iter().filter(|&&b| b == b',').count();
                if pipes > commas && pipes > tabs {
                    b'|'
                } else if tabs > commas && tabs > pipes {
                    b'\t'
                } else {
                    b','
                }
            } else {
                b','
            }
        }
    };

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .flexible(true)
        .double_quote(true)
        .escape(Some(b'\\'))
        .trim(csv::Trim::All)
        .from_reader(data);

    let headers = reader.headers()?.clone();
    let mut records = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut row_map = Map::new();
        row_map.insert("path".to_string(), json!(file_path));

        for (i, field) in record.iter().enumerate() {
            let col_name = headers
                .get(i)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("col_{}", i + 1));

            let trimmed = field.trim();
            // Handle single-quoted values and SQL/escape conventions: 'val', '', \'
            let normalized =
                if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
                    let inner = &trimmed[1..trimmed.len() - 1];
                    inner.replace("''", "'").replace("\\'", "'")
                } else {
                    trimmed.to_string()
                };

            let val = if normalized.is_empty() {
                Value::Null
            } else if normalized.eq_ignore_ascii_case("true") {
                json!(true)
            } else if normalized.eq_ignore_ascii_case("false") {
                json!(false)
            } else if (normalized.starts_with('0')
                && normalized.len() > 1
                && !normalized.starts_with("0."))
                || (normalized.starts_with("-0")
                    && normalized.len() > 2
                    && !normalized.starts_with("-0."))
            {
                json!(normalized)
            } else if let Ok(int_val) = normalized.parse::<i64>() {
                json!(int_val)
            } else if let Ok(float_val) = normalized.parse::<f64>() {
                json!(float_val)
            } else {
                json!(normalized)
            };
            row_map.insert(col_name, val);
        }
        records.push(Value::Object(row_map));
    }

    Ok(records)
}

/// Dispatches raw file bytes and extension with optional domain extractor support.
pub fn parse_file_bytes_with_custom(
    raw_bytes: &[u8],
    file_path: &str,
    ext: &str,
    chunk_size: usize,
    custom_extractor: Option<CustomFileExtractor>,
) -> Result<Vec<Value>> {
    // 1. Check custom domain extractor first
    if let Some(custom) = custom_extractor {
        if let Some(res) = custom(raw_bytes, file_path, ext, chunk_size) {
            return res;
        }
    }

    // 2. Transparent Zstandard decompression (.zst, .zstd)
    if ext == "zst" || ext == "zstd" {
        let decompressed = decompress_zstd(raw_bytes)?;
        let path = Path::new(file_path);
        let inner_ext = path
            .file_stem()
            .and_then(|s| Path::new(s).extension())
            .and_then(|e| e.to_str())
            .unwrap_or("txt")
            .to_lowercase();
        return parse_file_bytes_with_custom(
            &decompressed,
            file_path,
            &inner_ext,
            chunk_size,
            custom_extractor,
        );
    }

    // 3. General enterprise formats
    match ext {
        "csv" | "tsv" | "psv" => extract_delimited_records(raw_bytes, file_path, ext),
        "docx" => extract_docx_records(raw_bytes, file_path, chunk_size),
        "pptx" => extract_pptx_records(raw_bytes, file_path),
        "xml" => extract_xml_records(raw_bytes, file_path),
        "zip" => extract_zip_records(raw_bytes, file_path, chunk_size, custom_extractor),
        _ => {
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
