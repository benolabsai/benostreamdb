use arrow::array::LargeStringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bzip2::read::BzDecoder;
use clap::Parser;
use parquet::arrow::AsyncArrowWriter;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use regex::Regex;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tokio::fs::File as AsyncFile;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input_dir: String,

    #[arg(short, long)]
    output_dir: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut nodes_titles = Vec::new();
    let mut nodes_ids = Vec::new();
    let mut nodes_texts = Vec::new();

    let mut edges_sources = Vec::new();
    let mut edges_targets = Vec::new();

    let link_regex = Regex::new(r"\[\[(.*?)\]\]").unwrap();

    let mut paths: Vec<_> = std::fs::read_dir(&args.input_dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("bz2"))
        .collect();

    paths.sort_by_key(|e| e.path());

    // Streaming writers: the full-site dump (66M pages / 500M edges) must NOT
    // be accumulated in RAM — buffers are flushed to parquet row groups and
    // freed periodically. Each page appears in exactly one chunk, so no
    // cross-flush deduplication is required.
    std::fs::create_dir_all(&args.output_dir)?;

    let nodes_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::LargeUtf8, false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("summary", DataType::LargeUtf8, false),
    ]));
    let edges_schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::LargeUtf8, false),
        Field::new("target", DataType::LargeUtf8, false),
    ]));

    let nodes_file = AsyncFile::create(format!("{}/nodes.parquet", args.output_dir)).await?;
    let mut nodes_writer = AsyncArrowWriter::try_new(nodes_file, nodes_schema.clone(), None)?;
    let edges_file = AsyncFile::create(format!("{}/edges.parquet", args.output_dir)).await?;
    let mut edges_writer = AsyncArrowWriter::try_new(edges_file, edges_schema.clone(), None)?;

    const FLUSH_PAGES: usize = 1_000_000;
    const FLUSH_EDGES: usize = 8_000_000;

    let mut total_pages: u64 = 0;
    let mut total_edges: u64 = 0;

    macro_rules! flush_nodes {
        () => {
            if !nodes_ids.is_empty() {
                let ids = std::mem::take(&mut nodes_ids);
                let titles = std::mem::take(&mut nodes_titles);
                let texts = std::mem::take(&mut nodes_texts);
                let batch = RecordBatch::try_new(
                    nodes_schema.clone(),
                    vec![
                        Arc::new(LargeStringArray::from(ids)),
                        Arc::new(LargeStringArray::from(titles)),
                        Arc::new(LargeStringArray::from(texts)),
                    ],
                )?;
                nodes_writer.write(&batch).await?;
            }
        };
    }

    macro_rules! flush_edges {
        () => {
            if !edges_sources.is_empty() {
                let srcs = std::mem::take(&mut edges_sources);
                let dsts = std::mem::take(&mut edges_targets);
                let batch = RecordBatch::try_new(
                    edges_schema.clone(),
                    vec![
                        Arc::new(LargeStringArray::from(srcs)),
                        Arc::new(LargeStringArray::from(dsts)),
                    ],
                )?;
                edges_writer.write(&batch).await?;
            }
        };
    }

    for entry in paths {
        let path = entry.path();
        println!("Parsing {:?}", path);

        let file = File::open(&path)?;
        let bz2 = BzDecoder::new(file);
        let buf_reader = BufReader::new(bz2);

        let mut reader = Reader::from_reader(buf_reader);
        reader.trim_text(true);

        let mut buf = Vec::new();
        let mut inside_page = false;
        let mut current_tag = String::new();

        let mut current_title = String::new();
        let mut current_id = String::new();
        let mut current_text = String::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let name = e.name();
                    let name_str = String::from_utf8_lossy(name.as_ref()).into_owned();
                    if name_str == "page" {
                        inside_page = true;
                        current_title.clear();
                        current_id.clear();
                        current_text.clear();
                    }
                    current_tag = name_str;
                }
                Ok(Event::Text(e)) => {
                    if inside_page {
                        let text = e.unescape().unwrap_or_default().into_owned();
                        if current_tag == "title" {
                            current_title.push_str(&text);
                        } else if current_tag == "id" && current_id.is_empty() {
                            // There is an <id> for revision too, we only want the first one (page id)
                            current_id.push_str(&text);
                        } else if current_tag == "text" {
                            current_text.push_str(&text);
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    let name = e.name();
                    let name_str = String::from_utf8_lossy(name.as_ref()).into_owned();
                    if name_str == "page" {
                        inside_page = false;

                        // Process page
                        nodes_ids.push(current_id.clone());
                        nodes_titles.push(current_title.clone());

                        // Create summary (first 500 chars, safely handling utf-8)
                        let summary = if current_text.chars().count() > 500 {
                            let mut end_idx = 0;
                            for (i, _) in current_text.char_indices().take(500) {
                                end_idx = i;
                            }
                            // Add the length of the 500th char
                            if let Some(c) = current_text[end_idx..].chars().next() {
                                end_idx += c.len_utf8();
                            }
                            format!("{}...", &current_text[..end_idx])
                        } else {
                            current_text.clone()
                        };
                        nodes_texts.push(summary);

                        // Extract links
                        for cap in link_regex.captures_iter(&current_text) {
                            if let Some(link_match) = cap.get(1) {
                                let link_str = link_match.as_str();
                                // Handle [[Target|Display]]
                                let target = link_str.split('|').next().unwrap_or(link_str);
                                if !target.contains(':') {
                                    // ignore File:, Category: etc.
                                    edges_sources.push(current_title.clone());
                                    edges_targets.push(target.to_string());
                                    total_edges += 1;
                                }
                            }
                        }
                        total_pages += 1;

                        if nodes_titles.len() >= FLUSH_PAGES {
                            flush_nodes!();
                        }
                        if edges_sources.len() >= FLUSH_EDGES {
                            flush_edges!();
                        }

                        if nodes_titles.len() % 10000 == 0 {
                            println!("Processed {} pages...", nodes_titles.len());
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    eprintln!("Error at position {}: {:?}", reader.buffer_position(), e);
                    break;
                }
                _ => (),
            }
            buf.clear();
        } // end of inner loop over events
    } // end of for loop over files

    flush_nodes!();
    flush_edges!();

    nodes_writer.close().await?;
    edges_writer.close().await?;

    println!("Total pages processed: {total_pages}");
    println!("Total edges found: {total_edges}");
    println!(
        "Saved {}/nodes.parquet and {}/edges.parquet",
        args.output_dir, args.output_dir
    );

    Ok(())
}
