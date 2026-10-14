use secsgml::{calculate_documents_locations_in_tar, parse_sgml, parse_sgml_content_into_memory};

#[test]
fn test_tab_privacy_sample() {
    let path = "/tmp/secsgml_samples/tab-privacy.txt";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let data = std::fs::read(path).expect("read sample");
    let (mut meta, docs) = parse_sgml(&data, None).expect("parse sgml");

    assert_eq!(docs.len(), 4);
    assert_eq!(meta["accession-number"], "0001010412-10-000078");
    assert_eq!(meta["type"], "10-K");
    assert_eq!(meta["period"], "20091231");

    // Check standardized filer details
    let filer = &meta["filer"];
    assert_eq!(
        filer["company-data"]["conformed-name"],
        "TGFIN HOLDINGS INC"
    );
    assert_eq!(filer["company-data"]["cik"], "0000876134");
    assert_eq!(filer["company-data"]["assigned-sic"], "6770");
    assert_eq!(filer["filing-values"]["act"], "34");

    // Check first document
    let doc0 = &docs[0];
    assert_eq!(doc0.meta.r#type, "10-K");
    assert_eq!(doc0.meta.sequence, "1");
    assert_eq!(doc0.meta.filename, "tgfin10k123109040610finaldra.htm");
    assert!(!doc0.content.is_empty());

    // Check calculate_documents_locations_in_tar
    calculate_documents_locations_in_tar(&mut meta).expect("tar calculation");
    let doc0_meta = &meta["documents"][0];
    assert!(doc0_meta["secsgml_start_byte"].as_str().is_some());
    assert!(doc0_meta["secsgml_end_byte"].as_str().is_some());
}

#[test]
fn test_multiple_filers_sample() {
    let path = "/tmp/secsgml_samples/multiplefilers.txt";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let data = std::fs::read(path).expect("read sample");
    let (meta, docs) = parse_sgml(&data, None).expect("parse sgml");

    assert_eq!(docs.len(), 1);
    let filers = meta["filer"].as_array().expect("filers should be an array");
    assert_eq!(filers.len(), 399);
}

#[test]
fn test_archive_uuencoded_and_filtering() {
    let path = "/tmp/secsgml_samples/archive.txt";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let data = std::fs::read(path).expect("read sample");

    // Filter by type "8-K"
    let (_meta_8k, docs_8k) = parse_sgml(&data, Some(&["8-K"])).expect("parse filtered");
    assert_eq!(docs_8k.len(), 1);
    assert_eq!(docs_8k[0].meta.r#type, "8-K");

    // Parse all
    let (meta_all, docs_all) = parse_sgml_content_into_memory(&data, None).expect("parse memory");
    assert_eq!(docs_all.len(), 2);
    let docs_arr = meta_all["documents"].as_array().unwrap();
    assert_eq!(docs_arr[0]["type"], "8-K");
    assert_eq!(docs_arr[1]["type"], "EX-1");
}

#[test]
fn test_synthetic_sgml_with_uuencode() {
    let sgml = br#"<SEC-DOCUMENT>0000000000-00-000000.txt : 20260101
ACCESSION NUMBER: 0000000000-00-000000
CONFORMED SUBMISSION TYPE: 10-Q
CENTRAL INDEX KEY: 0001234567

<DOCUMENT>
<TYPE>10-Q
<SEQUENCE>1
<FILENAME>report.txt
<DESCRIPTION>Quarterly Report
<TEXT>
This is the narrative body text of the 10-Q filing.
</TEXT>
</DOCUMENT>

<DOCUMENT>
<TYPE>GRAPHIC
<SEQUENCE>2
<FILENAME>cat.bin
<DESCRIPTION>Cat image attachment
<TEXT>
begin 644 cat.bin
#0V%T
`
end
</TEXT>
</DOCUMENT>
"#;

    let (meta, docs) = parse_sgml(sgml, None).expect("parse synthetic");
    assert_eq!(meta["accession-number"], "0000000000-00-000000");
    assert_eq!(meta["type"], "10-Q");
    assert_eq!(meta["cik"], "0001234567");

    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].meta.filename, "report.txt");
    assert_eq!(
        std::str::from_utf8(&docs[0].content).unwrap().trim(),
        "This is the narrative body text of the 10-Q filing."
    );
    assert!(!docs[0].is_uuencoded);

    assert_eq!(docs[1].meta.filename, "cat.bin");
    assert!(docs[1].is_uuencoded);
    assert_eq!(docs[1].content, b"Cat");
}
