use thiserror::Error;

#[derive(Error, Debug)]
pub enum SecSgmlError {
    #[error("Truncated or malformed SGML document")]
    Truncated,
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("UTF-8 decoding error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),
    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
