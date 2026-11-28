//! The library's error type.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
  /// a rule that cannot be parsed or run
  Rule(String),
  /// a file that cannot be read or written
  Io(String),
  /// a pattern file that cannot be parsed, or a pattern that does not fit
  Pattern(String),
  /// the GPU: no adapter, a limit, a device error
  Gpu(String),
  /// an invalid configuration
  Config(String),
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Error::Rule(m) => write!(f, "rule: {m}"),
      Error::Io(m) => write!(f, "i/o: {m}"),
      Error::Pattern(m) => write!(f, "pattern: {m}"),
      Error::Gpu(m) => write!(f, "gpu: {m}"),
      Error::Config(m) => write!(f, "config: {m}"),
    }
  }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
  fn from(e: std::io::Error) -> Self {
    Error::Io(e.to_string())
  }
}

pub type Result<T> = std::result::Result<T, Error>;
