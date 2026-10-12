//! Port of `Jolt/Core/Result.h`.
//!
//! The C++ `Result<T>` (valid value, error string or empty) maps to [`Result<T>`] = `std::result::Result<T,
//! JoltError>`; the empty state is expressed as `Option<Result<T>>` where needed (e.g. cached creation results).

use std::fmt;

/// The error description if a result failed (the C++ `Result<T>` error string)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JoltError(pub String);

impl JoltError {
    /// Create an error from a description
    pub fn new(error: impl Into<String>) -> Self {
        Self(error.into())
    }

    /// Get the error description
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JoltError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for JoltError {}

impl From<String> for JoltError {
    fn from(error: String) -> Self {
        Self(error)
    }
}

impl From<&str> for JoltError {
    fn from(error: &str) -> Self {
        Self(error.to_string())
    }
}

/// Helper type that either contains a valid result or an error
pub type Result<T> = std::result::Result<T, JoltError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_message_round_trip() {
        fn create() -> Result<i32> {
            Err("Invalid shape".into())
        }
        assert_eq!(create().unwrap_err().to_string(), "Invalid shape");
        assert_eq!(JoltError::new("x").message(), "x");
    }
}
