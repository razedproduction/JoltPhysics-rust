//! Port of `Jolt/Core/StringTools.h` / `StringTools.cpp`.

use std::fmt::Display;

/// Size of the internal buffer of [`string_format!`](crate::string_format) (C++ `StringFormat`), including the
/// terminating zero of the C++ version.
pub const STRING_FORMAT_BUFFER_SIZE: usize = 1024;

/// Create a formatted text string for debugging purposes (C++ `StringFormat`). Takes Rust [`format!`] syntax
/// instead of `printf` syntax.
/// Note that, like the C++ version, the result is trimmed to 1023 bytes (at a character boundary).
#[macro_export]
macro_rules! string_format {
    ($($arg:tt)*) => {
        $crate::string_tools::truncate_to_format_buffer(::std::format!($($arg)*))
    };
}

/// Trim `string` to what fits in the 1024 byte buffer of the C++ `StringFormat` (used by
/// [`string_format!`](crate::string_format)).
pub fn truncate_to_format_buffer(mut string: String) -> String {
    let mut len = STRING_FORMAT_BUFFER_SIZE - 1;
    if string.len() > len {
        while !string.is_char_boundary(len) {
            len -= 1;
        }
        string.truncate(len);
    }
    string
}

/// Convert type to string
///
/// Uses [`Display`] instead of C++ `operator <<`, which formats floating point numbers differently.
pub fn convert_to_string<T: Display + ?Sized>(value: &T) -> String {
    value.to_string()
}

/// Replace substring with other string
pub fn string_replace(string: &mut String, search: &str, replace: &str) {
    // The C++ version never terminates for an empty search string
    debug_assert!(!search.is_empty());

    let mut index = 0;
    while let Some(found) = string[index..].find(search) {
        index += found;

        string.replace_range(index..index + search.len(), replace);

        index += replace.len();
    }
}

/// Convert a delimited string to an array of strings
///
/// The C++ version appends to an output vector (optionally clearing it first); here the elements are returned.
pub fn string_to_vector(string: &str, delimiter: &str) -> Vec<String> {
    debug_assert!(!delimiter.is_empty());

    let mut out = Vec::new();

    // No string? no elements
    if string.is_empty() {
        return out;
    }

    // Start with initial string
    let mut s = string;

    // Add to vector while we have a delimiter
    while !s.is_empty() {
        let Some(i) = s.find(delimiter) else { break };
        out.push(s[..i].to_string());
        s = &s[i + delimiter.len()..];
    }

    // Add final element
    out.push(s.to_string());
    out
}

/// Convert an array strings to a delimited string
pub fn vector_to_string<S: AsRef<str>>(vector: &[S], delimiter: &str) -> String {
    let mut out = String::new();

    for s in vector {
        // Add delimiter if not first element (like C++: also skipped while the output is still empty)
        if !out.is_empty() {
            out.push_str(delimiter);
        }

        // Add element
        out.push_str(s.as_ref());
    }
    out
}

/// Convert a string to lower case (ASCII only, like C++ `tolower` in the "C" locale)
pub fn to_lower(string: &str) -> String {
    string.to_ascii_lowercase()
}

/// Converts the lower 4 bits of `nibble` to a string that represents the number in binary format
pub fn nibble_to_binary(nibble: u32) -> &'static str {
    const NIBBLES: [&str; 16] = [
        "0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001", "1010", "1011", "1100", "1101",
        "1110", "1111",
    ];
    NIBBLES[(nibble & 0xf) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_format_truncates_like_cpp_buffer() {
        let long = "x".repeat(2000);
        assert_eq!(crate::string_format!("{long}").len(), 1023);
        let multibyte = "\u{e9}".repeat(1000); // 2 bytes each, 1023 is not a boundary
        assert_eq!(crate::string_format!("{multibyte}").len(), 1022);
    }

    #[test]
    fn vector_to_string_skips_delimiter_after_empty_prefix() {
        assert_eq!(vector_to_string(&["", "a", "b"], ","), "a,b");
    }

    #[test]
    fn string_to_vector_edge_cases() {
        assert_eq!(string_to_vector("a,", ","), ["a", ""]);
        assert_eq!(string_to_vector(",", ","), ["", ""]);
    }
}
