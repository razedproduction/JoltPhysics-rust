//! Port of UnitTests/Core/StringToolsTest.cpp

use jolt_core::string_format;
use jolt_core::string_tools as st;

#[test]
fn test_string_format() {
    assert_eq!(string_format!("Test: {}", 1234), "Test: 1234");
}

#[test]
fn test_convert_to_string() {
    assert_eq!(st::convert_to_string(&1234), "1234");
    assert_eq!(st::convert_to_string(&-1), "-1");
    assert_eq!(st::convert_to_string(&0x7fffffffffffffffu64), "9223372036854775807");
}

#[test]
fn string_replace() {
    let mut value = String::from("Hello this si si a test");
    st::string_replace(&mut value, "si", "is");
    assert_eq!(value, "Hello this is is a test");
    st::string_replace(&mut value, "is is", "is");
    assert_eq!(value, "Hello this is a test");
    st::string_replace(&mut value, "Hello", "Bye");
    assert_eq!(value, "Bye this is a test");
    st::string_replace(&mut value, "a test", "complete");
    assert_eq!(value, "Bye this is complete");
}

#[test]
fn string_to_vector() {
    let value = st::string_to_vector("", ",");
    assert!(value.is_empty());

    let value = st::string_to_vector("a,b,c", ",");
    assert_eq!(value[0], "a");
    assert_eq!(value[1], "b");
    assert_eq!(value[2], "c");

    let value = st::string_to_vector("a,.b,.c,", ".");
    assert_eq!(value[0], "a,");
    assert_eq!(value[1], "b,");
    assert_eq!(value[2], "c,");
}

#[test]
fn vector_to_string() {
    let mut input: Vec<String> = Vec::new();
    let value = st::vector_to_string(&input, ",");
    assert!(value.is_empty());

    input = vec!["a".into(), "b".into(), "c".into()];
    let value = st::vector_to_string(&input, ",");
    assert_eq!(value, "a,b,c");

    let value = st::vector_to_string(&input, ", ");
    assert_eq!(value, "a, b, c");
}

#[test]
fn to_lower() {
    assert_eq!(st::to_lower("123 HeLlO!"), "123 hello!");
}

#[test]
fn nibble_to_binary() {
    assert_eq!(st::nibble_to_binary(0b0000), "0000");
    assert_eq!(st::nibble_to_binary(0b0001), "0001");
    assert_eq!(st::nibble_to_binary(0b0010), "0010");
    assert_eq!(st::nibble_to_binary(0b0011), "0011");
    assert_eq!(st::nibble_to_binary(0b0100), "0100");
    assert_eq!(st::nibble_to_binary(0b0101), "0101");
    assert_eq!(st::nibble_to_binary(0b0110), "0110");
    assert_eq!(st::nibble_to_binary(0b0111), "0111");
    assert_eq!(st::nibble_to_binary(0b1000), "1000");
    assert_eq!(st::nibble_to_binary(0b1001), "1001");
    assert_eq!(st::nibble_to_binary(0b1010), "1010");
    assert_eq!(st::nibble_to_binary(0b1011), "1011");
    assert_eq!(st::nibble_to_binary(0b1100), "1100");
    assert_eq!(st::nibble_to_binary(0b1101), "1101");
    assert_eq!(st::nibble_to_binary(0b1110), "1110");
    assert_eq!(st::nibble_to_binary(0b1111), "1111");

    assert_eq!(st::nibble_to_binary(0xfffffff0), "0000");
}
