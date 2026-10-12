//! Port of UnitTests/Core/HashCombineTest.cpp

use jolt_core::hash_combine::{JoltHash, hash_bytes, hash_combine, hash_string};

#[test]
fn test_hash_bytes() {
    assert_eq!(hash_bytes(b"This is a test"), 2733878766136413408);
}

#[test]
fn test_hash_string() {
    assert_eq!(hash_string("This is a test"), 2733878766136413408);
}

#[test]
fn test_hash_struct() {
    let char_test: &str = "This is a test";
    assert_eq!(char_test.jolt_hash(), 2733878766136413408);

    let str_test = String::from("This is a test");
    assert_eq!(str_test.jolt_hash(), 2733878766136413408);
}

#[test]
fn test_hash_combine() {
    let val1: i32 = 0;
    let val1_hash = val1.jolt_hash();
    let val2: i32 = 1;
    let val2_hash = val2.jolt_hash();

    // Check non-commutative
    let mut seed1 = val1_hash;
    hash_combine(&mut seed1, &val2);
    let mut seed2 = val2_hash;
    hash_combine(&mut seed2, &val1);
    assert_ne!(seed1, seed2);

    // Check that adding a 0 changes the hash
    let mut seed3 = val1_hash;
    hash_combine(&mut seed3, &val1);
    assert_ne!(seed3, val1_hash);
}
