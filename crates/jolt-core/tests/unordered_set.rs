//! Port of UnitTests/Core/UnorderedSetTest.cpp

use std::sync::atomic::{AtomicBool, Ordering};

use jolt_core::{KeyHasher, UnorderedSet};

#[test]
fn test_unordered_set() {
    let mut set: UnorderedSet<i32> = UnorderedSet::new();
    assert_eq!(set.bucket_count(), 0);
    set.reserve(10);
    assert_eq!(set.bucket_count(), 16);

    // Check system limits
    assert_eq!(set.max_bucket_count(), 0x80000000);
    assert_eq!(set.max_size() as u64, 0x80000000u64 * 7 / 8);

    // Insert some entries
    assert_eq!(*set.insert(1).0, 1);
    assert!(set.insert(3).1);
    assert!(!set.insert(3).1);
    assert_eq!(set.len(), 2);
    assert_eq!(*set.find(&1).unwrap(), 1);
    assert_eq!(*set.find(&3).unwrap(), 3);
    assert!(set.find(&5).is_none());

    // Validate all elements are visited by a visitor
    let mut count = 0;
    let mut visited = [false; 10];
    for i in set.iter() {
        visited[*i as usize] = true;
        count += 1;
    }
    assert_eq!(count, 2);
    assert!(visited[1]);
    assert!(visited[3]);
    for i in set.iter_mut() {
        visited[*i as usize] = false;
        count -= 1;
    }
    assert_eq!(count, 0);
    assert!(!visited[1]);
    assert!(!visited[3]);

    // Copy the set
    let mut set2: UnorderedSet<i32> = UnorderedSet::new();
    assert!(set2.is_empty());
    set2 = set.clone();
    assert_eq!(*set2.find(&1).unwrap(), 1);
    assert_eq!(*set2.find(&3).unwrap(), 3);
    assert!(set2.find(&5).is_none());

    // Swap
    let mut set3: UnorderedSet<i32> = UnorderedSet::new();
    set3.swap(&mut set);
    assert_eq!(*set3.find(&1).unwrap(), 1);
    assert_eq!(*set3.find(&3).unwrap(), 3);
    assert!(set3.find(&5).is_none());
    assert!(set.is_empty());

    // Move construct
    let mut set4 = std::mem::take(&mut set3);
    assert_eq!(*set4.find(&1).unwrap(), 1);
    assert_eq!(*set4.find(&3).unwrap(), 3);
    assert!(set4.find(&5).is_none());
    assert!(set3.is_empty());

    // Move assign
    let mut set5: UnorderedSet<i32> = UnorderedSet::new();
    set5.insert(999);
    assert_eq!(*set5.find(&999).unwrap(), 999);
    set5 = std::mem::take(&mut set4);
    assert!(set5.find(&999).is_none());
    assert_eq!(*set5.find(&1).unwrap(), 1);
    assert_eq!(*set5.find(&3).unwrap(), 3);
    assert!(set4.is_empty());
}

#[test]
fn test_unordered_set_grow() {
    let mut set: UnorderedSet<i32> = UnorderedSet::new();
    for i in 0..10000 {
        assert!(set.insert(i).1);
    }

    assert_eq!(set.len(), 10000);

    for i in 0..10000 {
        assert_eq!(*set.find(&i).unwrap(), i);
    }

    assert!(set.find(&10001).is_none());

    for i in 0..5000 {
        assert_eq!(set.erase(&i), 1);
    }

    assert_eq!(set.len(), 5000);

    for i in 0..5000 {
        assert!(set.find(&i).is_none());
    }

    for i in 5000..10000 {
        assert_eq!(*set.find(&i).unwrap(), i);
    }

    assert!(set.find(&10001).is_none());

    for i in 0..5000 {
        assert!(set.insert(i).1);
    }

    assert!(!set.insert(0).1);

    assert_eq!(set.len(), 10000);

    for i in 0..10000 {
        assert_eq!(*set.find(&i).unwrap(), i);
    }

    assert!(set.find(&10001).is_none());
}

#[test]
fn test_unordered_set_hash_collision() {
    // A hash function that's guaranteed to collide
    struct MyBadHash;
    impl KeyHasher<i32> for MyBadHash {
        fn hash(_value: &i32) -> u64 {
            0
        }
    }

    let mut set: UnorderedSet<i32, MyBadHash> = UnorderedSet::new();
    for i in 0..10 {
        assert!(set.insert(i).1);
    }

    assert_eq!(set.len(), 10);

    for i in 0..10 {
        assert_eq!(*set.find(&i).unwrap(), i);
    }

    assert!(set.find(&11).is_none());

    for i in 0..5 {
        assert_eq!(set.erase(&i), 1);
    }

    assert_eq!(set.len(), 5);

    for i in 0..5 {
        assert!(set.find(&i).is_none());
    }

    for i in 5..10 {
        assert_eq!(*set.find(&i).unwrap(), i);
    }

    assert!(set.find(&11).is_none());

    for i in 0..5 {
        assert!(set.insert(i).1);
    }

    assert!(!set.insert(0).1);

    assert_eq!(set.len(), 10);

    for i in 0..10 {
        assert_eq!(*set.find(&i).unwrap(), i);
    }

    assert!(set.find(&11).is_none());
}

#[test]
fn test_unordered_set_add_remove_cyles() {
    let mut set: UnorderedSet<i32> = UnorderedSet::new();
    const BUCKET_COUNT: u32 = 64;
    set.reserve((set.max_load_factor() * BUCKET_COUNT as f32) as u32);
    assert_eq!(set.bucket_count(), BUCKET_COUNT);

    // Repeatedly add and remove elements to see if the set cleans up tombstones
    const NUM_ELEMENTS: i32 = 64 * 6 / 8; // We make sure that the map is max 6/8 full to ensure that we never grow the map but rehash it instead
    let mut add_counter = 0;
    let mut remove_counter = 0;
    for _ in 0..100 {
        for _ in 0..NUM_ELEMENTS {
            assert!(set.find(&add_counter).is_none());
            assert!(set.insert(add_counter).1);
            assert!(set.find(&add_counter).is_some());
            add_counter += 1;
        }

        assert_eq!(set.len(), NUM_ELEMENTS as u32);

        for _ in 0..NUM_ELEMENTS {
            assert!(set.find(&remove_counter).is_some());
            assert_eq!(set.erase(&remove_counter), 1);
            assert_eq!(set.erase(&remove_counter), 0);
            assert!(set.find(&remove_counter).is_none());
            remove_counter += 1;
        }

        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    // Test that adding and removing didn't resize the set
    assert_eq!(set.bucket_count(), BUCKET_COUNT);
}

#[test]
fn test_unordered_set_many_tomb_stones() {
    // A hash function that makes sure that consecutive ints end up in consecutive buckets starting at bucket 63
    struct MyBadHash;
    impl KeyHasher<i32> for MyBadHash {
        fn hash(value: &i32) -> u64 {
            ((*value + 63) << 7) as u64
        }
    }

    let mut set: UnorderedSet<i32, MyBadHash> = UnorderedSet::new();
    const BUCKET_COUNT: u32 = 64;
    set.reserve((set.max_load_factor() * BUCKET_COUNT as f32) as u32);
    assert_eq!(set.bucket_count(), BUCKET_COUNT);

    // Fill 32 buckets
    let mut add_counter = 0;
    for _ in 0..32 {
        assert!(set.insert(add_counter).1);
        add_counter += 1;
    }

    // Since we control the hash, we know in which order we'll visit the elements
    // The first element was inserted in bucket 63, so we start at 1
    let mut expected = 1;
    for &i in &set {
        assert_eq!(i, expected);
        expected = (expected + 1) & 31;
    }
    expected = 1;
    for &i in &set {
        assert_eq!(i, expected);
        expected = (expected + 1) & 31;
    }

    // Remove a bucket in the middle with so that the number of occupied slots
    // surrounding the bucket exceed 16 to force creating a tombstone,
    // then add one at the end
    for remove_counter in 16..16 + 100 {
        assert!(set.find(&remove_counter).is_some());
        assert_eq!(set.erase(&remove_counter), 1);
        assert!(set.find(&remove_counter).is_none());

        assert!(set.find(&add_counter).is_none());
        assert!(set.insert(add_counter).1);
        assert!(set.find(&add_counter).is_some());

        add_counter += 1;
    }

    // Check that the elements we inserted are still there
    assert_eq!(set.len(), 32);
    for i in 0..16 {
        assert_eq!(*set.find(&i).unwrap(), i);
    }
    for i in 0..16 {
        assert_eq!(*set.find(&(add_counter - 1 - i)).unwrap(), add_counter - 1 - i);
    }

    // Test that adding and removing didn't resize the set
    assert_eq!(set.bucket_count(), BUCKET_COUNT);
}

static REVERSED_HASH: AtomicBool = AtomicBool::new(false);

#[test]
fn test_unordered_set_rehash() {
    // A hash function for which we can switch the hashing algorithm
    struct MyBadHash;
    impl KeyHasher<i32> for MyBadHash {
        fn hash(value: &i32) -> u64 {
            ((if REVERSED_HASH.load(Ordering::Relaxed) { 127 - *value } else { *value }) << 7) as u64
        }
    }

    type Set = UnorderedSet<i32, MyBadHash>;
    let mut set = Set::new();
    const BUCKET_COUNT: u32 = 128;
    set.reserve((set.max_load_factor() * BUCKET_COUNT as f32) as u32);
    assert_eq!(set.bucket_count(), BUCKET_COUNT);

    // Fill buckets
    REVERSED_HASH.store(false, Ordering::Relaxed);
    const NUM_ELEMENTS: i32 = 96;
    for i in 0..NUM_ELEMENTS {
        assert!(set.insert(i).1);
    }

    // Check that we get the elements in the expected order
    for (expected, &i) in set.iter().enumerate() {
        assert_eq!(i, expected as i32);
    }

    // Change the hashing algorithm so that a rehash is forced to move elements.
    // The test is designed in such a way that it will both need to move elements to empty slots
    // and to move elements to slots that currently already have another element.
    REVERSED_HASH.store(true, Ordering::Relaxed);
    set.rehash(0);

    // Check that all elements are still there
    for i in 0..NUM_ELEMENTS {
        assert_eq!(*set.find(&i).unwrap(), i);
    }

    // The hash went from filling buckets 0 .. 95 with values 0 .. 95 to bucket 127 .. 31 with values 0 .. 95
    // However, we don't move elements if they still fall within the same batch, this means that the first 8
    // elements didn't move
    let mut it = set.iter();
    for i in 0..8 {
        assert_eq!(*it.next().unwrap(), i);
    }

    // The rest will have been reversed
    for i in (8..=95).rev() {
        assert_eq!(*it.next().unwrap(), i);
    }

    // Test that adding and removing didn't resize the set
    assert_eq!(set.bucket_count(), BUCKET_COUNT);
}
