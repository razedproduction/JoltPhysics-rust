//! Port of `Jolt/Core/UnorderedSet.h` (and the defaults of `UnorderedSetFwd.h`).

use crate::hash_table::{HashTable, HashTableDetail, JoltHasher};

/// Internal helper class to provide context for UnorderedSet
#[derive(Clone, Copy, Debug, Default)]
pub struct UnorderedSetDetail;

impl<K> HashTableDetail<K, K> for UnorderedSetDetail {
    /// The key is the key, just return it
    #[inline]
    fn key(key: &K) -> &K {
        key
    }
}

/// Hash Set class
/// - `K`: Key type
/// - `H`: Hash function (note should be 64-bits)
///
/// Key equality is [`PartialEq`] (C++ `KeyEqual` template parameter).
pub type UnorderedSet<K, H = JoltHasher> = HashTable<K, K, UnorderedSetDetail, H>;
