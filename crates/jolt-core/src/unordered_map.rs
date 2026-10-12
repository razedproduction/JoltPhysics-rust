//! Port of `Jolt/Core/UnorderedMap.h` (and the defaults of `UnorderedMapFwd.h`).

use crate::hash_table::{HashTable, HashTableDetail, JoltHasher, KeyHasher};

/// Internal helper class to provide context for UnorderedMap
#[derive(Clone, Copy, Debug, Default)]
pub struct UnorderedMapDetail;

impl<K, V> HashTableDetail<K, (K, V)> for UnorderedMapDetail {
    /// Get key from key value pair
    #[inline]
    fn key(key_value: &(K, V)) -> &K {
        &key_value.0
    }
}

/// Hash Map class
/// - `K`: Key type
/// - `V`: Value type
/// - `H`: Hash function (note should be 64-bits)
///
/// Key equality is [`PartialEq`] (C++ `KeyEqual` template parameter).
pub type UnorderedMap<K, V, H = JoltHasher> = HashTable<K, (K, V), UnorderedMapDetail, H>;

impl<K, V, H> HashTable<K, (K, V), UnorderedMapDetail, H>
where
    K: PartialEq,
    H: KeyHasher<K>,
{
    /// Get the value for `key`, inserting a default value if it does not exist (C++ `operator []`)
    pub fn get_or_insert_default(&mut self, key: K) -> &mut V
    where
        V: Default,
    {
        let (inserted, index) = self.insert_key::<false>(&key);
        if inserted {
            self.set_at(index, (key, V::default()));
        }
        &mut self.element_mut(index).1
    }

    /// Insert `value` for `key` if `key` does not exist yet, returns the key value pair and if it was inserted.
    /// Unlike C++, `value` is constructed before the call and is dropped when the key already exists.
    pub fn try_emplace(&mut self, key: K, value: V) -> (&mut (K, V), bool) {
        let (inserted, index) = self.insert_key::<false>(&key);
        if inserted {
            self.set_at(index, (key, value));
        }
        (self.element_mut(index), inserted)
    }

    /// Find the value for `key`, returns `None` if not found
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.find(key).map(|key_value| &key_value.1)
    }

    /// Find the value for `key`, returns `None` if not found
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.find_mut(key).map(|key_value| &mut key_value.1)
    }
}
