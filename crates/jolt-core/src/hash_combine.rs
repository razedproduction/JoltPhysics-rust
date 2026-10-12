//! Port of `Jolt/Core/HashCombine.h`: FNV-1a byte hashing, a 64 bit integer hash and a hash combiner.
//!
//! The C++ `Hash<T>` functor is the [`JoltHash`] trait here (named so it does not clash with [`std::hash::Hash`]).
//! Hash values must match the C++ implementation bit for bit, so all arithmetic wraps like C++ unsigned math.

/// Default seed of [`hash_bytes`] / [`hash_string`] (the FNV-1a 64 bit offset basis).
pub const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;

/// The FNV-1a 64 bit prime.
const FNV_PRIME: u64 = 0x100000001b3;

/// Implements the FNV-1a hash algorithm
/// @see <https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function>
/// @param data Data block of bytes
/// @return Hash
#[inline]
pub const fn hash_bytes(data: &[u8]) -> u64 {
    hash_bytes_with_seed(data, FNV_OFFSET_BASIS)
}

/// Same as [`hash_bytes`], with an explicit seed (can be used to pass in the hash of a previous operation).
#[inline]
pub const fn hash_bytes_with_seed(data: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    let mut i = 0;
    while i < data.len() {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// Calculate the FNV-1a hash of `string`.
/// @see <https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function>
///
/// Like the C++ version (which reads a `const char *`) hashing stops at the first NUL byte, and each byte is
/// sign-extended as a (signed, on x86) `char` before being mixed in, so non-ASCII strings hash differently from
/// [`hash_bytes`]. See `port/DEVIATIONS.md`.
#[inline]
pub const fn hash_string(string: &str) -> u64 {
    hash_string_with_seed(string, FNV_OFFSET_BASIS)
}

/// Same as [`hash_string`], with an explicit seed.
#[inline]
pub const fn hash_string_with_seed(string: &str, seed: u64) -> u64 {
    let bytes = string.as_bytes();
    let mut hash = seed;
    let mut i = 0;
    while i < bytes.len() && bytes[i] != 0 {
        hash ^= bytes[i] as i8 as i64 as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// A 64 bit hash function by Thomas Wang, Jan 1997
/// See: <http://web.archive.org/web/20071223173210/http://www.concentric.net/~Ttwang/tech/inthash.htm>
/// @param value Value to hash
/// @return Hash
#[inline]
pub const fn hash64(value: u64) -> u64 {
    let mut hash = value;
    hash = (!hash).wrapping_add(hash << 21); // hash = (hash << 21) - hash - 1;
    hash ^= hash >> 24;
    hash = hash.wrapping_add(hash << 3).wrapping_add(hash << 8); // hash * 265
    hash ^= hash >> 14;
    hash = hash.wrapping_add(hash << 2).wrapping_add(hash << 4); // hash * 21
    hash ^= hash >> 28;
    hash = hash.wrapping_add(hash << 31);
    hash
}

/// Port of the C++ `Hash<T>` functor: a deterministic 64 bit hash (unlike [`std::hash::Hash`], whose values are not
/// specified). Implement it for your own types with [`make_hashable!`](crate::make_hashable).
pub trait JoltHash {
    /// Hash `self`.
    fn jolt_hash(&self) -> u64;
}

impl<T: JoltHash + ?Sized> JoltHash for &T {
    #[inline]
    fn jolt_hash(&self) -> u64 {
        (**self).jolt_hash()
    }
}

/// A hash function for floats
impl JoltHash for f32 {
    #[inline]
    fn jolt_hash(&self) -> u64 {
        let value = if *self == 0.0 { 0.0f32 } else { *self }; // Convert -0.0f to 0.0f
        hash_bytes(&value.to_ne_bytes())
    }
}

/// A hash function for doubles
impl JoltHash for f64 {
    #[inline]
    fn jolt_hash(&self) -> u64 {
        let value = if *self == 0.0 { 0.0f64 } else { *self }; // Convert -0.0 to 0.0
        hash_bytes(&value.to_ne_bytes())
    }
}

/// A hash function for strings (C++ `std::string_view`: hashes all bytes, NUL bytes included)
impl JoltHash for str {
    #[inline]
    fn jolt_hash(&self) -> u64 {
        hash_bytes(self.as_bytes())
    }
}

/// A hash function for String
impl JoltHash for String {
    #[inline]
    fn jolt_hash(&self) -> u64 {
        hash_bytes(self.as_bytes())
    }
}

/// A fallback function for generic pointers (hashes the address)
impl<T: ?Sized> JoltHash for *const T {
    #[inline]
    fn jolt_hash(&self) -> u64 {
        hash_bytes(&(self.cast::<()>() as usize).to_ne_bytes())
    }
}

/// A fallback function for generic pointers (hashes the address)
impl<T: ?Sized> JoltHash for *mut T {
    #[inline]
    fn jolt_hash(&self) -> u64 {
        self.cast_const().jolt_hash()
    }
}

/// Helper macro to define a hash function for trivial types (C++ `JPH_DEFINE_TRIVIAL_HASH`)
macro_rules! define_trivial_hash {
    ($($type:ty),*) => {
        $(
            impl JoltHash for $type {
                #[inline]
                fn jolt_hash(&self) -> u64 {
                    hash_bytes(&self.to_ne_bytes())
                }
            }
        )*
    };
}

// Commonly used types (C++ `char` is `i8` / `u8` here; Rust `char` is 4 bytes and has no C++ equivalent)
define_trivial_hash!(i8, u8, i32, u32, u64);

/// Helper function that hashes a single value into `seed`
/// Based on <https://github.com/jonmaiga/mx3> by Jon Maiga
#[inline]
pub fn hash_combine<T: JoltHash + ?Sized>(seed: &mut u64, value: &T) {
    const C: u64 = 0xbea225f9eb34556d;

    let mut h = *seed;
    let mut x = value.jolt_hash();

    // See: https://github.com/jonmaiga/mx3/blob/master/mx3.h
    // mix_stream(h, x)
    x = x.wrapping_mul(C);
    x ^= x >> 39;
    h = h.wrapping_add(x.wrapping_mul(C));
    h = h.wrapping_mul(C);

    // mix(h)
    h ^= h >> 32;
    h = h.wrapping_mul(C);
    h ^= h >> 29;
    h = h.wrapping_mul(C);
    h ^= h >> 32;
    h = h.wrapping_mul(C);
    h ^= h >> 29;

    *seed = h;
}

/// Hash combiner (C++ `HashCombineArgs`): primes the seed with the hash of the first value, then combines all
/// remaining values into it with [`hash_combine`].
#[macro_export]
macro_rules! hash_combine_args {
    ($first:expr $(, $rest:expr)* $(,)?) => {{
        // Prime the seed by hashing the first value
        #[allow(unused_mut)]
        let mut seed: u64 = $crate::hash_combine::JoltHash::jolt_hash(&$first);

        // Hash all remaining values together
        $( $crate::hash_combine::hash_combine(&mut seed, &$rest); )*

        seed
    }};
}

/// Hash combiner to use a custom struct as a key (C++ `JPH_MAKE_HASHABLE`): implements [`JoltHash`] for `$type` by
/// combining the listed fields with [`hash_combine_args!`](crate::hash_combine_args).
///
/// Usage:
///
/// ```
/// struct SomeHashKey {
///     key1: String,
///     key2: String,
///     key3: u32,
/// }
///
/// jolt_core::make_hashable!(SomeHashKey, key1, key2, key3);
/// ```
#[macro_export]
macro_rules! make_hashable {
    ($type:ty, $($field:ident),+ $(,)?) => {
        impl $crate::hash_combine::JoltHash for $type {
            fn jolt_hash(&self) -> u64 {
                $crate::hash_combine_args!($(self.$field),+)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_string_sign_extends_like_cpp_char() {
        // U+00E9 = [0xc3, 0xa9]: C++ `uint64(char)` sign-extends bytes >= 0x80
        assert_ne!(hash_string("\u{e9}"), hash_bytes("\u{e9}".as_bytes()));
        assert_eq!(hash_string("ab\0cd"), hash_string("ab"));
    }

    #[test]
    fn negative_zero_hashes_as_zero() {
        assert_eq!((-0.0f32).jolt_hash(), 0.0f32.jolt_hash());
        assert_eq!((-0.0f64).jolt_hash(), 0.0f64.jolt_hash());
    }

    #[test]
    fn hash64_matches_multiplication_form() {
        for v in [0u64, 1, 0x1234_5678_9abc_def0, u64::MAX] {
            let mut h = v;
            h = (h << 21).wrapping_sub(h).wrapping_sub(1);
            h ^= h >> 24;
            h = h.wrapping_mul(265);
            h ^= h >> 14;
            h = h.wrapping_mul(21);
            h ^= h >> 28;
            h = h.wrapping_add(h << 31);
            assert_eq!(hash64(v), h);
        }
    }

    struct Key {
        a: String,
        b: u32,
    }
    crate::make_hashable!(Key, a, b);

    #[test]
    fn make_hashable_combines_fields() {
        let key = Key { a: "x".into(), b: 7 };
        let mut expected = "x".jolt_hash();
        hash_combine(&mut expected, &7u32);
        assert_eq!(key.jolt_hash(), expected);
    }
}
