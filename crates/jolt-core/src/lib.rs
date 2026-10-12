//! Core utilities and math for the Rust port of [Jolt Physics](https://github.com/jrouwe/JoltPhysics).
//!
//! Ported from `Jolt/Core` and `Jolt/Math`. Core utilities live at the crate root (one module per
//! C++ header, e.g. `Jolt/Core/HashCombine.h` -> `hash_combine`), math lives in [`math`].
//! See `port/PORTING_PLAN.md` for the porting status.

pub mod binary_heap;
pub mod color;
pub mod hash_combine;
pub mod hash_table;
pub mod insertion_sort;
pub mod linear_curve;
pub mod math;
pub mod quick_sort;
pub mod result;
pub mod scope_exit;
pub mod static_array;
pub mod string_tools;
pub mod unordered_map;
pub mod unordered_set;

pub use binary_heap::{binary_heap_pop, binary_heap_push};
pub use color::Color;
pub use hash_table::{HashTable, JoltHasher, KeyHasher};
pub use insertion_sort::{insertion_sort, insertion_sort_by};
pub use linear_curve::LinearCurve;
pub use quick_sort::{quick_sort, quick_sort_by};
pub use result::{JoltError, Result};
pub use static_array::StaticArray;
pub use unordered_map::UnorderedMap;
pub use unordered_set::UnorderedSet;
