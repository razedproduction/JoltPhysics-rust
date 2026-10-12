//! Core utilities and math for the Rust port of [Jolt Physics](https://github.com/jrouwe/JoltPhysics).
//!
//! Ported from `Jolt/Core` and `Jolt/Math`. Core utilities live at the crate root (one module per
//! C++ header, e.g. `Jolt/Core/HashCombine.h` -> `hash_combine`), math lives in [`math`].
//! See `port/PORTING_PLAN.md` for the porting status.

pub mod math;
