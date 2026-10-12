//! Rust port of [Jolt Physics](https://github.com/jrouwe/JoltPhysics).
//!
//! This facade crate re-exports the workspace crates:
//! - [`jolt_core`] (core utilities and [`math`]) at the root,
//! - [`geometry`] ([`jolt_geometry`]),
//! - [`physics`] ([`jolt_physics`]).

pub use jolt_core::*;
pub use jolt_geometry as geometry;
pub use jolt_physics as physics;
