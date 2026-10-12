//! Port of `Jolt/Core/ScopeExit.h`.

use std::fmt;

/// Class that calls a function when it goes out of scope
#[must_use = "the exit function runs when the ScopeExit is dropped; binding it to `_` drops it immediately"]
pub struct ScopeExit<F: FnOnce()> {
    function: Option<F>,
}

impl<F: FnOnce()> ScopeExit<F> {
    /// Constructor specifies the exit function
    #[inline]
    pub fn new(function: F) -> Self {
        Self { function: Some(function) }
    }

    /// Call the exit function now instead of when going out of scope
    #[inline]
    pub fn invoke(&mut self) {
        if let Some(function) = self.function.take() {
            function();
        }
    }

    /// No longer call the exit function when going out of scope
    #[inline]
    pub fn release(&mut self) {
        self.function = None;
    }
}

/// Destructor calls the exit function
impl<F: FnOnce()> Drop for ScopeExit<F> {
    #[inline]
    fn drop(&mut self) {
        self.invoke();
    }
}

impl<F: FnOnce()> fmt::Debug for ScopeExit<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopeExit").field("invoked", &self.function.is_none()).finish()
    }
}

/// Usage: `scope_exit!(|| { code to call on scope exit });`
///
/// Guards created in the same scope run in reverse order of creation.
#[macro_export]
macro_rules! scope_exit {
    ($function:expr) => {
        let _scope_exit_guard = $crate::scope_exit::ScopeExit::new($function);
    };
}
