//! Port of UnitTests/Core/ScopeExitTest.cpp

use std::cell::Cell;

use jolt_core::scope_exit;
use jolt_core::scope_exit::ScopeExit;

#[test]
fn test_scope_exit_order() {
    let value = Cell::new(0);
    {
        // Last created should be first destroyed
        scope_exit!(|| {
            assert_eq!(value.get(), 1);
            value.set(2);
        });
        scope_exit!(|| {
            assert_eq!(value.get(), 0);
            value.set(1);
        });
        assert_eq!(value.get(), 0);
    }
    assert_eq!(value.get(), 2);
}

#[test]
fn test_scope_exit_release() {
    let value = Cell::new(0);
    {
        let mut scope_exit = ScopeExit::new(|| value.set(value.get() + 1));
        assert_eq!(value.get(), 0);
        // Don't call the exit function anymore
        scope_exit.release();
    }
    assert_eq!(value.get(), 0);
}

#[test]
fn test_scope_exit_invoke() {
    let value = Cell::new(0);
    {
        let mut scope_exit = ScopeExit::new(|| value.set(value.get() + 1));
        assert_eq!(value.get(), 0);
        scope_exit.invoke();
        assert_eq!(value.get(), 1);
        // Should not call again on exit
    }
    assert_eq!(value.get(), 1);
}
