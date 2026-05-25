# ya-ripgrep-core

Native ripgrep-backed matching primitives for YA Agent filesystem tools.

This package exposes a small Python extension built with PyO3. It uses ripgrep's Rust crates for pure matching work while filesystem enumeration and file reads remain owned by `FileOperator`.
