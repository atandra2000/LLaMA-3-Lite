# Files

- [Configuration, Environment, and Artifacts](configuration-and-artifacts.md) - Operational reference for the single training configuration dictionary, CUDA and Triton gates, data and checkpoint paths, scheduling controls, artifact names, telemetry, and safe run-setting changes. Also records fallback behavior and the focused tests that protect configuration and lifecycle invariants.
- [Testing, Validation, and CI](testing-and-ci.md) - Map the CPU-friendly, numerical, smoke, GPU, documentation-reference, benchmark, documentation-build, and GitHub Actions checks so a change can be validated with the narrowest safe command. Records device selection, marker behavior, offline telemetry stubbing, artifact round trips, and the boundaries between tests and documentation generation.
