# Contributing

Follow the installation instructions in the README. Keep each pull request focused on one
behavioral change and explain its impact on supported hardware, privacy, or estimates.

## Verification

Run from the repository root:

```sh
python -m pytest
python -m build
```

Add regression tests for behavior changes. Record the operating system, runtime, model
format, and hardware assumptions for compatibility or performance observations. Do not
equate a local smoke test with universal runtime support.

Keep credentials, model files, local caches, and generated build artifacts out of commits.
Follow [SECURITY.md](SECURITY.md) for security-sensitive reports.
