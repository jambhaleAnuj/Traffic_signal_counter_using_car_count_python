# Contributing to Adaptive Traffic Signal Management (ATSM)

Thanks for your interest in contributing! This project welcomes bug reports, feature suggestions, documentation improvements, and pull requests.

## Ground rules

- Be respectful and follow our [Code of Conduct](./CODE_OF_CONDUCT.md).
- For substantial changes, please open an issue first to discuss what you would like to change.
- Keep PRs focused and small where possible; add tests or minimal repro where applicable.

## How to contribute

1. Fork the repo and create your branch from `main`.
2. Set up your environment and run the demo to reproduce baseline behavior.
3. Make changes with clear commit messages.
4. If you change public behavior, update the README and/or add small tests or examples.
5. Open a PR using the provided template. Link any related issues.

## Development tips

- Use Python 3.12+.
- Prefer `yolov8n.pt` for quick testing; switch to `yolov8l.pt` to validate accuracy claims.
- Keep thresholds and constants configurable where reasonable.

## Reporting issues

Please include:

- Environment (OS, Python version, CPU/GPU, CUDA if any)
- Exact command(s) you ran and full output logs
- Screenshots or short screen captures, if visual behavior is involved
- Minimal reproduction, if possible

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 for code, and CC BY-SA 4.0 for documentation/paper content.
