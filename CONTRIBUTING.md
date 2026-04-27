# Contributing to Hippo Pipeline

Thank you for your interest in contributing! This project is a distributed
LLM inference pipeline for Apple Silicon using MLX.

## How to Contribute

### Bug Reports

- Open a GitHub Issue with the **bug** label
- Include: macOS version, Mac model, MLX version, Python version,
  full error trace
- If possible, include the minimal reproduction steps

### Feature Requests

- Open a GitHub Issue with the **enhancement** label
- Describe the use case and expected benefit
- For performance-related features, include benchmark data if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Test locally with both Rank 0 and Rank 1
5. Submit PR with description of changes and test results

### Code Style

- Python 3.9+ compatible
- Follow existing naming conventions in the codebase
- Add docstrings for new public functions/classes
- Keep functions focused — one responsibility per function

### Testing

- For inference changes: test with a small prompt first, then a longer one
- For network changes: test with both Wi-Fi and Thunderbolt if possible
- Report benchmark numbers (tok/s, step latency) in your PR description

## Communication

- GitHub Issues for bugs and features
- GitHub Discussions for questions and ideas

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0.
