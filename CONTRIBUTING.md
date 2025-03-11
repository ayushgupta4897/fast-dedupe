# Contributing to fast-dedupe

Thank you for considering contributing to fast-dedupe! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run the tests to ensure they pass
5. Submit a pull request

## Development Setup

1. Clone your fork of the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Running Tests

Run the tests with pytest:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=fastdedupe
```

## Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

You can run all of these with:

```bash
black fastdedupe
isort fastdedupe
flake8 fastdedupe
mypy fastdedupe
```

## Pull Request Process

1. Ensure your code passes all tests and style checks
2. Update the README.md with details of changes if appropriate
3. Update the version number in `fastdedupe/__init__.py` following [Semantic Versioning](https://semver.org/)
4. The pull request will be merged once it has been reviewed and approved

## Feature Requests and Bug Reports

Please use the GitHub issue tracker to submit feature requests and bug reports.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License. 