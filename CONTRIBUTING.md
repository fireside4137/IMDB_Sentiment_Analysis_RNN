# Contributing to IMDB Sentiment Analysis

Thank you for considering contributing to this project! We welcome contributions of all kinds – bug reports, documentation improvements, new features, and code enhancements.

## How to Contribute

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/IMDB_sentiment_analysis.git
   cd IMDB_sentiment_analysis
   ```
3. **Create a new branch** for your work:
   ```bash
   git checkout -b my-feature-or-fix
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```
5. **Make your changes** following the style guidelines below.
6. **Run the test suite** to ensure everything passes:
   ```bash
   pytest
   ```
7. **Commit your changes** with a clear commit message.
8. **Push** the branch to your fork and **open a Pull Request** against the `main` branch of the upstream repository.

## Style Guidelines

- **Code formatting**: `black` (line length 88) and `isort` are enforced via `pre-commit`. Run `pre-commit run --all-files` before committing.
- **Linting**: `flake8` is used for basic lint checks.
- **Type checking**: `mypy` must pass with `--strict`.
- **Docstrings**: Follow the NumPy docstring style for public functions and classes.
- **Tests**: Add unit tests for new functionality. Aim for **90%+** coverage.

## Pull Request Checklist

- [ ] Code follows the style guidelines.
- [ ] All tests pass (`pytest`).
- [ ] Type checking passes (`mypy src`).
- [ ] Documentation is updated where applicable.
- [ ] The PR description explains the changes and any related issue numbers.

## Reporting Issues

If you encounter a bug or have a suggestion, please open an issue on GitHub with:
- A clear title.
- A description of the problem or feature request.
- Steps to reproduce (for bugs) or use‑case details (for features).

We appreciate your help in making this project better!
