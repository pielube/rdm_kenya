# Contributing to RDM Kenya

Thanks for your interest in contributing! The notes below outline how to get
started and what we expect from contributions.

## Getting started

1. Fork the repository and create your branch from `main`.
2. Set up a local environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. Make your changes with clear, focused commits.
4. Update or add documentation when behavior changes.

## Development guidelines

- Keep changes focused and aligned with the RDM workflow and analysis tools.
- Follow existing code style in the `src/rdm_kenya` package.
- Prefer descriptive commit messages.

## Reporting issues

If you find a bug or have a feature request, please open an issue describing:

- The expected behavior.
- The actual behavior.
- Steps to reproduce (including relevant inputs or configuration files).

## Pull requests

When opening a pull request, please include:

- A summary of the change and the motivation.
- Any relevant screenshots or sample outputs.
- Notes on testing you performed.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you agree to uphold these expectations.
