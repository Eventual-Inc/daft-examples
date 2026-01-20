# Daft Examples Test Suite

This directory contains the test suite for daft-examples, including unit tests, integration tests, and CI/CD workflows.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Markers](#test-markers)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Install Test Dependencies

```bash
# Install with test dependencies
uv pip install -e ".[test]"

# Or install test dependencies separately
uv pip install pytest pytest-timeout pytest-xdist pytest-cov python-dotenv
```

### Run All Tests

```bash
# Run all tests with parallel execution
pytest -n auto

# Run with coverage
pytest -n auto --cov=quickstart --cov=usage_patterns --cov=use_cases
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests for a specific module
pytest tests/quickstart/
```

## Test Structure

The test suite is organized to mirror the project structure:

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared pytest fixtures
└── quickstart/                  # Tests for quickstart examples
    ├── test_01_hello_world.py
    ├── test_02_semantic_search.py
    ├── test_03_data_enrichment.py
    ├── test_04_audio_file.py
    └── test_05_video_file.py
```

### Test File Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Test Organization Pattern

Each test file typically contains two test classes:

1. **Integration Tests**: Test the full example end-to-end
   - Marked with `@pytest.mark.integration`
   - Tests that examples run successfully
   - Validates output format and content
   - Checks that expected artifacts are created

2. **Unit Tests**: Test individual components
   - Marked with `@pytest.mark.unit`
   - Tests that functions/imports are available
   - Validates basic functionality
   - Fast, no external dependencies

Example structure:

```python
import pytest
import subprocess
import sys


@pytest.mark.integration
class TestExampleName:
    """Integration tests for example_name.py"""

    def test_example_runs_successfully(self, quickstart_dir):
        """Test that the example runs without errors"""
        # Test implementation

    def test_output_shows_expected_content(self, quickstart_dir):
        """Test that output displays expected content"""
        # Test implementation


@pytest.mark.unit
class TestExampleComponents:
    """Unit tests for example components"""

    def test_function_exists(self):
        """Test that required function is available"""
        # Test implementation
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run tests in parallel (using all available cores)
pytest -n auto

# Run tests with specific number of workers
pytest -n 4
```

### Test Selection

```bash
# Run tests by marker
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Exclude slow tests

# Run tests by path
pytest tests/quickstart/                    # All quickstart tests
pytest tests/quickstart/test_01_*.py       # Specific test file

# Run tests by name pattern
pytest -k "audio"           # All tests with "audio" in name
pytest -k "test_function"   # Specific test function
```

### Coverage

```bash
# Run with coverage
pytest --cov=quickstart --cov=usage_patterns --cov=use_cases

# Generate HTML coverage report
pytest --cov=quickstart --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Timeout Control

```bash
# Set custom timeout (default is 120 seconds)
pytest --timeout=300

# Disable timeout for debugging
pytest --timeout=0
```

## Test Markers

Markers categorize tests for selective execution. All markers must be registered in `pytest.ini`.

### Available Markers

- **unit**: Unit tests that don't require external dependencies or long execution time
  ```bash
  pytest -m unit
  ```

- **integration**: Integration tests that may require external services or longer execution time
  ```bash
  pytest -m integration
  ```

- **slow**: Tests that take a long time to run
  ```bash
  pytest -m "not slow"  # Skip slow tests
  ```

- **requires_credentials**: Tests that require API credentials
  ```bash
  pytest -m "not requires_credentials"  # Skip credential-dependent tests
  ```

- **requires_openai**: Tests that require OpenAI API key
  ```bash
  pytest -m "not requires_openai"
  ```

- **requires_turbopuffer**: Tests that require Turbopuffer API key
  ```bash
  pytest -m "not requires_turbopuffer"
  ```

### Combining Markers

```bash
# Run unit tests OR integration tests (union)
pytest -m "unit or integration"

# Run integration tests that don't require credentials
pytest -m "integration and not requires_credentials"

# Run all tests except slow and credential-dependent
pytest -m "not slow and not requires_credentials"
```

## Writing Tests

### Adding a New Test File

1. **Create the test file** following the naming convention:
   ```bash
   tests/quickstart/test_06_new_example.py
   ```

2. **Use the standard structure**:
   ```python
   """Tests for quickstart/06_new_example.py"""
   import pytest
   import subprocess
   import sys


   @pytest.mark.integration
   class TestNewExample:
       """Integration tests for new example"""

       def test_example_runs_successfully(self, quickstart_dir):
           """Test that the example runs without errors"""
           example_path = quickstart_dir / "06_new_example.py"

           result = subprocess.run(
               [sys.executable, "-m", "uv", "run", str(example_path)],
               capture_output=True,
               text=True,
               timeout=60
           )

           assert result.returncode == 0, f"Example failed: {result.stderr}"


   @pytest.mark.unit
   class TestNewExampleComponents:
       """Unit tests for new example components"""

       def test_required_function_exists(self):
           """Test that required function is available"""
           from daft.functions import required_function

           assert callable(required_function)
   ```

3. **Add appropriate markers** based on test characteristics

4. **Update CI workflows** if the example requires special setup

### Best Practices

1. **Use descriptive test names**: Test names should clearly indicate what is being tested
   ```python
   # Good
   def test_output_shows_video_metadata(self):

   # Bad
   def test_output(self):
   ```

2. **Test one thing per test**: Each test should verify a single behavior
   ```python
   # Good - separate tests
   def test_example_runs_successfully(self):
   def test_output_shows_metadata(self):

   # Bad - testing multiple things
   def test_everything(self):
   ```

3. **Use fixtures for common setup**: Leverage pytest fixtures in `conftest.py`
   ```python
   def test_example(self, quickstart_dir, temp_output_dir):
       # quickstart_dir and temp_output_dir are fixtures
   ```

4. **Add helpful assertion messages**: Include context in assertions
   ```python
   assert result.returncode == 0, f"Example failed: {result.stderr}"
   ```

5. **Handle external dependencies gracefully**: Skip tests when dependencies are unavailable
   ```python
   @pytest.mark.skipif(not has_openai_key(), reason="OpenAI API key not available")
   def test_with_openai(self):
   ```

### Available Fixtures

Defined in `tests/conftest.py`:

- **project_root**: Path to the project root directory
- **quickstart_dir**: Path to quickstart examples directory
- **temp_output_dir**: Temporary directory for test outputs
- **has_openai_key**: Boolean indicating if OpenAI API key is available
- **has_turbopuffer_key**: Boolean indicating if Turbopuffer API key is available
- **load_env**: Automatically loads .env file (autouse fixture)

## CI/CD Integration

### GitHub Actions Workflows

The project uses three main CI workflows:

1. **test-quickstart.yml**: Tests quickstart examples
2. **test-patterns.yml**: Tests usage patterns and use cases
3. **test-daft-cloud.yml**: Tests Daft Cloud integration

### Key CI Features

Based on Daft's CI patterns, our workflows include:

1. **Concurrency Control**: Cancels outdated PR runs
   ```yaml
   concurrency:
     group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}
     cancel-in-progress: true
   ```

2. **Skip Check**: Skips CI for doc-only changes
   ```yaml
   jobs:
     skipcheck:
       # Detects if only markdown/docs were changed
   ```

3. **Retry Logic**: Retries dependency installation on failure
   ```yaml
   - uses: nick-fields/retry@v3
     with:
       max_attempts: 3
   ```

4. **Parallel Testing**: Runs tests in parallel with pytest-xdist
   ```yaml
   - run: pytest -n auto
   ```

5. **Coverage Reporting**: Uploads coverage to Codecov
   ```yaml
   - uses: codecov/codecov-action@v4
   ```

### Running CI Locally

You can simulate CI runs locally:

```bash
# Install all dependencies
uv pip install -e ".[test]"

# Run the full test suite as CI would
pytest -n auto -m "unit or integration" --cov=quickstart --cov-report=term

# Run only fast tests (skip slow integration tests)
pytest -n auto -m "unit" --cov=quickstart --cov-report=term
```

## Troubleshooting

### Common Issues

#### 1. Tests Timeout

**Symptom**: Tests fail with `TIMEOUT` error

**Solution**:
```bash
# Increase timeout for specific test
pytest --timeout=300

# Or disable timeout while debugging
pytest --timeout=0
```

#### 2. Import Errors

**Symptom**: `ModuleNotFoundError` or `ImportError`

**Solution**:
```bash
# Ensure package is installed in editable mode
uv pip install -e .

# Verify environment
python -c "import daft; print(daft.__version__)"
```

#### 3. FFmpeg Not Found

**Symptom**: Audio/video tests fail with FFmpeg errors

**Solution**:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Verify installation
ffmpeg -version
```

#### 4. Missing API Keys

**Symptom**: Tests fail due to missing credentials

**Solution**:
```bash
# Create .env file in project root
cat > .env << EOF
OPENAI_API_KEY=your_key_here
TURBOPUFFER_API_KEY=your_key_here
EOF

# Or skip credential-dependent tests
pytest -m "not requires_credentials"
```

#### 5. Parallel Test Conflicts

**Symptom**: Tests pass individually but fail when run in parallel

**Solution**:
```bash
# Run tests sequentially for debugging
pytest -n 0

# Or use file-level parallelization
pytest -n auto --dist loadfile
```

### Debugging Tips

1. **Run with verbose output**:
   ```bash
   pytest -vv --tb=long
   ```

2. **Run specific test**:
   ```bash
   pytest tests/quickstart/test_04_audio_file.py::TestAudioFile::test_example_runs_successfully -v
   ```

3. **Show print statements**:
   ```bash
   pytest -s
   ```

4. **Drop into debugger on failure**:
   ```bash
   pytest --pdb
   ```

5. **Show local variables in traceback**:
   ```bash
   pytest -l
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the [pytest documentation](https://docs.pytest.org/)
2. Review the [Daft documentation](https://www.getdaft.io/projects/docs/)
3. Open an issue on GitHub with:
   - Test command that failed
   - Full error output
   - Python version (`python --version`)
   - Pytest version (`pytest --version`)
   - Environment info (`uv pip freeze`)

## Configuration Files

### pytest.ini

Main pytest configuration file at project root:
- Test discovery patterns
- Default options and flags
- Marker definitions
- Timeout settings
- Coverage configuration

### pyproject.toml

Project configuration including:
- Test dependencies in `[project.optional-dependencies]`
- Additional pytest options in `[tool.pytest.ini_options]`
- Coverage settings in `[tool.coverage.*]`

### conftest.py

Shared pytest fixtures and configuration:
- Session-scoped fixtures for paths
- Environment setup (auto-loads .env)
- Credential detection helpers
- Mock data fixtures

## Continuous Improvement

The test suite is continuously evolving. Contributions are welcome:

1. Add tests for new examples
2. Improve existing test coverage
3. Add new fixtures for common patterns
4. Update documentation
5. Optimize CI/CD workflows

When adding tests, ensure they:
- Follow the established patterns
- Include appropriate markers
- Have clear, descriptive names
- Include docstrings
- Run quickly (or are marked as slow)
- Clean up any created resources
