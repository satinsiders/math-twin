# Math Twin Generator

Generate SAT-style "twin" math problems from a source question and official solution. The project exposes a Python API and command line interface.

## Installation

This project requires Python 3.12+. Install the package with its dependencies in editable mode:

```bash
pip install -e .
```

## Running

Set your OpenAI API key in the environment:

```bash
export OPENAI_API_KEY=your-key-here
```

Run the generator using the provided command line interface. You can supply your own problem and solution files or use the built-in demos:

```bash
# With your own files
python -m twin_generator.cli --problem path/to/problem.txt --solution path/to/solution.txt --out twin.json

# Run a small demo
python -m twin_generator.cli --demo
```

Passing `--graph-demo` runs a demo that produces a graph visual. Append `--preview` to automatically preview the generated graph if any. Use `--log-level INFO` (or `DEBUG`) to see each pipeline step.

## Tool annotation support

The `agents.tool.function_tool` helper derives a JSON schema for the parameters of a wrapped function using its type hints. Supported annotation patterns include:

- `int` and `float` → `"number"`
- `bool` → `"boolean"`
- `str` → `"string"`
- container types such as `list[T]`, `tuple[T, ...]`, and `set[T]` → `"array"`
- mappings like `dict[K, V]` → `"object"`
- `Optional[T]` is treated the same as `T`
- `Union` is supported when all members resolve to the same JSON schema type; heterogeneous unions fall back to `"string"`

## Development

Run tests with `pytest` and type checking with `mypy`:

```bash
pytest
mypy --config-file=mypy.ini twin_generator
```

Formatting and linting are handled by `flake8` and `mypy` via pre-commit configuration.
