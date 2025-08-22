# Math Twin Generator

Generate SAT-style "twin" math problems from a reference problem and official solution. The project exposes both a Python API and a command line interface built on a linear pipeline of large-language-model (LLM) "agents" and utility tools.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Command Line Interface](#command-line-interface)
5. [Python API](#python-api)
6. [Pipeline Overview](#pipeline-overview)
7. [Output Format](#output-format)
8. [Extending the System](#extending-the-system)
9. [Development](#development)
10. [Project Layout](#project-layout)
11. [Contributing](#contributing)

## Overview

**Math Twin** creates new multiple‑choice problems that target the same underlying concept as a source question while varying surface features. Each stage in the pipeline uses an LLM agent with a narrowly scoped prompt:

- **ParserAgent** – extract structured data from the source problem and solution.
- **ConceptAgent** – identify the key concept and outline a canonical solution path.
- **TemplateAgent** – replace literals with symbolic parameters and record any visual requirements.
- **SampleAgent** – sample concrete parameters that satisfy the template.
- **OperationsAgent** – invoke registered tools to compute intermediate results.
- **StemChoiceAgent** – draft the new SAT‑style question and answer choices.
- **FormatterAgent** – produce the final minified JSON payload.
- **QAAgent** – validate JSON formatting and internal consistency.

Optional agents such as **SymbolicSolveAgent** and **SymbolicSimplifyAgent** handle heavy symbolic manipulation when required.

## Installation

This project requires **Python 3.12+**. Install in editable mode with development dependencies:

```bash
pip install -e .
```

If you plan to run the test suite or type checks you will also need the dev requirements:

```bash
pip install -r requirements-dev.txt
```

## Quick Start

Set your OpenAI API key and invoke the demo generator:

```bash
export OPENAI_API_KEY=your-key-here
python -m twin_generator.cli --demo
```

Use `--graph-demo` to generate a problem that includes a graph visual. Append `--preview` to automatically open the generated PNG.

## Command Line Interface

The CLI wrapper lives at `twin_generator/cli.py` and mirrors the Python API. Key options include:

| Flag | Description |
| ---- | ----------- |
| `--problem PATH` | Path to source problem text. |
| `--solution PATH` | Path to official solution text. |
| `--demo` | Run a trivial built‑in demo problem. |
| `--graph-demo` | Run the demo that produces a graph. |
| `--out PATH` | Write the resulting JSON to a file instead of stdout. |
| `--preview` | Display the generated graph image if present. |
| `--log-level LEVEL` | One of `WARNING`, `INFO`, or `DEBUG`. |

Example invocation with your own files:

```bash
python -m twin_generator.cli --problem path/to/problem.txt --solution path/to/solution.txt --out twin.json --log-level INFO
```

## Python API

Import the package and call :pyfunc:`twin_generator.generate_twin` directly. The helper returns a [`PipelineState`](twin_generator/pipeline_state.py) dataclass:

```python
from twin_generator import generate_twin

state = generate_twin(problem_text, solution_text)
print(state.twin_stem)
print(state.choices)
```

Pass `force_graph=True` or `graph_spec=...` to require a visual. The returned state also contains `graph_path`, `table_html`, and other intermediate fields for advanced workflows.

## Pipeline Overview

Internally the generator constructs a simple directed graph of step functions and executes them sequentially. The orchestration logic lives in [`twin_generator/pipeline.py`](twin_generator/pipeline.py) and [`twin_generator/pipeline_runner.py`](twin_generator/pipeline_runner.py). Each step receives and returns a `PipelineState` instance, enabling early exits or retries. JSON‑producing steps are automatically re‑run until valid JSON is obtained or the retry limit is reached.

The default pipeline performs the following high‑level operations:

1. Parse source problem/solution into structured fields.
2. Extract the underlying concept and canonical steps.
3. Parameterize the problem template and declare domains.
4. Sample concrete parameters and compute any symbolic results.
5. Run any registered tool operations.
6. Draft the final stem and answer choices.
7. Minify and validate the JSON payload.
8. Optionally preview or persist any generated visuals.

## Output Format

The final JSON structure mirrors the fields of `PipelineState` and includes at minimum:

```json
{
  "twin_stem": "...problem text...",
  "choices": ["A", "B", "C", "D"],
  "answer_index": 1,
  "answer_value": "B",
  "rationale": "Explanation of why B is correct",
  "graph_path": "optional/path.png",
  "table_html": "<table>...</table>"
}
```

Intermediate fields such as `parsed`, `concept`, `template`, `params`, `symbolic_solution`, and more are also preserved on the `PipelineState` object for debugging or downstream consumption.

## Extending the System

- **Agents** – Agent definitions live in [`twin_generator/agents.py`](twin_generator/agents.py). Each agent specifies a name, prompt instructions, and optional model.
- **Tools** – Use [`agents.tool.function_tool`](agents/tool.py) to expose a Python function as a callable tool. Type hints are converted into the JSON schema understood by OpenAI's tool‑calling interface.
- **Templates** – See [`docs/template_agent.md`](docs/template_agent.md) for the JSON schema produced by the `TemplateAgent`.

### OperationsAgent

Each item in the `operations` array may call a registered tool by including a
`"tool"` field.  The newly added `symbolic_solve_tool` finds exact solutions to
symbolic equations using SymPy.

```json
{
  "tool": "symbolic_solve_tool",
  "eq_json": "{\"equation\": \"x**2 - 1\", \"variable\": \"x\"}",
  "output": "roots"
}
```

The `eq_json` string must encode an object with `equation` (the equation or
expression, optional `=` sign allowed) and `variable` (single symbol name or
array of names). The tool returns a JSON array of solution dictionaries with
simplified expressions.

You can register additional tools or swap out agents to customize the pipeline for different domains.

## Development

Run the test suite and type checker:

```bash
pytest
mypy --config-file=mypy.ini twin_generator
```

Formatting and linting are enforced via [pre‑commit](https://pre-commit.com/):

```bash
pre-commit run --files path/to/changed_file.py
```

## Project Layout

```
math-twin/
├── agents/              # Minimal Agent and tool helpers used by the pipeline
├── docs/                # Additional documentation and schemas
├── tests/               # Pytest suite covering pipeline behavior
├── twin_generator/      # Core package
│   ├── cli.py           # Command line interface entry point
│   ├── pipeline.py      # High-level pipeline orchestration
│   ├── pipeline_runner.py  # Step execution engine
│   ├── pipeline_steps.py   # Individual step functions
│   ├── pipeline_state.py   # Dataclass capturing inputs, intermediates, and outputs
│   └── agents.py        # LLM agent definitions
└── README.md            # This file
```

## Contributing

Issues and pull requests are welcome! If you add or modify pipeline steps, please include accompanying tests and ensure `pytest`, `mypy`, and `pre-commit` all pass before submitting.

