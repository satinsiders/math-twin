# Available Tools

Agents must read this document before calling any tool.

## calc_answer_tool
Evaluates a mathematical expression with given parameters using SymPy. Use for exact numeric computation of expressions and intermediate results.

## render_graph_tool
Renders a graph described by a JSON specification to a PNG file and returns the file path. Use when a problem requires a plotted graph visual.

## make_html_table_tool
Converts a JSON table specification into an HTML `<table>` string with escaped values. Use to generate table visuals for problems.

## sanitize_params_tool
Sanitizes a JSON mapping of parameters, keeping only those convertible to numeric SymPy expressions and reporting skipped keys. Use for parameter validation.

## validate_output_tool
Coerces answer fields and validates the formatter output for structural consistency. Use in QA steps to ensure final JSON correctness.

## check_asset_tool
Verifies that a referenced graph file exists or that table HTML is non-empty. Use to confirm generated assets are present.

## symbolic_solve_tool
Solves symbolic equations for specified variables and returns simplified solutions as JSON. Use to obtain exact symbolic results within operations.

