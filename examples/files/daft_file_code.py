# /// script
# description = "Extract Python functions from code files using daft.File"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.14"]
# ///

import daft
from daft import DataType, col
from daft.functions import unnest, file as daft_file

@daft.func(
    return_dtype=DataType.list(
        DataType.struct(
            {
                "name": DataType.string(),
                "signature": DataType.string(),
                "docstring": DataType.string(),
                "start_line": DataType.int64(),
                "end_line": DataType.int64(),
            }
        )
    ),
    on_error="log"
)
def extract_functions(file: daft.File):
    """Extract all function definitions from a Python file."""
    import ast
    
    with file.open() as f:
        file_content = f.read().decode("utf-8")
    
    tree = ast.parse(file_content)
    results = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = f"def {node.name}({ast.unparse(node.args)})"
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"
            
            results.append({
                "name": node.name,
                "signature": signature,
                "docstring": ast.get_docstring(node),
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            })
    
    return results


if __name__ == "__main__":
    from daft import col
    
    # Discover Python files from this repo's examples
    df = (
        daft.from_glob_path("examples/**/*.py")
        .with_column("file", daft_file(col("path")))
        .with_column("functions", extract_functions(col("file")))
        .explode("functions")
        .select("path", "size", unnest(col("functions")))
    )
    
    df.show(3)
