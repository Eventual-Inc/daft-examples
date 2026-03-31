# /// script
# description = "take a bunch of filepaths, filter for Python files, extract Python functions, caption them all/generate a docstring, run embeddings."
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.6", "numpy", "python-dotenv"]
# ///

from pydantic import BaseModel

import daft
from daft import DataType

FUNCTION_SCHEMA = DataType.struct(
    {
        "name": DataType.string(),
        "code": DataType.string(),
        "docstring": DataType.string(),
        "start_line": DataType.int64(),
        "end_line": DataType.int64(),
        "decorators": DataType.list(DataType.string()),
        "signature": DataType.string(),
        "body": DataType.string(),
        "is_async": DataType.bool(),
    }
)

CLASSES_SCHEMA = DataType.list(
    DataType.struct(
        {
            "name": DataType.string(),
            "code": DataType.string(),
            "docstring": DataType.string(),
            "start_line": DataType.int64(),
            "end_line": DataType.int64(),
            "decorators": DataType.list(DataType.string()),
            "bases": DataType.list(DataType.string()),
            "methods": DataType.list(FUNCTION_SCHEMA),
        }
    )
)


def _extract_function_metadata(node, file_content):
    import ast

    # Get the source code segment
    code_segment = ast.get_source_segment(file_content, node)

    # Get the docstring
    docstring = ast.get_docstring(node)

    # Get decorators
    decorators = [ast.get_source_segment(file_content, d) for d in node.decorator_list]

    # Get signature
    signature = f"def {node.name}({ast.unparse(node.args)})"
    if node.returns:
        signature += f" -> {ast.unparse(node.returns)}"

    # Get body
    # We want the code starting from the first statement in the body
    body = ""
    if node.body:
        start_line = node.body[0].lineno
        end_line = node.end_lineno
        # Split file content into lines (0-indexed list, but lineno is 1-indexed)
        lines = file_content.splitlines()
        # Extract lines from start_line-1 to end_line
        body_lines = lines[start_line - 1 : end_line]
        body = "\n".join(body_lines)

    return {
        "name": node.name,
        "code": code_segment,
        "docstring": docstring,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "decorators": decorators,
        "signature": signature,
        "body": body,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
    }


@daft.func(return_dtype=CLASSES_SCHEMA, on_error="log")
def extract_classes(
    file: daft.File,
):
    """retrieve all classes (with their methods) from the file"""
    import ast

    with file.open() as f:
        file_content = f.read().decode("utf-8")

    tree = ast.parse(file_content)

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get the source code segment
            code_segment = ast.get_source_segment(file_content, node)

            # Get the docstring
            docstring = ast.get_docstring(node)

            # Get decorators
            decorators = [ast.get_source_segment(file_content, d) for d in node.decorator_list]

            # Get bases
            bases = [ast.unparse(b) for b in node.bases]

            # Get methods
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(_extract_function_metadata(child, file_content))

            results.append(
                {
                    "name": node.name,
                    "code": code_segment,
                    "docstring": docstring,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "decorators": decorators,
                    "bases": bases,
                    "methods": methods,
                }
            )

    return results


@daft.func(return_dtype=DataType.list(FUNCTION_SCHEMA), on_error="log")
def extract_functions(file: daft.File):
    """retrieve all functions from the file"""
    import ast

    with file.open() as f:
        file_content = f.read().decode("utf-8")

    tree = ast.parse(file_content)

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            results.append(_extract_function_metadata(node, file_content))

    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    from daft import col, lit
    from daft.functions import embed_text, file, prompt, unnest

    load_dotenv()

    class Caption(BaseModel):
        source_file: str
        code: str
        docstring: str
        code_type: str

    repo = "../../**/*.py"

    df = daft.from_glob_path(repo).with_column("file", file(col("path")))

    classes_df = (
        df.with_column("classes", extract_classes(col("file")))
        .explode("classes")
        .select("path", unnest(col("classes")))
    )

    methods_df = (
        classes_df.select(col("path"), col("name").alias("class_name"), col("methods"))
        .explode("methods")
        .select(
            col("path"),
            col("class_name"),
            col("methods").struct.get("name").alias("name"),
            col("methods").struct.get("signature"),
            col("methods").struct.get("docstring"),
            col("methods").struct.get("body"),
        )
    )

    # Caption and Embed Methods
    methods_df = (
        methods_df.with_column(
            "prompt_input",
            lit("Explain what this Python method does in one concise sentence. Focus on the purpose and logic.\n\n")
            + lit("Method: ")
            + col("name").fill_null("")
            + lit("\nSignature: ")
            + col("signature").fill_null("")
            + lit("\nDocstring: ")
            + col("docstring").fill_null("")
            + lit("\nCode:\n")
            + col("body").fill_null(""),
        )
        .with_column(
            "caption",
            prompt(col("prompt_input"), model="gpt-4o-mini", provider="openai"),
        )
        .with_column(
            "embedding",
            embed_text(col("caption"), model="text-embedding-3-small", provider="openai"),
        )
    )

    functions_df = (
        df.with_column("functions", extract_functions(col("file")))
        .explode("functions")
        .select("path", unnest(col("functions")))
    )

    # Caption and Embed Functions
    functions_df = (
        functions_df.with_column(
            "prompt_input",
            lit("Explain what this Python function does in one concise sentence. Focus on the purpose and logic.\n\n")
            + lit("Function: ")
            + col("name").fill_null("")
            + lit("\nSignature: ")
            + col("signature").fill_null("")
            + lit("\nDocstring: ")
            + col("docstring").fill_null("")
            + lit("\nCode:\n")
            + col("body").fill_null(""),
        )
        .with_column(
            "caption",
            prompt(col("prompt_input"), model="gpt-4o-mini", provider="openai"),
        )
        .with_column(
            "embedding",
            embed_text(col("caption"), model="text-embedding-3-small", provider="openai"),
        )
    )

    print("Classes:")
    classes_df.show()

    print("Methods (with Captions and Embeddings):")
    methods_df.show()

    print("Functions (with Captions and Embeddings):")
    functions_df.show()
