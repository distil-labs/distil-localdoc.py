import argparse
import os
import sys
import ast
import re
from typing import List, Tuple

import argparse

from openai import OpenAI

DEFAULT_QUESTION_PROMPT_DO_NOT_CHANGE = """Provide the documentation for the function according to task description."""


class DistilLabsLLM(object):
    def __init__(self, model_name: str, api_key: str = "EMPTY", port: int = 11434):
        self.model_name = model_name
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key)

    def get_prompt(
            self,
            question: str,
            context: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": """
You are a problem solving model working on task_description XML block:
<task_description>Generate a complete Google-style docstring for the given Python function or class.

The docstring must follow Google's style guide:
- Start with a one-line summary (imperative mood: "Calculate", not "Calculates")
- Follow with a blank line and detailed description if needed
- Args section: List each parameter with type and description
- Returns section: Describe return value with type
- Raises section: List exceptions that may be raised
- Example section: Provide usage example for non-trivial functions

Format specifications:
- Use proper indentation (4 spaces)
- Parameter descriptions start with capital letter, end without period unless multiple sentences
- Type information is optional in descriptions if type hints are present
- Examples use >>> prompts for code

Input: Python function/class code with signature and body
Output: Just the complete docstring content (without the triple quotes). Do not return anything else
</task_description>
You will be given a single task with context in the context XML block and the task in the question XML block
Solve the task in question block based on the context in context block.
Generate only the answer, do not generate anything else
/no_think
""",
            },
            {
                "role": "user",
                "content": f"""

Now for the real task, solve the task in question block based on the context in context block.
Generate only the solution, do not generate anything else
<context>{context}</context>
<question>{question}</question>
/no_think
""",
            },
        ]

    def invoke(self, question: str, context: str) -> str:
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.get_prompt(question, context),
            temperature=0,
        )
        return chat_response.choices[0].message.content


def _get_source_segment(source: str, node: ast.AST) -> str:
    try:
        seg = ast.get_source_segment(source, node)
        if seg is not None:
            return seg
    except Exception:
        pass
    # Fallback using lineno/end_lineno
    try:
        lines = source.splitlines()
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(lines[start:end])
    except Exception:
        return ""


def _has_docstring(node: ast.AST) -> bool:
    try:
        return ast.get_docstring(node) is not None
    except Exception:
        return False


def _leading_ws(s: str) -> str:
    i = 0
    while i < len(s) and s[i] in (" ", "\t"):
        i += 1
    return s[:i]


def _clean_model_output(s: str) -> str:
    if s is None:
        return ""
    text = s
    # Remove any <think>...</think> blocks or stray tags
    try:
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    except Exception:
        pass
    text = text.strip()
    # Strip code fences if present
    if text.startswith("```"):
        # remove the first fence line
        lines = text.splitlines()
        # drop first line (``` or ```python)
        lines = lines[1:]
        # drop trailing fence if exists
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Remove surrounding triple quotes if model returned them
    for q in ('"""', "'''"):
        if text.startswith(q) and text.endswith(q):
            text = text[len(q):-len(q)].strip()
            break
    return text


def _build_docstring_block(content: str, indent: str) -> List[str]:
    content = content.rstrip()
    quote = '"""'
    if not content:
        return [f"{indent}{quote}\n", f"{indent}{quote}\n"]
    if "\n" not in content:
        return [f"{indent}{quote}{content}{quote}\n"]
    lines = content.splitlines()
    out = [f"{indent}{quote}\n"]
    out.extend([f"{indent}{line}\n" for line in lines])
    out.append(f"{indent}{quote}\n")
    return out


def _collect_targets(tree: ast.AST) -> List[Tuple[str, ast.AST]]:
    # Build parent mapping to distinguish top-level functions vs nested
    parent: dict[ast.AST, ast.AST] = {}
    for p in ast.walk(tree):
        for child in ast.iter_child_nodes(p):
            parent[child] = p

    targets: List[Tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = getattr(node, "name", "")
            # Skip dunder functions like __init__, __str__, etc.
            if name.startswith("__"):
                continue
            if _has_docstring(node):
                continue
            par = parent.get(node)
            if isinstance(par, (ast.Module, ast.ClassDef)):
                targets.append((f"function {name or '<anonymous>'}", node))
        # Explicitly skip ClassDef nodes (per requirement)
    # Sort by starting line to make processing predictable
    targets.sort(key=lambda t: getattr(t[1], "lineno", 0))
    return targets


def document_file(
        file_path: str,
        model: str = "localdoc_qwen3",
        port: int = 11434,
) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Failed to parse Python file: {e}", file=sys.stderr)
        sys.exit(1)

    targets = _collect_targets(tree)
    if not targets:
        # Nothing to change; return original path
        return file_path

    client = DistilLabsLLM(model_name=model, api_key="EMPTY", port=port)

    lines = source.splitlines(keepends=True)
    # ops: list of (index, kind, payload)
    # kind 'insert': payload -> List[str] (doc_lines) to insert at index
    # kind 'split': payload -> Tuple[new_sig_line:str, doc_lines:List[str], trailing_line:Optional[str]]
    ops: List[Tuple[int, str, object]] = []

    for label, node in targets:
        # Determine insertion indentation and line index
        if not getattr(node, "body", None):
            # Unexpected, skip
            continue
        first_body = node.body[0]
        node_lineno = getattr(node, "lineno", 1)
        first_body_lineno = getattr(first_body, "lineno", node_lineno)
        one_liner = (first_body_lineno == node_lineno)
        if one_liner:
            # For one-line definitions like: def f(): pass
            # Body indent should be def indent + 4 spaces
            indent = _leading_ws(lines[node_lineno - 1]) + "    "
        else:
            insert_at = max(0, first_body_lineno - 1)
            indent = _leading_ws(lines[insert_at]) if insert_at < len(lines) else (
                    _leading_ws(lines[node_lineno - 1]) + "    "
            )

        context = _get_source_segment(source, node)
        try:
            raw = client.invoke(DEFAULT_QUESTION_PROMPT_DO_NOT_CHANGE, context)
        except Exception as e:
            print(f"Model invocation failed for {label}: {e}", file=sys.stderr)
            continue
        content = _clean_model_output(raw)
        doc_lines = _build_docstring_block(content, indent)

        if one_liner:
            def_idx = node_lineno - 1
            orig = lines[def_idx]
            colon_pos = orig.find(":")
            if colon_pos == -1:
                # Fallback: insert after line (won't be a real docstring but keeps file valid)
                ops.append((def_idx + 1, "insert", doc_lines))
            else:
                new_sig = orig[:colon_pos + 1] + "\n"
                after = orig[colon_pos + 1:].rstrip("\n")
                trailing_stmt = after.strip()
                trailing_line = None
                if trailing_stmt:
                    trailing_line = indent + trailing_stmt + "\n"
                ops.append((def_idx, "split", (new_sig, doc_lines, trailing_line)))
        else:
            ops.append((insert_at, "insert", doc_lines))

    # Apply insertions bottom-up to not invalidate positions
    for idx, kind, payload in sorted(ops, key=lambda x: x[0], reverse=True):
        if kind == "insert":
            doc_lines = payload  # type: ignore
            if idx > 0 and lines[idx - 1] and not lines[idx - 1].endswith("\n"):
                lines[idx - 1] = lines[idx - 1] + "\n"
            lines[idx:idx] = doc_lines
        elif kind == "split":
            new_sig, doc_lines, trailing_line = payload  # type: ignore
            lines[idx] = new_sig
            insert_lines = list(doc_lines)
            if trailing_line:
                insert_lines.append(trailing_line)
            lines[idx + 1:idx + 1] = insert_lines

    new_source = "".join(lines)

    base, ext = os.path.splitext(file_path)
    if ext.lower() != ".py":
        out_path = f"{file_path}_documented.py"
    else:
        out_path = f"{base}_documented{ext}"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_source)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docstrings for a Python file using a local model.")
    parser.add_argument("--file", required=True, help="Path to the Python file to document")
    parser.add_argument("--model", default="localdoc_qwen3", help="Model name to use (default: localdoc_qwen3)")
    parser.add_argument("--port", type=int, default=11434, help="Local model server port (default: 11434)")
    parser.add_argument("--style", default="google", help="Docstring style (currently only 'google' supported)")
    args = parser.parse_args()

    in_path = args.file
    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Only 'google' is supported by the current prompt; warn otherwise
    if args.style.lower() != "google":
        print("Warning: only 'google' style is supported currently; using Google style.", file=sys.stderr)

    out_path = document_file(in_path, model=args.model, port=args.port)
    if out_path == in_path:
        print("No missing docstrings found. Original file left unchanged.")
    else:
        print(f"Documented file written to: {out_path}")


if __name__ == "__main__":
    main()
