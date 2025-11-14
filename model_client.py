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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="localdoc_qwen3", required=False)
    parser.add_argument("--port", type=int, default=11434, required=False)
    args = parser.parse_args()

    client = DistilLabsLLM(model_name=args.model, api_key="EMPTY", port=args.port)
    test_python_function = """from typing import List, TypeVar, Callable
import asyncio

T = TypeVar('T')

async def async_batch_processor(items: List[T], process_func: Callable[[T], T], batch_size: int = 10) -> List[T]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        tasks = [process_func(item) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results"""

    print(client.invoke(DEFAULT_QUESTION_PROMPT_DO_NOT_CHANGE, test_python_function))
