# distil-localdoc.py
<p align="center">
    <img src="logo.png" alt="drawing" width="400"/>
</p>

We trained an SLM assistant for automatic Python documentation - a Qwen3 0.6B parameter model that generates complete, properly formatted docstrings for your code in Google style. Run it locally, keeping your proprietary code secure!

## Installation

First, install [Ollama](https://ollama.com), following the instructions on their website.

Then set up the virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install huggingface_hub openai
```

Available models hosted on huggingface:
- [distil-labs/Distil-Localdoc-Qwen3-0.6B](https://huggingface.co/distil-labs/Distil-Localdoc-Qwen3-0.6B)

## Setup

Finally, download the models from huggingface and build them locally:

```bash
hf download distil-labs/Distil-Localdoc-Qwen3-0.6B --local-dir distil-model
cd distil-model
ollama create localdoc_qwen3 -f Modelfile
```

## Usage

Next, we load the model and your Python file. By default we load the downloaded Qwen3 0.6B model and generate Google-style docstrings.

```bash
python localdoc.py --file your_script.py
```

The tool will generate an updated file with `_documented` suffix (e.g., `your_script_documented.py`).

## Features

The assistant can generate docstrings for:
- **Functions**: Complete parameter descriptions, return values, and raised exceptions
- **Methods**: Instance and class method documentation with proper formatting. The tool skips double underscore (dunder: __xxx) methods.

## Examples
Feel free to run them yourself using the files in [examples](examples)

### Before:
```python
def calculate_total(items, tax_rate=0.08, discount=None):
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    if discount:
        subtotal *= (1 - discount)
    return subtotal * (1 + tax_rate)
```

### After (Google style):
```python
def calculate_total(items, tax_rate=0.08, discount=None):
    """
    Calculate the total cost of items, applying a tax rate and optionally a discount.
    
    Args:
        items: List of item objects with price and quantity
        tax_rate: Tax rate expressed as a decimal (default 0.08)
        discount: Discount rate expressed as a decimal; if provided, the subtotal is multiplied by (1 - discount)
    
    Returns:
        Total amount after applying the tax
    
    Example:
        >>> items = [{'price': 10, 'quantity': 2}, {'price': 5, 'quantity': 1}]
        >>> calculate_total(items, tax_rate=0.1, discount=0.05)
        22.5
    """
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    if discount:
        subtotal *= (1 - discount)
    return subtotal * (1 + tax_rate)
```

### Before:
```python
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []
    
    def process(self, raw_data):
        cleaned = [x for x in raw_data if x is not None]
        return [self.transform(x) for x in cleaned]
```

### After (Google style):
```python
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    def process(self, raw_data):
        """
        Calculate and return the transformed values from a list of raw data.
        
        Args:
            raw_data: List of strings to process
            None
        
        Returns:
            List of transformed strings
        
        Example:
            >>> data = ['apple', None, 'banana', 'cherry']
            >>> process(data)
            ['a', 'b', 'c']
        """
        cleaned = [x for x in raw_data if x is not None]
        return [self.transform(x) for x in cleaned]
```

### Before
```python
async def fetch_user_data(user_id, session, timeout=30):
    url = f"https://api.example.com/users/{user_id}"
    async with session.get(url, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Failed to fetch user {user_id}")
        return await response.json()
```

### After

```python
async def fetch_user_data(user_id, session, timeout=30):
    """
    Calculate user data from a user ID using an HTTP GET request.
    
    Args:
        user_id: The ID of the user to retrieve
        session: An aiohttp client session used to perform the request
        timeout: Number of seconds to wait for a response before timing out (default 30)
    
    Returns:
        The JSON‑decoded user data as a dictionary
    
    Raises:
        ValueError: If the HTTP response status is not 200
    
    Example:
        >>> import aiohttp, asyncio
        >>> async def main():
        ...     async with aiohttp.ClientSession() as session:
        ...         data = await fetch_user_data(123, session)
        ...         print(data)
        >>> asyncio.run(main())
    """
    url = f"https://api.example.com/users/{user_id}"
    async with session.get(url, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Failed to fetch user {user_id}")
        return await response.json()
```
## Using Your Own Code

Simply provide any Python file with functions or classes that need documentation:

```bash
python localdoc.py --file /path/to/your/file.py
```

The tool will:
1. Parse your Python file using AST
2. Identify all functions and methods without docstrings
3. Generate appropriate docstrings based on the code structure
4. Preserve all original code and existing docstrings
5. Output a new file with `_documented` suffix

**Note**: The tool only adds docstrings where they're missing. Existing docstrings are never modified or overwritten.

## Training & Evaluation

The tuned models were trained using knowledge distillation, leveraging the teacher model GPT-OSS-120B. The data+config+script used for finetuning can be found in [finetuning](/finetuning). We used 28 Python functions and classes as seed data and supplemented them with 10,000 synthetic examples covering various domains (data science, web development, utilities, algorithms).

We compare the teacher model and the student model on 250 held-out test examples using LLM-as-a-judge evaluation:

| Model              | Size | Accuracy      |
|--------------------|------|---------------|
| GPT-OSS (thinking) | 120B | 0.81 +/- 0.02 |
| Qwen3 0.6B (tuned) | 0.6B | 0.76 +/- 0.01 |
| Qwen3 0.6B (base)  | 0.6B | 0.55 +/- 0.04 |

**Evaluation Criteria:**
- **LLM-as-a-judge**: 
The training config file and train/test data splits are available under `data/`.

## Why Local?

**Privacy & Security**: Proprietary codebases contain intellectual property, trade secrets, and sensitive business logic. Sending your code to cloud APIs for documentation creates:
- IP exposure risks
- Compliance violations (GDPR, SOC 2, etc.)
- Security audit failures
- Dependency on external services

**Speed & Cost**: Process entire codebases in minutes without API rate limits or per-token charges.

## FAQ

**Q: Why don't we just use GPT-4/Claude API for this?**

Because your proprietary code shouldn't leave your infrastructure. Cloud APIs create security risks, compliance issues, and ongoing costs. Our models run locally with comparable quality.

**Q: Can I document existing docstrings or update them?**

Currently, the tool only adds missing docstrings. Updating existing documentation is planned for future releases. For now, you can manually remove docstrings you want regenerated.

**Q: Which docstring style can I use?**

- **Google**: Most readable, great for general Python projects

**Q: The model does not work as expected**

A: The tool calling on our platform is in active development! [Follow us on LinkedIn](https://www.linkedin.com/company/distil-labs/) for updates, or [join our community](https://join.slack.com/t/distil-labs-community/shared_invite/zt-36zqj87le-i3quWUn2bjErRq22xoE58g). You can also manually refine any generated docstrings.

**Q: Can you train a model for my company's documentation standards?**

A: Visit our [website](https://www.distillabs.ai) and reach out to us, we offer custom solutions tailored to your coding standards and domain-specific requirements.

**Q: Does this support type hints or other Python documentation tools?**

A: Type hints are parsed and incorporated into docstrings. Integration with tools like pydoc, Sphinx, and MkDocs is on our roadmap.

---

**Next Steps**: We're working on git integration to automatically document all modified functions in a commit, making documentation truly seamless in your development workflow.


---

## Links

<p align="center">
  <a href="https://www.distillabs.ai/?utm_source=github&utm_medium=referral&utm_campaign=distil-localdoc">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-distillabs-home.svg?raw=true" alt="Distil Labs Homepage" />
  </a>
  <a href="https://github.com/distil-labs">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-github.svg?raw=true" alt="GitHub" />
  </a>
  <a href="https://huggingface.co/distil-labs">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-huggingface.svg?raw=true" alt="Hugging Face" />
  </a>
  <a href="https://www.linkedin.com/company/distil-labs/">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-linkedin.svg?raw=true" alt="LinkedIn" />
  </a>
  <a href="https://distil-labs-community.slack.com/join/shared_invite/zt-36zqj87le-i3quWUn2bjErRq22xoE58g">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-slack.svg?raw=true" alt="Slack" />
  </a>
  <a href="https://x.com/distil_labs">
    <img src="https://github.com/distil-labs/badges/blob/main/badge-twitter.svg?raw=true" alt="Twitter" />
  </a>
</p>
