# My LLM Engineering Journey

> Hands-on exploration of Large Language Models — from API calls to running AI locally on my machine.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Jupyter Notebooks | Interactive experimentation |
| Groq API | Fast cloud inference (LLaMA models) |
| Ollama | Run LLMs locally — no internet, no cost |
| OpenAI SDK | Unified client for all OpenAI-compatible APIs |
| dotenv | Manage API keys securely |
| uv | Fast Python package manager |

---

## Week 1 — Foundations

### Day 1 — First Frontier LLM Project: Web Summarizer

**What I built:** A "Reader's Digest of the internet" — give it a URL, get back a summary.

**Key concepts:**
- Loading API keys securely from a `.env` file using `dotenv`
- System prompts vs user prompts — how models expect structured input
- The `messages` format: `[{"role": "system", ...}, {"role": "user", ...}]`
- Scraping website contents and passing them to an LLM
- Rendering LLM responses as formatted Markdown in the notebook

**Core code pattern:**
```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant that summarizes websites."},
    {"role": "user", "content": "Here is the website content: ..."}
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages
)
print(response.choices[0].message.content)
```

---

### Day 2 — Chat Completions API Deep Dive

**What I learned:** How the Chat Completions API actually works under the hood — and that every major provider copied it.

**Key concepts:**
- The Chat Completions API is just an HTTP POST to an endpoint
- Making raw API calls with `requests` before using the SDK wrapper
- The OpenAI SDK is just a lightweight Python wrapper around HTTP calls — it contains no model code
- **OpenAI-compatible endpoints**: Groq, Gemini, Ollama all share the same API interface
- Switching providers = just changing `base_url` and `api_key`

**Providers and their endpoints:**

| Provider | base_url | Key prefix |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `sk-proj-` |
| Groq | `https://api.groq.com/openai/v1` | `gsk_` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `AIz` |
| Ollama | `http://localhost:11434/v1` | anything |

**Running models locally with Ollama:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # can be anything
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response.choices[0].message.content)
```

**Debugging lesson:** `KeyError: 'choices'` means the API returned an error response, not a completion. Always `print(response.json())` to see the actual error.

**Jupyter kernel lesson:** The kernel keeps variables alive in memory. Stopping Ollama from the terminal doesn't clear existing response objects. Rerun the cell to confirm a service is truly stopped.

---

### Day 4 — Tokenization & The Illusion of Memory

**What I learned:** How LLMs actually process text, and why they appear to "remember" things.

**Key concepts:**

**Tokenization** — LLMs don't see words, they see tokens:
```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
tokens = encoding.encode("Hi my name is Ed and I like banoffee pie")

for token_id in tokens:
    print(f"{token_id} = {encoding.decode([token_id])}")
```

**The Illusion of Memory** — every call to an LLM is completely stateless:
```python
# This will NOT know your name — stateless!
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What's my name?"}
]

# This WILL know your name — full history passed in
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi! I'm Chetan!"},
    {"role": "assistant", "content": "Hi Chetan! How can I help?"},
    {"role": "user", "content": "What's my name?"}
]
```

**Key insight:** ChatGPT's "memory" is a trick — the entire conversation is passed in every single time. We pay for that compute on every call, and we want to, because that's what gives the model context.

---

### Day 5 — Multi-Step Agentic AI: Company Brochure Generator

**What I built:** A system that takes a company name and URL, crawls relevant pages, and generates a professional brochure.

**Key concepts:**
- **Multi-step LLM pipelines** — using one LLM call to decide which links to follow, then another to generate content
- **Structured JSON output** — forcing the model to respond in a specific JSON schema using `response_format={"type": "json_object"}`
- **One-shot prompting** — providing an example response inside the prompt to guide the model's output format
- **Streaming responses** — typewriter animation effect using `stream=True`
- First taste of **Agentic AI design patterns** — chaining multiple LLM calls together

**Streaming example:**
```python
stream = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or '', end='', flush=True)
```

**Structured JSON output:**
```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    response_format={"type": "json_object"}
)
links = json.loads(response.choices[0].message.content)
```

---

## Key Learnings from Week 1

| # | Learning |
|---|---|
| 1 | One OpenAI SDK client works for all providers — just swap `base_url` and `api_key` |
| 2 | Every LLM call is stateless — "memory" is an illusion created by passing the full conversation |
| 3 | Ollama runs powerful models 100% locally — no cost, no data leaving your machine |
| 4 | `KeyError: 'choices'` = API returned an error, not a completion — always inspect `response.json()` |
| 5 | Jupyter kernel keeps variables alive — restart and rerun all cells when debugging |
| 6 | LLMs see tokens, not words — tokenization shapes cost and context length |
| 7 | Chaining multiple LLM calls together is the foundation of Agentic AI |

---

## Project Structure

```
llm_engineering/
├── week1/
│   ├── day1.ipynb        # Web scraper + summarizer
│   ├── day2.ipynb        # Chat Completions API, Groq, Ollama
│   ├── day4.ipynb        # Tokenization + memory illusion
│   ├── day5.ipynb        # Brochure generator (multi-step agentic)
│   └── week1 EXERCISE.ipynb
├── week2/ ... week8/
├── setup/
│   └── troubleshooting.ipynb
├── .env                  # API keys — never commit this!
└── MY_JOURNEY.md
```

---

## Environment Setup

`.env` file in project root:
```
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIz...
```

> Never commit `.env` to GitHub. Make sure `.gitignore` includes `.env`.
