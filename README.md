# Minimal Agentic Demo

I built this demo to understand the tool-use loop in BFCL. I give a local Llama
model a question and two tools, execute the tool it selects, and send the result
back until the model produces a final answer.

```text
Question + tool definitions -> Llama -> tool call -> execution result
                                 ^                         |
                                 +---- message history ----+
                                 |
                                 +-> final answer
```

## Run

I use Python 3.9+ and `llama3.1:8b` running locally in Ollama. With Ollama running and the model installed, I run this command from the repository folder:

```bash
python3 demo.py
```

For a fresh Ollama setup, I download the model once with `ollama pull llama3.1:8b`.
If Ollama is not running, I start the Ollama app or run `ollama serve` in another
terminal.

The default question asks for the total price of three notebooks. The fictional
store lists a notebook at CNY 12.5, so the expected result is CNY 37.5.
The terminal shows each action, its brief purpose, the tool result, and the final
answer. The wording of the model's response may vary.

## How I implemented the loop

1. I define `get_price` and `multiply`, including their parameter descriptions
   and Python implementations, in `agent_demo/tools.py`.
2. I put the tool definitions and the question into the message history.
3. I send that history to local Ollama at `127.0.0.1:11434/api/chat`.
4. Llama returns a JSON action. For a `tool` action, I look up the named function
   in a fixed registry and execute it with the supplied arguments.
5. I append the model's action and the actual tool result to the history, then
   ask Llama for its next action.
6. I stop when Llama returns `final`. A limit of eight model calls prevents an
   endless loop. Invalid actions and tool errors are fed back within that limit.

For this question, the expected sequence is:

```text
get_price(item="notebook") -> unit_price: 12.5
multiply(a=12.5, b=3)     -> result: 37.5
final                    -> an answer based on those results
```

I do not hard-code this sequence in the loop. The model selects each action from
the question and the accumulated results. Its `reason` field is a brief action
description.

## Files

```text
demo.py                 Entry point
agent_demo/agent.py     Prompt, action parsing, and execution loop
agent_demo/models.py    Local Ollama request
agent_demo/tools.py     Tool descriptions and implementations
agent_demo/__main__.py  Console output and error handling
tests/test_agent.py     Small checks for the loop and local request format
```

I keep my reading notes and archived runs in `docs/`. Both `doc/` and `docs/` are
ignored by Git so they stay local.

## Reference

I followed the workflow in
[BFCL's BaseHandler](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py)
and its
[Llama adapter](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/llama.py).
I use JSON actions and let Ollama handle the chat template. This project only
demonstrates the workflow; it does not reproduce BFCL's benchmark or scoring.

Paper: [The Berkeley Function Calling Leaderboard (ICML 2025)](https://proceedings.mlr.press/v267/patil25a.html).

License: MIT.
