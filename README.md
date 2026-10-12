# Minimal Agentic Demo

I built this demo to understand the tool-use loop in BFCL. I give a local Llama
model a question and tools, execute the tool it selects, and send the result
back until the model produces a final answer.

```text
Question + tool definitions -> Llama -> tool call -> execution result
                                 ^                         |
                                 +---- message history ----+
                                 |
                                 +-> final answer
```

## Run

I use Python 3.9+ and `llama3.1:8b` running locally in Ollama. With Ollama running
and the model installed, I run this command from the repository folder:

```bash
python3 demo.py
```

For a fresh Ollama setup, I download the model once with `ollama pull llama3.1:8b`.
If Ollama is not running, I start the Ollama app or run `ollama serve` in another
terminal. No additional Python packages are required.

The default question asks for the total price of three notebooks. The fictional
store lists a notebook at CNY 12.5, so the expected result is CNY 37.5.
The terminal shows the selected tools, results, and final answer.

## Editable prompts and tools

I keep the task content separate from the execution loop:

| File | What I edit |
| --- | --- |
| `prompt_library/system.txt` | Instructions and the tagged output format |
| `prompt_library/question.txt` | The user's question |
| `tool_library/definitions.json` | Tool descriptions, parameter definitions, and which tools are available |
| `tool_library/functions.py` | Python implementations and the `REGISTRY` mapping |

The files are read at the start of each run, so edits take effect on the next
run. The system prompt contains one `{{tools}}` placeholder, which is replaced
with the selected tool definitions. I keep the `<tool>` / `<final>` format
when changing the wording of the default prompt.

I write the prompt around the current stationery-store task: look up a unit
price and calculate a purchase total. It contains the task context, output
format, and tool definitions. Validation, error handling, and step limits live
in the code.

I can also keep several prompt or tool-definition files and select them without
editing the loop. For example, after creating `prompt_library/alternative.txt`:

```bash
python3 demo.py --prompt prompt_library/alternative.txt
```

The optional `--question` and `--tools` arguments likewise accept file paths.
Omitting these arguments uses the defaults listed above. Tool definitions may
select a subset of registered functions; disabled tools cannot be executed.
To add a new tool, I add its function and registry entry in `functions.py`, then
add its description in the selected definitions file. I update any task-specific
instructions in the prompt as needed.

## Tool-tag extraction

My default prompt asks for a short action description followed by one tool block:

```text
I will look up the price.
<tool>{"name":"get_price","arguments":{"item":"notebook"}}</tool>
```

I save the full output first, extract the text between `<tool>` and `</tool>`,
parse the enclosed JSON, and execute the selected function. The complete original
output, including the action description and tags, stays in the conversation.
The action description is not an internal reasoning trace.

When the task is complete, I extract the answer from its final block. The parser
allows surrounding explanation and keeps the full original output in the log:

```text
<final>The total price is 37.5 CNY.</final>
```

I reject incomplete tags, multiple tool blocks, and mixed tool/final outputs
before executing anything. The error is returned to the model so it can correct
its next response. Ollama's forced JSON output mode is disabled because the full
response now contains text and tags; the content inside a tool block is still JSON.

The parser also accepts JSON-only actions. I keep a JSON-format version of the
same store task in `prompt_library/json.txt` for comparison:

```bash
python3 demo.py --prompt prompt_library/json.txt
```

## Complete run records

I save each run to `runs/run_<UTC timestamp>.json` and refresh `runs/latest.json`
with the same content. This keeps previous experiments available for comparison.
Both successful and failed or interrupted runs save the information collected
so far. These are actual runtime records, not copies of the old examples.

Each record contains:

- `config`: original prompt, question, and tool-definition file contents and
  paths, the parsed tool definitions, and the assembled system prompt.
- `steps[].input_messages`: a separate snapshot of the complete conversation
  sent at that step, including previous actions and tool results.
- `steps[].request`: the URL and exact JSON request body sent to local Ollama,
  including the model and generation settings.
- `steps[].response_body`: the original Ollama response body before JSON parsing,
  including response metadata returned by the service.
- `steps[].model_response`: the original assistant content before action parsing.
- `steps[].tool_call_text`: the exact text extracted from a tool block, before
  JSON parsing (also retained when that JSON is invalid).
- `steps[].decision` and `observation`: the parsed action and actual tool result.
- The final answer or error, status, and start/end timestamps.

If a request fails before a response arrives, that step has no response body.
If action parsing fails, its original output is still saved. The extracted
tool text and parsed action are additional fields, not replacements for the
original output. Other output protocols would need corresponding parser changes.

The saved input is the complete request sent by this client. Ollama applies the
model's chat template internally; its final tokenized prompt and model weights
are not exported by this demo. A `completed` status means the model returned an
answer, not that a benchmark verified its correctness.

## How I implemented the loop

1. I load the prompt, question, and selected tool definitions.
2. I insert the definitions into the prompt and build the initial messages.
3. I save an input snapshot and send the request to local Ollama.
4. I preserve the original response, extract its tool block, and parse the JSON inside.
5. For a `tool` action, I execute a permitted function and append the actual
   result to the conversation. Then I repeat the model call.
6. A `<final>` block ends the loop. I allow at most eight model calls and save the
   complete record when the run finishes.

For the default question, the expected sequence is:

```text
get_price(item="notebook") -> unit_price: 12.5
multiply(a=12.5, b=3)     -> result: 37.5
final                    -> an answer based on those results
```

I do not hard-code that sequence. The model selects its actions from the question
and previous results. I store its brief action description in the parsed `reason` field.

## Files

```text
demo.py                  Entry point
prompt_library/          Editable system prompt and question
tool_library/            Tool definitions and Python functions
agent_demo/config.py     Load and validate prompt/tool files
agent_demo/agent.py      Model/tool loop and action parser
agent_demo/models.py     Local Ollama request and original response capture
agent_demo/tools.py      Dispatch enabled tools
agent_demo/__main__.py   Optional file arguments and run-record saving
tests/test_agent.py      Regression checks
runs/                    Actual run records (generated locally)
```

I keep `runs/`, `doc/`, and `docs/` ignored by Git. My prompt and tool libraries
are part of the repository. Earlier JSON files in `docs/archive/` remain local
historical records and are not read by the demo.

## Reference

I followed the workflow in
[BFCL's BaseHandler](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py)
and its
[Llama adapter](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/llama.py).
I use tagged tool calls with JSON payloads and let Ollama handle the chat template. This project only
demonstrates the workflow; it does not reproduce BFCL's benchmark or scoring.

Paper: [The Berkeley Function Calling Leaderboard (ICML 2025)](https://proceedings.mlr.press/v267/patil25a.html).

License: MIT.
