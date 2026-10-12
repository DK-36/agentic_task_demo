"""Small regression checks; scripted replies are only used inside tests."""

import copy
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from unittest.mock import patch
from urllib.error import URLError

from agent_demo.__main__ import main
from agent_demo.agent import AgentError, parse_decision, run_agent
from agent_demo.config import load_config

DEFAULT_QUESTION = load_config()["question"]
from agent_demo.models import LocalLlama, MODEL, URL
from agent_demo.tools import execute_tool


def call(name, **arguments):
    return json.dumps({"action": "tool", "reason": "Run the next tool.",
                       "name": name, "arguments": arguments})


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.histories = []

    def complete(self, messages, trace=None):
        self.histories.append(copy.deepcopy(messages))
        return next(self.responses)


class DemoTests(unittest.TestCase):
    def setUp(self):
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_loop_preserves_actions_and_actual_results(self):
        model = ScriptedModel([
            call("get_price", item="notebook"),
            call("multiply", a=12.5, b=3),
            '{"action":"final","answer":"CNY 37.5"}',
        ])
        self.assertEqual(run_agent(model), "CNY 37.5")
        history = model.histories[-1]
        self.assertEqual(history[1]["content"], DEFAULT_QUESTION)
        self.assertEqual([m["role"] for m in history],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        price = json.loads(history[3]["content"])["observation"]["data"]
        total = json.loads(history[5]["content"])["observation"]["data"]
        self.assertEqual(price["unit_price"], 12.5)
        self.assertEqual(total["result"], 37.5)

    def test_errors_are_returned_and_loop_is_bounded(self):
        model = ScriptedModel(["not json", call("unknown"), call("multiply", a="x", b=3)])
        with self.assertRaises(AgentError):
            run_agent(model, max_steps=3)
        for history in model.histories[1:]:
            self.assertFalse(json.loads(history[-1]["content"])["observation"]["ok"])

    def test_final_answer_stops_without_tools(self):
        model = ScriptedModel(['{"action":"final","answer":"Hello."}'])
        with patch("agent_demo.agent.execute_tool") as execute:
            self.assertEqual(run_agent(model, "Hello"), "Hello.")
            execute.assert_not_called()

    def test_tool_validation(self):
        for name, arguments in [("get_price", {}), ("get_price", {"item": "unknown"}),
                                ("multiply", {"a": True, "b": 3}),
                                ("multiply", {"a": float("nan"), "b": 3})]:
            with self.subTest(name=name, arguments=arguments):
                self.assertFalse(execute_tool(name, arguments)["ok"])

    def test_local_ollama_request(self):
        body = json.dumps({"message": {"content": '{"action":"final","answer":"ok"}'}}).encode()
        messages = [{"role": "user", "content": "Hello"}]
        with patch("agent_demo.models.urlopen", return_value=io.BytesIO(body)) as request:
            self.assertIn('"ok"', LocalLlama().complete(messages))
            sent = request.call_args.args[0]
            payload = json.loads(sent.data)
            self.assertEqual(sent.full_url, URL)
            self.assertEqual(payload["model"], MODEL)
            self.assertEqual(payload["messages"], messages)
            self.assertNotIn("format", payload)
            self.assertFalse(payload["stream"])

    def test_saved_record_matches_actual_execution(self):
        model = ScriptedModel([
            call("get_price", item="notebook"),
            call("multiply", a=12.5, b=3),
            '{"action":"final","answer":"CNY 37.5"}',
        ])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "runs" / "latest.json"
            with patch("agent_demo.__main__.RUN_LOG_PATH", path), patch(
                "agent_demo.__main__.LocalLlama", return_value=model
            ):
                self.assertEqual(main([]), 0)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["answer"], "CNY 37.5")
            self.assertEqual(len(record["steps"]), 3)
            for step in record["steps"]:
                self.assertEqual(json.loads(step["model_response"]), step["decision"])
            self.assertEqual(record["steps"][0]["observation"]["data"]["unit_price"], 12.5)
            self.assertEqual(record["steps"][1]["observation"]["data"]["result"], 37.5)
            self.assertEqual(record["steps"][2]["decision"]["answer"], record["answer"])
            self.assertEqual(record["steps"][0]["input_messages"], model.histories[0])
            self.assertEqual(record["steps"][2]["input_messages"], model.histories[2])
            self.assertEqual(len(record["steps"][0]["input_messages"]), 2)
            self.assertEqual(record["config"]["system_prompt"], model.histories[0][0]["content"])
            archives = list(path.parent.glob("run_*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(), path.read_text())

    def test_failed_run_replaces_previous_record_and_keeps_partial_steps(self):
        model = ScriptedModel([call("get_price", item="notebook")])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "latest.json"
            path.write_text('{"status":"completed","answer":"old answer"}')
            with patch("agent_demo.__main__.RUN_LOG_PATH", path), patch(
                "agent_demo.__main__.LocalLlama", return_value=model
            ), redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(main([]), 1)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("error", record)
            self.assertNotIn("answer", record)
            self.assertEqual(record["steps"][0]["observation"]["data"]["unit_price"], 12.5)
            self.assertEqual(record["steps"][1]["step"], 2)
            self.assertNotIn("model_response", record["steps"][1])
            self.assertEqual(len(record["steps"][1]["input_messages"]), 4)

    def test_interrupted_run_is_saved(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "latest.json"
            with patch("agent_demo.__main__.RUN_LOG_PATH", path), patch(
                "agent_demo.__main__.LocalLlama.complete", side_effect=KeyboardInterrupt
            ), patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(main([]), 130)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(len(record["steps"]), 1)
            self.assertEqual(record["steps"][0]["step"], 1)
            self.assertEqual(len(record["steps"][0]["input_messages"]), 2)

    def test_prompt_and_tool_files_are_reloaded_and_disabled_tools_do_not_run(self):
        with TemporaryDirectory() as folder:
            prompt, question, tools = [Path(folder) / name for name in ("system.txt", "question.txt", "tools.json")]
            prompt.write_text("First prompt. Tools: {{tools}}")
            question.write_text("Multiply 2 by 3.")
            multiply_spec = next(t for t in load_config()["tools"] if t["name"] == "multiply")
            tools.write_text(json.dumps([multiply_spec]))
            config = load_config(prompt, question, tools)
            self.assertIn("First prompt", config["system_prompt"])
            prompt.write_text("Changed prompt. Tools: {{tools}}")
            config = load_config(prompt, question, tools)
            model = ScriptedModel([call("get_price", item="notebook"),
                                   '{"action":"final","answer":"Cannot use that tool."}'])
            with patch("agent_demo.tools.REGISTRY", {"get_price": lambda **kwargs: self.fail("Disabled tool executed")}):
                run_agent(model, config=config)
            self.assertTrue(model.histories[0][0]["content"].startswith("Changed prompt"))
            self.assertEqual(model.histories[0][1]["content"], "Multiply 2 by 3.")
            feedback = json.loads(model.histories[1][-1]["content"])
            self.assertFalse(feedback["observation"]["ok"])
            tools.write_text('[{"name":"unregistered"}]')
            with self.assertRaises(ValueError):
                load_config(prompt, question, tools)

    def test_cli_accepts_prompt_question_and_tools_files(self):
        with TemporaryDirectory() as folder:
            prompt, question, tools = [Path(folder) / name for name in ("system.txt", "question.txt", "tools.json")]
            prompt.write_text("Answer directly. Available tools: {{tools}}")
            question.write_text("Say hello.")
            tools.write_text("[]")
            model = ScriptedModel(['{"action":"final","answer":"Hello."}'])
            path = Path(folder) / "runs/latest.json"
            with patch("agent_demo.__main__.RUN_LOG_PATH", path), patch(
                "agent_demo.__main__.LocalLlama", return_value=model
            ):
                self.assertEqual(main(["--prompt", str(prompt), "--question", str(question), "--tools", str(tools)]), 0)
            record = json.loads(path.read_text())
            self.assertEqual(record["config"]["sources"]["prompt"]["text"], prompt.read_text())
            self.assertEqual(record["config"]["tools"], [])
            self.assertEqual(record["question"], "Say hello.")

    def test_original_request_and_response_are_preserved_before_action_parsing(self):
        content = "  <tool>unparseable output</tool>\n"
        body = json.dumps({"message": {"role": "assistant", "content": content}, "eval_count": 7})
        steps = []
        with patch("agent_demo.models.urlopen", return_value=io.BytesIO(body.encode())) as request:
            with self.assertRaises(AgentError):
                run_agent(LocalLlama(), max_steps=1, steps=steps)
        entry = steps[0]
        self.assertEqual(entry["response_body"], body)
        self.assertEqual(entry["model_response"], content)
        self.assertEqual(entry["request"]["body"].encode(), request.call_args.args[0].data)
        self.assertEqual(json.loads(entry["request"]["body"])["messages"], entry["input_messages"])
        self.assertFalse(entry["observation"]["ok"])
        self.assertNotIn("decision", entry)

    def test_tagged_workflow_saves_original_and_extracted_output(self):
        outputs = [
            'I will check the price.\n<tool>\n{"name":"get_price","arguments":{"item":"notebook"}}\n</tool>',
            'I will calculate the total.\n<tool>{"name":"multiply","arguments":{"a":12.5,"b":3}}</tool>',
            '<final>The total is CNY 37.5.</final>',
        ]
        model = ScriptedModel(outputs)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "latest.json"
            with patch("agent_demo.__main__.RUN_LOG_PATH", path), patch(
                "agent_demo.__main__.LocalLlama", return_value=model
            ):
                self.assertEqual(main([]), 0)
            record = json.loads(path.read_text())
            self.assertEqual(record["answer"], "The total is CNY 37.5.")
            self.assertEqual([s["model_response"] for s in record["steps"]], outputs)
            first = record["steps"][0]
            self.assertEqual(first["tool_call_text"], '\n{"name":"get_price","arguments":{"item":"notebook"}}\n')
            self.assertEqual(first["decision"]["reason"], "I will check the price.")
            self.assertEqual(first["observation"]["data"]["unit_price"], 12.5)
            self.assertEqual(record["steps"][1]["observation"]["data"]["result"], 37.5)
            self.assertEqual(model.histories[1][2]["content"], outputs[0])

    def test_ambiguous_or_incomplete_tags_never_execute(self):
        tool = '<tool>{"name":"multiply","arguments":{"a":2,"b":3}}</tool>'
        invalid = [tool + tool, tool + '<final>6</final>',
                   tool.replace('</tool>', ''), '<tool>{broken}</tool>',
                   '<tool>[]</tool>', '<tool>{"action":"final","answer":"done"}</tool>',
                   '<final></final>', tool + 'extra text',
                   '<tool>{"name":"multiply","arguments":[]}</tool>']
        for output in invalid:
            with self.subTest(output=output), patch("agent_demo.agent.execute_tool") as execute:
                model = ScriptedModel([output, '<final>Please clarify.</final>'])
                steps = []
                self.assertEqual(run_agent(model, steps=steps), "Please clarify.")
                execute.assert_not_called()
                self.assertEqual(steps[0]["model_response"], output)
                self.assertFalse(steps[0]["observation"]["ok"])

    def test_final_block_allows_surrounding_explanation(self):
        raw = "The tools returned 37.5.\n<final>The total is CNY 37.5.</final>\nDone."
        model = ScriptedModel([raw])
        steps = []
        self.assertEqual(run_agent(model, steps=steps), "The total is CNY 37.5.")
        self.assertEqual(steps[0]["model_response"], raw)
        self.assertEqual(steps[0]["decision"]["answer"], "The total is CNY 37.5.")

    def test_tagged_and_legacy_json_outputs_are_supported(self):
        decision = parse_decision('<tool>{"name":"multiply","arguments":{"a":2,"b":3}}</tool>')
        self.assertEqual(decision["action"], "tool")
        self.assertEqual(parse_decision('<final>6</final>'), {"action": "final", "answer": "6"})
        self.assertEqual(parse_decision(call("multiply", a=2, b=3))["arguments"], {"a": 2, "b": 3})

    def test_invalid_service_response_is_preserved(self):
        trace = {}
        with patch("agent_demo.models.urlopen", return_value=io.BytesIO(b"not JSON")):
            with self.assertRaises(RuntimeError):
                LocalLlama().complete([], trace=trace)
        self.assertEqual(trace["response_body"], "not JSON")
        self.assertIn("request", trace)

    def test_service_failure_has_no_fallback(self):
        for body in (b"{}", b"null", b"not json", b'{"message":{"content":""}}'):
            with self.subTest(body=body), patch("agent_demo.models.urlopen", return_value=io.BytesIO(body)):
                with self.assertRaises(RuntimeError):
                    LocalLlama().complete([])
        with patch("agent_demo.models.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(RuntimeError):
                LocalLlama().complete([])


if __name__ == "__main__":
    unittest.main()
