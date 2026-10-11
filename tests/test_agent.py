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
from agent_demo.agent import AgentError, DEFAULT_QUESTION, run_agent
from agent_demo.models import LocalLlama, MODEL, URL
from agent_demo.tools import execute_tool


def call(name, **arguments):
    return json.dumps({"action": "tool", "reason": "Run the next tool.",
                       "name": name, "arguments": arguments})


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.histories = []

    def complete(self, messages):
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
            self.assertEqual(payload["format"], "json")
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
                self.assertEqual(main(), 0)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["answer"], "CNY 37.5")
            self.assertEqual(len(record["steps"]), 3)
            for step in record["steps"]:
                self.assertEqual(json.loads(step["model_response"]), step["decision"])
            self.assertEqual(record["steps"][0]["observation"]["data"]["unit_price"], 12.5)
            self.assertEqual(record["steps"][1]["observation"]["data"]["result"], 37.5)
            self.assertEqual(record["steps"][2]["decision"]["answer"], record["answer"])

    def test_failed_run_replaces_previous_record_and_keeps_partial_steps(self):
        model = ScriptedModel([call("get_price", item="notebook")])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "latest.json"
            path.write_text('{"status":"completed","answer":"old answer"}')
            with patch("agent_demo.__main__.RUN_LOG_PATH", path), patch(
                "agent_demo.__main__.LocalLlama", return_value=model
            ), redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(main(), 1)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("error", record)
            self.assertNotIn("answer", record)
            self.assertEqual(record["steps"][0]["observation"]["data"]["unit_price"], 12.5)
            self.assertEqual(record["steps"][1], {"step": 2})

    def test_interrupted_run_is_saved(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "latest.json"
            with patch("agent_demo.__main__.RUN_LOG_PATH", path), patch(
                "agent_demo.__main__.LocalLlama.complete", side_effect=KeyboardInterrupt
            ), patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(main(), 130)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(record["steps"], [{"step": 1}])

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
