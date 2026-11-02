from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from unittest import mock

from workflow import llm


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LlmLogTests(unittest.TestCase):
    def test_every_call_is_logged_without_the_key(self):
        answer = {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                  "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"ok\": 1}"}]}]}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"WORKFLOW_LLM_LOG_DIR": folder, "WORKFLOW_LLM_TAG": "room_007"}), \
                mock.patch.object(llm.urllib.request, "urlopen", lambda *a, **k: _Answer(json.dumps(answer).encode())):
            text, call = llm.ask("do X", "the room", model="gpt-5.6-sol", base_url="https://api.openai.com/v1",
                                 api_key="sk-secret-test-key")
            raw = open(os.path.join(folder, "room_007.jsonl"), encoding="utf-8").read()
        self.assertEqual(text, "{\"ok\": 1}")
        record = json.loads(raw)
        self.assertEqual((record["tag"], record["instructions"], record["prompt"], record["response"], record["ok"]),
                         ("room_007", "do X", "the room", "{\"ok\": 1}", True))
        self.assertEqual(record["usage"]["prompt_tokens"], 10)
        self.assertNotIn("sk-secret-test-key", raw)
        self.assertNotIn("Authorization", raw)

    def test_nothing_is_written_without_the_setting(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORKFLOW_LLM_LOG_DIR", None)
            llm._log({"prompt": "x"})
            self.assertEqual(os.listdir(folder), [])


class RelayAndRateTests(unittest.TestCase):
    def test_relay_settings_are_read_in_place_and_hosts_checked(self):
        from workflow import credentials
        with tempfile.TemporaryDirectory() as folder:
            env = os.path.join(folder, "relay.env")
            with open(env, "w") as f:
                f.write("OPENAI_API_KEY=sk-test-relay\nOPENAI_MODEL=gpt-5.6-sol\nOPENAI_BASE_URL=https://old.example/codex/v1\n")
            with mock.patch.object(credentials, "RELAY_ENV_FILE", credentials.Path(env)):
                s = credentials.api_settings("relay")
                self.assertEqual((s["base_url"], s["api_key"], s["model"]),
                                 ("https://api.kbsfdxc.net/codex/v1", "sk-test-relay", "gpt-5.6-sol"))
                self.assertEqual(credentials.api_settings("relay", "https://api.kbsfdxc.com/codex/v1/")["base_url"],
                                 "https://api.kbsfdxc.com/codex/v1")
                with self.assertRaises(credentials.FatalAPIError):
                    credentials.api_settings("relay", "https://evil.example.com/v1")

    def test_the_rate_limit_waits_when_the_minute_is_full(self):
        waits = []
        with tempfile.TemporaryDirectory() as folder:
            state = os.path.join(folder, "rpm.json")
            with mock.patch.dict(os.environ, {"WORKFLOW_RPM": "2", "WORKFLOW_RPM_FILE": state}):
                llm._throttle()
                llm._throttle()
                self.assertEqual(len(json.load(open(state))), 2)

                def fake_sleep(seconds):  # the third request waits; pretend the minute passed
                    waits.append(seconds)
                    with open(state, "w") as f:
                        f.write("[]")
                with mock.patch.object(llm.time, "sleep", fake_sleep):
                    llm._throttle()
        self.assertEqual(len(waits), 1)
        self.assertGreater(waits[0], 50)

    def test_the_relay_gets_chat_completions_with_a_system_message(self):
        responses = {"status": "completed", "usage": {},
                     "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
        chat = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
        sent = []

        def fake_urlopen(request, timeout=None):
            sent.append((request.full_url, json.loads(request.data)))
            return _Answer(json.dumps(chat if request.full_url.endswith("/chat/completions") else responses).encode())
        with mock.patch.object(llm.urllib.request, "urlopen", fake_urlopen), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORKFLOW_RPM", None)
            llm.ask("do X", "room", model="gpt-5.6-sol", base_url="https://api.kbsfdxc.net/codex/v1", api_key="k")
            llm.ask("do X", "room", model="gpt-5.6-sol", base_url="https://api.openai.com/v1", api_key="k")
        (relay_url, relay), (official_url, official) = sent
        self.assertTrue(relay_url.endswith("/codex/v1/chat/completions"))
        self.assertEqual(relay["messages"], [{"role": "system", "content": "do X"}, {"role": "user", "content": "room"}])
        self.assertTrue(official_url.endswith("/v1/responses"))
        self.assertEqual(official["instructions"], "do X")
        self.assertEqual(official["input"], [{"role": "user", "content": "room"}])
