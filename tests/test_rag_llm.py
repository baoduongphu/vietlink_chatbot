"""Regression tests for the shared LLM adapter used by RAG."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock

from rag.llm import _llm_call_oneshot


class RagLLMAdapterTest(TestCase):
    def test_gpt_keeps_json_object_response_format(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"status":"done"}'))]
        )
        output = _llm_call_oneshot("gpt", client, "model", "prompt", 0, 100)
        self.assertEqual(output, '{"status":"done"}')
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_gemini_keeps_chat_send_message_flow(self) -> None:
        client = MagicMock()
        chat = client.chats.create.return_value
        chat.send_message.return_value = SimpleNamespace(text='{"status":"done"}')
        output = _llm_call_oneshot("gemini", client, "model", "prompt", 0, 100)
        self.assertEqual(output, '{"status":"done"}')
        chat.send_message.assert_called_once_with("prompt")
