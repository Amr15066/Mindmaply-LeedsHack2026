"""GeminiClient / DeepSeekClient against stubbed SDK and HTTP layers (no network)."""

from types import SimpleNamespace

import pytest
from google.genai import types

from mindmap_chat.llm import base, deepseek, gemini
from mindmap_chat.llm.base import TRUNCATION_NOTICE, LLMResponseError


def text_response(text, finish=types.FinishReason.STOP):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text=text)]),
                finish_reason=finish,
            )
        ]
    )


class FakeModels:
    def __init__(self):
        self.next_response = text_response("hello")
        self.next_embedding = types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.1, 0.2, 0.3])]
        )
        self.generate_calls = []
        self.embed_calls = []

    def generate_content(self, *, model, contents, config=None):
        self.generate_calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        return self.next_response

    def embed_content(self, *, model, contents, config=None):
        self.embed_calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        return self.next_embedding


@pytest.fixture
def fake_models(monkeypatch):
    models = FakeModels()

    class FakeClient:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key
            self.models = models

    monkeypatch.setattr(gemini.genai, "Client", FakeClient)
    return models


class TestGeminiClient:
    def test_returns_text_and_sends_generation_config(self, fake_models):
        client = gemini.GeminiClient()
        assert client.call("hi") == "hello"
        call = fake_models.generate_calls[0]
        assert call.contents == "hi"
        assert call.model == gemini.config.gemini.model_name
        assert call.config.max_output_tokens == gemini.config.gemini.max_output_tokens
        assert call.config.response_mime_type is None

    def test_json_mode_requests_json_mime_type(self, fake_models):
        fake_models.next_response = text_response('{"a": 1}')
        assert gemini.GeminiClient().call_json("p") == {"a": 1}
        assert fake_models.generate_calls[0].config.response_mime_type == "application/json"

    def test_blocked_prompt_raises_clear_error(self, fake_models):
        """Regression (§2.10): zero candidates used to crash with an opaque error."""
        fake_models.next_response = types.GenerateContentResponse(
            candidates=[],
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.SAFETY
            ),
        )
        with pytest.raises(LLMResponseError, match="SAFETY"):
            gemini.GeminiClient().call("p")

    def test_safety_stop_without_text_raises(self, fake_models):
        fake_models.next_response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)]
        )
        with pytest.raises(LLMResponseError, match="SAFETY"):
            gemini.GeminiClient().call("p")

    def test_truncated_answer_is_marked(self, fake_models, caplog):
        """Regression (§4.6): MAX_TOKENS answers were presented as complete."""
        fake_models.next_response = text_response("half an ans", types.FinishReason.MAX_TOKENS)
        assert gemini.GeminiClient().call("p") == "half an ans" + TRUNCATION_NOTICE
        assert "truncated" in caplog.text
        assert caplog.records[-1].levelname == "WARNING"

    def test_truncated_json_is_not_marked(self, fake_models):
        fake_models.next_response = text_response('{"a": "cut', types.FinishReason.MAX_TOKENS)
        assert gemini.GeminiClient().call("p", json_mode=True) == '{"a": "cut'

    def test_embed(self, fake_models):
        assert gemini.GeminiClient().embed("text") == [0.1, 0.2, 0.3]
        assert fake_models.embed_calls[0].model == gemini.config.embeddings.model
        assert fake_models.embed_calls[0].contents == ["text"]
        assert (
            fake_models.embed_calls[0].config.output_dimensionality
            == gemini.config.embeddings.embedding_dim
        )

    def test_embed_batch_is_one_request(self, fake_models):
        fake_models.next_embedding = types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[1.0]), types.ContentEmbedding(values=[2.0])]
        )
        assert gemini.GeminiClient().embed_batch(["a", "b"]) == [[1.0], [2.0]]
        assert len(fake_models.embed_calls) == 1
        assert fake_models.embed_calls[0].contents == ["a", "b"]

    def test_embed_batch_count_mismatch_raises(self, fake_models):
        with pytest.raises(LLMResponseError):
            gemini.GeminiClient().embed_batch(["a", "b"])  # stub returns one embedding

    def test_embed_batch_of_nothing_makes_no_request(self, fake_models):
        assert gemini.GeminiClient().embed_batch([]) == []
        assert fake_models.embed_calls == []

    def test_empty_embedding_raises(self, fake_models):
        fake_models.next_embedding = types.EmbedContentResponse(embeddings=[])
        with pytest.raises(LLMResponseError):
            gemini.GeminiClient().embed("text")


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def deepseek_post(monkeypatch, fake_models):
    sent = []
    reply = {"choices": [{"message": {"content": "ds answer"}, "finish_reason": "stop"}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(json)
        return FakeHTTPResponse(reply)

    monkeypatch.setattr(deepseek.requests, "post", fake_post)
    return SimpleNamespace(sent=sent, reply=reply)


class TestDeepSeekClient:
    def test_constructs_and_answers(self, deepseek_post):
        """Regression: DeepSeekClient referenced a config section that did not exist."""
        client = deepseek.DeepSeekClient()
        assert client.call("hi") == "ds answer"
        assert deepseek_post.sent[0]["model"] == deepseek.config.deepseek.model_name
        assert "response_format" not in deepseek_post.sent[0]

    def test_json_mode(self, deepseek_post):
        deepseek_post.reply["choices"][0]["message"]["content"] = '{"a": 1}'
        assert deepseek.DeepSeekClient().call_json("p") == {"a": 1}
        assert deepseek_post.sent[0]["response_format"] == {"type": "json_object"}

    def test_truncation_is_marked(self, deepseek_post):
        deepseek_post.reply["choices"][0]["finish_reason"] = "length"
        assert deepseek.DeepSeekClient().call("p") == "ds answer" + TRUNCATION_NOTICE

    def test_empty_choices_raise(self, deepseek_post):
        deepseek_post.reply["choices"] = []
        with pytest.raises(LLMResponseError):
            deepseek.DeepSeekClient().call("p")

    def test_embeddings_use_the_gemini_embedding_model(self, deepseek_post, fake_models):
        assert deepseek.DeepSeekClient().embed("x") == [0.1, 0.2, 0.3]
        assert fake_models.embed_calls[0].model == gemini.config.embeddings.model


def test_llm_response_error_is_a_runtime_error():
    assert issubclass(base.LLMResponseError, RuntimeError)
