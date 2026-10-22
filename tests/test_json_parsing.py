import json

import pytest

from mindmap_chat.llm.base import _extract_json_payload, _repair_json_payload


class TestExtractJsonPayload:
    def test_plain_object_is_returned_as_is(self):
        assert _extract_json_payload(' {"a": 1} ') == '{"a": 1}'

    def test_strips_surrounding_prose_and_fences(self):
        text = 'Sure! Here you go:\n```json\n{"a": {"b": 2}}\n```\nAnything else?'
        assert json.loads(_extract_json_payload(text)) == {"a": {"b": 2}}

    def test_braces_inside_strings_do_not_confuse_depth(self):
        text = 'prefix {"msg": "a } inside", "n": 1} suffix'
        assert json.loads(_extract_json_payload(text)) == {"msg": "a } inside", "n": 1}

    def test_escaped_quotes_inside_strings(self):
        text = 'x {"msg": "she said \\"hi}\\"", "n": 1} y'
        assert json.loads(_extract_json_payload(text)) == {"msg": 'she said "hi}"', "n": 1}

    @pytest.mark.parametrize("text", ["", "no json here", '{"unclosed": 1'])
    def test_returns_none_when_no_complete_object(self, text):
        assert _extract_json_payload(text) is None


class TestRepairJsonPayload:
    def test_escapes_raw_newlines_inside_strings(self):
        broken = '{"summary": "line one\nline two"}'
        assert json.loads(_repair_json_payload(broken)) == {"summary": "line one\nline two"}

    def test_closes_unbalanced_braces(self):
        assert json.loads(_repair_json_payload('{"a": {"b": 1}')) == {"a": {"b": 1}}

    def test_closes_unterminated_string(self):
        assert json.loads(_repair_json_payload('{"a": "trunc')) == {"a": "trunc"}

    @pytest.mark.parametrize("text", ["", "no braces at all"])
    def test_returns_none_without_an_object(self, text):
        assert _repair_json_payload(text) is None


class TestCallJson:
    def test_parses_clean_json(self, fake_llm):
        fake_llm.raw_responses = ['{"x": 1}']
        assert fake_llm.call_json("p") == {"x": 1}

    def test_parses_fenced_json(self, fake_llm):
        fake_llm.raw_responses = ['```json\n{"x": [1, 2]}\n```']
        assert fake_llm.call_json("p") == {"x": [1, 2]}

    def test_repairs_truncated_json(self, fake_llm):
        fake_llm.raw_responses = ['{"x": "cut off']
        assert fake_llm.call_json("p") == {"x": "cut off"}

    def test_raises_on_garbage(self, fake_llm):
        fake_llm.raw_responses = ["definitely not json"]
        with pytest.raises(json.JSONDecodeError):
            fake_llm.call_json("p")
