"""Text through Ollama against a stand-in (httpx.MockTransport): no model, no GPU, never a pull."""

import asyncio
import json

import httpx
import pytest

from herde.config import TextModel
from herde.engines.text import Cancelled, EngineError, InvalidTask, TooLarge
from herde.engines.text.ollama import OllamaEngine
from herde.engines.text.stub import StubText, fill
from herde.protocol import Message, TextInput

GEMMA = TextModel(name="gemma4-31b", tag="gemma4:31b-mlx", context=32768)
QWEN = TextModel(name="qwen3.6-35b", tag="qwen3.6:35b-mlx", context=16384)
SCHEMA = {"type": "object", "required": ["de", "en"], "properties": {"de": {"type": "string"}, "en": {"type": "string"}}}


class FakeOllama:
    def __init__(self, pieces=None, final=None, status=200, tags=("gemma4:31b-mlx",), ps=()):
        self.pieces = pieces if pieces is not None else ['{"de": "Hallo",', ' "en": "Hello"}']
        self.final = final or {"done": True, "done_reason": "stop", "prompt_eval_count": 3900, "prompt_eval_duration": 58_000_000_000,
                               "eval_count": 30, "eval_duration": 2_700_000_000, "load_duration": 15_000_000_000}
        self.status = status
        self.tags, self.ps = tags, ps
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": t, "model": t} for t in self.tags]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": t, "model": t} for t in self.ps]})
        if path == "/api/chat":
            if self.status != 200:
                return httpx.Response(self.status, json={"error": "model not found"})
            lines = [json.dumps({"message": {"role": "assistant", "content": p}, "done": False}) for p in self.pieces] + [json.dumps(self.final)]
            return httpx.Response(200, content="\n".join(lines).encode())
        return httpx.Response(404, json={"error": "no such path"})

    def paths(self):
        return [r.url.path for r in self.requests]


def engine(fake, models=(GEMMA, QWEN), max_tokens=4096):
    e = OllamaEngine("http://127.0.0.1:11434", models, "60m", max_tokens, transport=httpx.MockTransport(fake))
    asyncio.run(e.refresh())
    return e


def task(model="gemma4-31b", max_tokens=1024, schema=SCHEMA):
    return TextInput(model=model, max_tokens=max_tokens, temperature=1.0, top_p=0.95, top_k=64, system="You are a radio host.",
                     messages=(Message("user", "The moment, as JSON data: {}"),), json_schema=schema)


def test_only_listed_models_ollama_already_has_are_offered():
    fake = FakeOllama(tags=("gemma4:31b-mlx", "llama3:latest"), ps=("gemma4:31b-mlx",))
    e = engine(fake)
    assert e.available == {"gemma4-31b"} and e.loaded == {"gemma4-31b"}, "qwen is listed but not pulled: not offered"
    caps = e.caps(lambda name: {"read_tps": 67.0})
    assert caps == [{"kind": "text", "model": "gemma4-31b", "location": "local", "json_schema": True, "context": 32768, "max_tokens": 4096,
                     "loaded": True, "speed": {"read_tps": 67.0}}]
    assert e.can("gemma4-31b") and not e.can("qwen3.6-35b")
    assert "/api/pull" not in fake.paths()


def test_a_bare_tag_is_ollamas_latest():
    e = engine(FakeOllama(tags=("llama3:latest",)), models=(TextModel(name="llama3", tag="llama3"),))
    assert e.available == {"llama3"}


def test_a_task_is_answered_in_its_schema_and_measured():
    fake = FakeOllama()
    e = engine(fake)
    progress = []
    out = asyncio.run(e.generate(task(), cancelled=lambda: False, progress=progress.append))
    assert json.loads(out.content) == {"de": "Hallo", "en": "Hello"}
    assert (out.finish, out.input_tokens, out.output_tokens, out.load_ms, out.read_ms, out.write_ms) == ("stop", 3900, 30, 15000, 58000, 2700)
    body = json.loads(fake.requests[-1].content)
    assert body["model"] == "gemma4:31b-mlx" and body["format"] == SCHEMA and body["think"] is False and body["stream"] is True
    assert body["keep_alive"] == "60m"
    assert body["options"] == {"num_predict": 1024, "num_ctx": 32768, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert body["messages"][0] == {"role": "system", "content": "You are a radio host."}
    assert progress and progress == sorted(progress)
    assert "gemma4-31b" in e.loaded
    assert "/api/pull" not in fake.paths(), "never a pull"


def test_a_model_writing_only_blank_space_is_stopped():
    e = engine(FakeOllama(pieces=['{"de": "Hallo"', *(["          "] * 30)]))
    with pytest.raises(EngineError, match="blank"):
        asyncio.run(e.generate(task(), cancelled=lambda: False))


def test_a_cut_answer_says_so():
    e = engine(FakeOllama(final={"done": True, "done_reason": "length", "prompt_eval_count": 10, "eval_count": 1024}))
    assert asyncio.run(e.generate(task(), cancelled=lambda: False)).finish == "length"


def test_what_it_cannot_do_is_invalid_or_too_large_and_a_cancel_stops_it():
    e = engine(FakeOllama())
    with pytest.raises(InvalidTask):
        asyncio.run(e.generate(task(model="qwen3.6-35b"), cancelled=lambda: False))
    with pytest.raises(TooLarge):
        asyncio.run(engine(FakeOllama(), max_tokens=512).generate(task(max_tokens=1024), cancelled=lambda: False))
    with pytest.raises(Cancelled):
        asyncio.run(e.generate(task(), cancelled=lambda: True))
    with pytest.raises(InvalidTask):
        asyncio.run(engine(FakeOllama(status=404)).generate(task(), cancelled=lambda: False))
    with pytest.raises(EngineError):
        asyncio.run(engine(FakeOllama(status=500)).generate(task(), cancelled=lambda: False))


def test_ollama_down_means_no_text_until_it_answers():
    def down(request):
        raise httpx.ConnectError("refused")

    e = OllamaEngine("http://127.0.0.1:11434", (GEMMA,), transport=httpx.MockTransport(down))
    asyncio.run(e.refresh())
    assert not e.ready and e.available == set()


def test_the_stub_fills_a_schema_with_the_least_that_fits():
    assert fill(SCHEMA) == {"de": "Stub text.", "en": "Stub text."}
    nested = {"type": "object", "required": ["a", "b", "c", "d", "e"], "properties": {
        "a": {"enum": ["yes", "no"]}, "b": {"type": "integer", "minimum": 3}, "c": {"type": "array", "minItems": 1, "items": {"type": "boolean"}},
        "d": {"type": ["null", "string"]}, "e": {"type": "object", "required": ["x"], "properties": {"x": {"const": 7}}}}}
    assert fill(nested) == {"a": "yes", "b": 3, "c": [False], "d": "Stub text.", "e": {"x": 7}}
    out = asyncio.run(StubText((GEMMA,)).generate(task(), cancelled=lambda: False))
    assert json.loads(out.content) == {"de": "Stub text.", "en": "Stub text."} and out.finish == "stop"
