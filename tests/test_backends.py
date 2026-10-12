"""The model backends' calls, against stand-ins for mlx-audio and qwen-tts: no model, no GPU.

The real libraries are not installed for the tests; what is checked is that
each backend calls them the way their documentation says — the names, the
arguments, the clone prompt made once per recording — and turns their float
samples into the PCM the rest of the worker speaks.
"""

import sys
import types
from typing import ClassVar

import pytest
from test_voices import recording

from herde.engines.tts.core import Sampling, TtsEngine
from herde.engines.tts.voices import discover

np = pytest.importorskip("numpy")
SAMPLING = Sampling(temperature=0.7, max_tokens=500)


class FakeQwenModel:
    loaded: ClassVar[list] = []

    def __init__(self, model_id, kwargs):
        self.model_id, self.kwargs = model_id, kwargs
        self.calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        m = cls(model_id, kwargs)
        cls.loaded.append(m)
        return m

    def generate_custom_voice(self, **kw):
        self.calls.append(("custom", kw))
        return [np.full(2400, 0.5, dtype=np.float32)], 24000

    def create_voice_clone_prompt(self, **kw):
        self.calls.append(("prompt", kw))
        return [f"prompt for {kw['ref_audio']}"]

    def generate_voice_clone(self, **kw):
        self.calls.append(("clone", kw))
        return [np.full(4800, -2.0, dtype=np.float32)], 24000


@pytest.fixture
def fake_torch(monkeypatch):
    FakeQwenModel.loaded = []
    seeds = []
    cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True, manual_seed_all=lambda s: seeds.append(("cuda", s)))
    torch = types.SimpleNamespace(
        cuda=cuda, version=types.SimpleNamespace(hip=None), bfloat16="bf16", float16="fp16", float32="fp32", manual_seed=lambda s: seeds.append(("cpu", s)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=FakeQwenModel))
    return seeds


def test_torch_speaks_presets_with_direction_and_clones_from_a_prompt_made_once(fake_torch, tmp_path):
    from herde.engines.tts.torch_backend import TorchBackend

    recording(tmp_path, "Faith.de")
    backend = TorchBackend("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "abc123", device="cuda:0")
    engine = TtsEngine(backend, own=discover(tmp_path, ("de", "en"), ("Sohee",)))
    engine.load()
    custom, base = FakeQwenModel.loaded
    assert custom.kwargs == {"device_map": "cuda:0", "dtype": "bf16"} and custom.model_id.endswith("CustomVoice")
    assert base.kwargs == {"device_map": "cuda:0", "dtype": "bf16", "revision": "abc123"}
    assert base.calls == [("prompt", {"ref_audio": str(tmp_path / "Faith.de.wav"), "ref_text": base.calls[0][1]["ref_text"]})]

    speech = engine.synthesize("Guten Abend.", "sohee", "de", "Warm.", 0.8, 5)
    kind, kw = custom.calls[-1]
    assert kind == "custom" and (kw["text"], kw["speaker"], kw["language"], kw["instruct"]) == ("Guten Abend.", "Sohee", "German", "Warm.")
    assert (kw["temperature"], kw["top_k"], kw["top_p"], kw["repetition_penalty"], kw["do_sample"]) == (0.8, 50, 1.0, 1.05, True)
    assert kw["max_new_tokens"] > 64
    assert fake_torch[-2:] == [("cpu", 5), ("cuda", 5)], "the seed for both"
    assert speech.sample_rate == 24000 and speech.pcm16[:2] == (16383).to_bytes(2, "little", signed=True)

    clone = engine.synthesize("Hallo zusammen.", "Faith", "de", "ignored", 0.7, 0)
    kind, kw = base.calls[-1]
    assert kind == "clone" and kw["voice_clone_prompt"] == [f"prompt for {tmp_path / 'Faith.de.wav'}"] and "instruct" not in kw
    assert clone.pcm16[:2] == (-32767).to_bytes(2, "little", signed=True), "samples beyond ±1 are clipped"


def test_torch_on_an_older_card_uses_fp16_and_without_a_gpu_says_how_to_start(fake_torch, monkeypatch):
    from herde.engines.tts.torch_backend import TorchBackend

    sys.modules["torch"].cuda.is_bf16_supported = lambda: False
    TorchBackend("m").load([])
    assert FakeQwenModel.loaded[-1].kwargs["dtype"] == "fp16"
    sys.modules["torch"].cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="NVIDIA"):
        TorchBackend("m").load([])


def test_torch_through_rocm_speaks_on_an_amd_gpu_as_on_cuda_and_without_one_says_how_to_start(fake_torch):
    from herde.engines.tts.torch_backend import TorchBackend

    sys.modules["torch"].version.hip = "7.2.26015"
    TorchBackend("m").load([])
    assert FakeQwenModel.loaded[-1].kwargs == {"device_map": "cuda:0", "dtype": "bf16"}, "ROCm's torch names an AMD GPU cuda"
    sys.modules["torch"].cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match=r"AMD.*/dev/kfd.*compose\.amd\.yaml"):
        TorchBackend("m").load([])


class FakeMlxModel:
    def __init__(self, kind):
        self.config = types.SimpleNamespace(tts_model_type=kind)
        self.sample_rate = 24000
        self.calls = []

    def generate_custom_voice(self, **kw):
        self.calls.append(("custom", kw))
        return iter([types.SimpleNamespace(sample_rate=24000, audio=np.zeros(240, dtype=np.float32))] * 2)

    def generate(self, **kw):
        self.calls.append(("clone", kw))
        return iter([types.SimpleNamespace(sample_rate=24000, audio=np.zeros(480, dtype=np.float32))])


@pytest.fixture
def fake_mlx(monkeypatch):
    models = {}

    def load_model(model_id, revision=None):
        models[model_id] = FakeMlxModel("base" if "Base" in model_id else "custom_voice")
        models[model_id].revision = revision
        return models[model_id]

    seeds = []
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", types.ModuleType("mlx_audio.tts"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", types.SimpleNamespace(load_audio=lambda path, sample_rate: f"audio of {path}"))
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", types.SimpleNamespace(random=types.SimpleNamespace(seed=seeds.append)))
    return models, seeds


def test_mlx_speaks_as_the_arche_worker_did(fake_mlx, tmp_path, monkeypatch):
    from herde.engines.tts.mlx_backend import MlxBackend

    # Set, so the test's end restores them: the backend only sets what is not set yet.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "already-there"))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    models, seeds = fake_mlx
    recording(tmp_path, "Faith.en")
    backend = MlxBackend("mlx-community/CustomVoice", "rev1", "mlx-community/Base", "", hf_home=tmp_path / "hf")
    engine = TtsEngine(backend, own=discover(tmp_path, ("de", "en"), ("Sohee",)))
    engine.load()
    assert models["mlx-community/CustomVoice"].revision == "rev1" and models["mlx-community/Base"].revision is None
    engine.synthesize("Guten Abend.", "Sohee", "de", "Warm.", 0.7, 9)
    kind, kw = models["mlx-community/CustomVoice"].calls[-1]
    assert kind == "custom" and (kw["speaker"], kw["language"], kw["instruct"], kw["stream"], kw["verbose"]) == ("Sohee", "German", "Warm.", False, False)
    assert seeds == [9]
    speech = engine.synthesize("Hello there.", "Faith", "en", "", 0.7, 0)
    kind, kw = models["mlx-community/Base"].calls[-1]
    assert kind == "clone" and kw["ref_audio"] == f"audio of {tmp_path / 'Faith.en.wav'}" and kw["lang_code"] == "english"
    assert speech.ms == 20, "480 samples at 24 kHz"


def test_mlx_refuses_a_checkpoint_of_the_wrong_kind(fake_mlx):
    from herde.engines.tts.mlx_backend import MlxBackend

    with pytest.raises(ValueError, match="CustomVoice"):
        MlxBackend("mlx-community/Base", "").load([])
