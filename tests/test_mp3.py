import subprocess

import pytest

from arche_worker import mp3


def test_the_ffmpeg_command():
    assert mp3.command("/x/ffmpeg", 24000, "96k") == [
        "/x/ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", "96k", "-f", "mp3", "pipe:1",
    ]


def test_encode_pipes_the_samples_through_ffmpeg():
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"ID3mp3", stderr=b"")

    data, ms = mp3.encode(b"\x00\x00" * 48000, 24000, "96k", ffmpeg="/x/ffmpeg", run=run)
    assert (data, ms) == (b"ID3mp3", 2000)
    assert calls[0][1]["input"] == b"\x00\x00" * 48000
    assert calls[0][0][0] == "/x/ffmpeg"


def test_a_failing_ffmpeg_is_an_encode_error():
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"Unknown encoder")

    with pytest.raises(mp3.EncodeError, match="Unknown encoder"):
        mp3.encode(b"\x00\x00", 24000, ffmpeg="ffmpeg", run=run)
    with pytest.raises(mp3.EncodeError):
        mp3.encode(b"", 24000, ffmpeg="ffmpeg", run=run)


def test_ffmpeg_from_homebrew_when_the_path_lacks_it(monkeypatch, tmp_path):
    brew = tmp_path / "ffmpeg"
    brew.write_text("")
    monkeypatch.setattr(mp3.shutil, "which", lambda name: None)
    monkeypatch.setattr(mp3, "HOMEBREW_FFMPEG", brew)
    assert mp3.find_ffmpeg() == str(brew)
    monkeypatch.setattr(mp3, "HOMEBREW_FFMPEG", tmp_path / "missing")
    with pytest.raises(mp3.EncodeError, match="brew install ffmpeg"):
        mp3.find_ffmpeg()
