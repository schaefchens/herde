# herde

A worker that does AI work — speech and text, later transcription — for any
project that speaks its protocol. It runs on your own computers and on those
of people who lend theirs: a Mac, or a Linux or Windows PC with Docker. The
worker always connects out to its projects, so it needs no public address and
no open port; it works behind any firewall.

There is no central server. Each project hosts its side of the protocol
([protocol/PROTOCOL.md](protocol/PROTOCOL.md)) and holds its own queue; a
worker serves every project in its config and picks the next task itself:

- **live** work first (a moment on air), the earliest due first; then
  **interactive** (a person waits); then **background** work, in turns by the
  weights you give the projects;
- only what it can finish before it is due, by its own measured speed;
- one task at a time on the GPU across speech and text (on a Mac both get two
  to five times slower when they share it), unless you allow more;
- it polls while it works: that keeps its tasks' leases alive, and a project
  always knows it is there.

ARCHE (a Christian community radio) is the first project: its PHP serves the
protocol at `/api/worker/v2`.

## What runs where

| Computer | Speech (Qwen3-TTS 1.7B) | Text | How |
|---|---|---|---|
| Mac with Apple Silicon | mlx-audio on the GPU | Ollama (the app) | natively: Docker cannot reach a Mac's GPU |
| Linux or Windows, NVIDIA (8 GB+) | the official `qwen-tts` on CUDA | Ollama container | Docker, `herde:cuda` |
| Linux or Windows, CPU only | — | Ollama container (slow) | Docker, `herde:base` |

On Windows, Docker Desktop with WSL2 passes an NVIDIA GPU through; AMD and
Intel GPUs are not supported in Docker there yet.

## Install on a Mac

```
git clone https://github.com/schaefchens/herde.git && cd herde
macos/install.sh
```

It makes `.venv` (Python 3.13 and the tested mlx-audio set), a config at
`~/.config/herde/config.toml` (readable only by you) and a launchd job file,
and prints how to start it. It can take over an `arche-worker` config: its
stations become projects (at their `/api/worker/v2`), their keys key files,
and its voices folder comes along. Then:

```
.venv/bin/herde check        # the config, ffmpeg, Ollama, and each project's answer
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/de.schaefchens.herde.plist
```

The first start downloads the speech model (about 4.2 GB; 4 GB more for
voices of your own). The log is `~/Library/Logs/herde.log`.

## Install with Docker (Linux, Windows)

The images are on GitHub's registry: `ghcr.io/schaefchens/herde:cuda` (speech
and text, an NVIDIA GPU with 8 GB or more) and `:base` (text through Ollama, no
speech). On Linux you need the NVIDIA Container Toolkit; on Windows, Docker
Desktop with WSL2 and a current NVIDIA driver. Nothing to clone or build.

**To lend your GPU to a project for speech** (what ARCHE asks for), two
commands, each on one line (they work in a Linux shell and in PowerShell):

```
docker run --rm -v herde:/config ghcr.io/schaefchens/herde:cuda join <url> <code>
docker run -d --name herde --restart unless-stopped --gpus all --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges -v herde:/config -v herde-cache:/cache -p 127.0.0.1:8737:8737 ghcr.io/schaefchens/herde:cuda
```

The first joins with the invite the project's admin gave you: it writes the
config and the key into the volume `herde`. The second runs the worker, now
and after every restart; the first start downloads the speech model (about
4 GB) into `herde-cache`. Then:

- the status page, with Pause and Resume: <http://127.0.0.1:8737>;
- pause from a shell: `docker exec herde herde pause 2h`, resume: `docker exec herde herde resume`;
- what it does: `docker logs -f herde`;
- an update: `docker pull ghcr.io/schaefchens/herde:cuda`, `docker rm -f herde`, then the second command again;
- to stop for good: `docker rm -f herde` (and `docker volume rm herde herde-cache` to forget the key and the model).

**With Ollama for text too**, use the compose files in `docker/`:

```
mkdir herde && cd herde && mkdir config cache
curl -fsSLO https://raw.githubusercontent.com/schaefchens/herde/main/docker/compose.yaml
curl -fsSLO https://raw.githubusercontent.com/schaefchens/herde/main/docker/compose.nvidia.yaml
docker compose run --rm herde join https://radio.example.org/api/worker/v2 7KQ3-M2XP-…
docker compose -f compose.yaml -f compose.nvidia.yaml up -d      # with an NVIDIA GPU
docker compose up -d                                           # without: text only
```

`join` writes `config/config.toml` and the project's key into `config/keys/`.
On Linux the container runs as your user (put `HERDE_UID` and `HERDE_GID` into
`.env` if yours is not 1000). Text models come from the Ollama container:
`docker compose exec ollama ollama pull qwen3.6:35b`, then list it in the
config (`[[text.models]]`, see `docker/config.example.toml`).

Either way the container runs as a plain user on a read-only file system, with
no capabilities; the key stays in a file, never in an environment variable.
To build an image yourself: `docker build -f docker/Dockerfile --target cuda -t
ghcr.io/schaefchens/herde:cuda .` (compose then uses yours).

## Joining, pausing, hours

- **Join** with the invite a project's admin made (shown once, valid for a
  few days): `herde join <url> <code>`. Joining again with a new code renews
  the key.
- **Pause** when you need the computer: `herde pause 2h`, `herde pause until
  18:00`, `herde pause` (until `herde resume`). Running work finishes first.
  The status page has the same buttons: <http://127.0.0.1:8737>.
- **Hours**: `windows = ["Mon-Fri 18:00-08:00", "Sat-Sun"]` in the config, for
  all projects or one. Outside them the worker takes nothing new and tells its
  projects when it is back.
- `herde status` shows what it does now.

Every setting is in [config.example.toml](config.example.toml).

## Speech

Qwen's nine presets speak ten languages and take a spoken direction
(`instruct`). A voice of your own is cloned from a recording by Qwen's Base
model: put `<Name>.<lang>.wav` (3–30 s, PCM) and `<Name>.<lang>.txt` (exactly
what it says) into the voices folder, and restart. It is offered in the
languages it has recordings for, and takes no direction — its words and its
recording carry the feeling. A recording of a real person only with their
consent.

`herde lab cases.json` speaks the same words in several ways (how a text is
cut into takes, the temperature) for you to compare by ear; ARCHE's
`bin/replay-show.php` writes such a file. `herde bench` measures this
computer's speed with the real models, so the scheduler's first estimates fit.

## Text

Through Ollama: a model is offered under the name projects ask for, mapped to
Ollama's tag on this computer (`gemma4-31b` → `gemma4:31b-mlx` on a Mac,
`gemma4:31b` elsewhere). herde never pulls a model — a project could otherwise
fill your disk. Answers follow the task's JSON schema, thinking is off, and a
model that writes only blank space is stopped early.

## Privacy and safety

- A project decides what each worker may see: work with people's words goes
  only to computers it trusts as its own; a lent computer gets public work.
- The worker keeps a task's input and output only while it works on it and
  never writes them to its log. Keys live in files readable only by you.
- It follows no URL a task names, downloads nothing a task names, gives text
  models no tools, and checks every id against the protocol's patterns.
- Keep a Mac's disk encrypted (FileVault). If a key gets out, the project's
  admin issues a new one.

## Try it: a little project of your own

`herde serve` is a small project on your computer: a page where you invite
workers, give them speech and text jobs, and hear or read what they made. It is
the project side of the protocol, so a worker treats it like any project.

```
herde serve
```

It prints the page's address with its key (`http://127.0.0.1:8740/#token=…`);
open it, invite a worker, run the `herde join …` command the page shows on that
computer, and submit a job. Without a GPU, a worker with `[tts] backend =
"stub"` and `[text] backend = "stub"` makes silent clips and fixed answers: the
whole path, end to end. The page listens on this computer only; its database
and clips live in `~/.local/share/herde-serve` (`--data`).

In Docker, beside a worker:

```
docker run -d --name herde-serve -p 127.0.0.1:8740:8740 -v herde-serve:/data ghcr.io/schaefchens/herde:base serve
docker logs herde-serve        (the page's address, with its key)
```

A worker on another computer needs an https address: put a reverse proxy
(Caddy, nginx) in front and start `herde serve --public-url https://…`, so
invites name that address.

Scripts use the same API as the page, with the key in `X-Herde-Token`:

| Call | |
|---|---|
| `POST /api/jobs` | `{"kind": "tts", "text", "voice", "lang", "instruct"?, "temperature"?}` or `{"kind": "text", "model", "prompt" or "messages", "system"?, "json_schema"?, "max_tokens"?}`, both with an optional `class` (`live`, `interactive`, `background`) and `due_in` (seconds) → `{"id"}` |
| `GET /api/jobs/{id}` | its state, its worker, and for text the answer |
| `GET /api/jobs/{id}/audio` | the MP3 of a speech job that is done |
| `DELETE /api/jobs/{id}` | cancel (a worker holding it hears so with its next poll) |
| `GET /api/state` | workers, open invites, the latest jobs |
| `POST /api/invites` | `{"name"}` → a one-time code and the command to join with it |

## For a project

Implement [protocol/PROTOCOL.md](protocol/PROTOCOL.md): five calls, JSON over
HTTPS, the key in `X-Worker-Key`. [protocol/schema.json](protocol/schema.json)
describes every message and [protocol/fixtures](protocol/fixtures) holds an
example of every exchange — test your side against them. Two working
implementations: `herde serve` ([herde/serve/project.py](herde/serve/project.py),
Python and SQLite, about 500 lines), and ARCHE's PHP (`server/app/Workers/` in
its repo) on shared hosting.

## Tests

`./test.sh` runs ruff and the tests: no model, no GPU, no network. The
image against a fake project, end to end:

```
docker compose -f tests/integration/compose.yaml up --build --exit-code-from project
```

## License

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)): use herde, change
it, build on it and earn money with it — run it for others, sell a service or
a product with it — as long as you keep the copyright notice and say what you
changed. The license also grants the patents its contributors hold on their
code. It grants no right to the name "herde" or to schaefchens' names and
logos: a product of your own goes under a name of your own.
