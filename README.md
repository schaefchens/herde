# ARCHE voice worker

A small program for an Apple Silicon Mac that speaks for the station's AI
hosts with Qwen3-TTS, an open speech model, on the Mac's own GPU. The Mac
needs internet access but no public address: the worker asks the station for
voice tasks every few seconds, speaks them, and uploads each clip as an MP3.
One worker can serve several stations (production and dev), and several Macs
can work for one station at the same time. When no worker is online, the
hosts that use it fall back to the next host in their lineup.

## What you need

- A Mac with Apple Silicon (M1 or newer). An M1 Max speaks about twice as
  fast as real time: 10 seconds of speech take about 5 seconds.
- About 5 GB of disk for the model (downloaded once).
- Python 3.13 and ffmpeg: `brew install python@3.13 ffmpeg`.

## Set up

1. Install:

   ```
   worker/setup.sh
   ```

   This creates `worker/.venv` and, if there is none yet, a config at
   `~/.config/arche-worker/config.toml` (readable only by you).

2. In the station's /mod › KI-Moderation › Rechner, add a worker. /mod shows
   its key once: copy it into the config's `[[stations]]` block, together
   with the station's address.

3. Check the config, ffmpeg and each station, without loading the model:

   ```
   worker/run.sh check
   ```

4. Start it:

   ```
   worker/run.sh
   ```

   The first start downloads the model (about 4.2 GB). If the qwen3-tts-ui
   demo already downloaded it, point `hf_home` in the config's `[engine]` at
   its `.runtime/huggingface` folder instead.

Then, in /mod › KI-Moderation, give a host the voice "Eigener Rechner
(Qwen)" and pick a voice per language (for German, Sohee). `worker/run.sh
once` does a single task and stops, handy for a first try.

## Several stations

Add one `[[stations]]` block per station, the most important first; the
worker asks them in that order and speaks one task at a time:

```toml
[[stations]]
name = "production"
url = "https://radio.schaefchens.de"
key = "…"

[[stations]]
name = "dev"
url = "http://localhost:8080"
key = "…"
```

Plain `http` is accepted only for this Mac itself (`localhost`).

## Start at login

1. Copy `launchd/de.schaefchens.arche-worker.plist` to
   `~/Library/LaunchAgents/` and replace the `/Users/YOU/…` paths in it.
2. Load it:

   ```
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/de.schaefchens.arche-worker.plist
   ```

3. Restart it after a change with
   `launchctl kickstart -k gui/$(id -u)/de.schaefchens.arche-worker`, stop it with
   `launchctl bootout gui/$(id -u)/de.schaefchens.arche-worker`.

The log is in `~/Library/Logs/arche-worker.log`. With `keep_awake = true` the
worker keeps the Mac from going to sleep while it runs; a closed lid on
battery still sends it to sleep, and the hosts then fall back to other voices.

## Privacy

The texts the worker speaks include what listeners sent: their first names,
places and prayer requests. They stay on this Mac only as long as a take
lasts and are never written to the log. Keep the Mac's disk encrypted
(FileVault), and never share the config or a key. If a key gets out, issue a
new one in /mod › KI-Moderation › Rechner.

## Tests

`npm run test:worker` (or `worker/test.sh`) runs the worker's tests. They
need neither the model nor a GPU.
