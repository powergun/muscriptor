# The HTTP server, soundfonts, and auralization

The `muscriptor serve` command turns the same `TranscriptionModel` used by the CLI into a long-running HTTP service. The service has one job that matters — stream a transcription of an uploaded audio file to a browser as it is generated — plus a handful of supporting endpoints: instrument metadata, a compressed soundfont for the browser's in-page synthesizer, a server-side rendering endpoint that mixes the transcription against the original audio, and (optionally) the static frontend bundle itself. This chapter covers the Python that implements all of that: `muscriptor/server.py` (the FastAPI app), `muscriptor/soundfonts.py` (where the soundfonts come from), and `muscriptor/utils/auralization.py` (the FluidSynth rendering). The TypeScript frontend under `web/` is out of scope except where the server's wire format constrains it.

The design center is `POST /transcribe`: a multipart upload in, a `text/event-stream` (Server-Sent Events) out, with the model's own generator streamed straight to the socket and a single global lock enforcing one transcription at a time with newest-request-wins preemption. Everything else is comparatively conventional FastAPI.

## The `serve` command (wiring from `main.py`)

`serve` is a Typer subcommand (`muscriptor/main.py:289-321`). Its options and defaults:

- `--host` (default `127.0.0.1`, `main.py:291`) — loopback only by default, so the server is not exposed off-box unless the operator opts in.
- `--port` (default `8222`, `main.py:292`).
- `--model` / `-m` (default `None`, `main.py:293-303`) — a size keyword (`small`/`medium`/`large`), a local safetensors path, or an `hf://` / `http(s)://` URL. `None` resolves to the `medium` variant via `TranscriptionModel`'s `_resolve_source` / `_DEFAULT_SIZE`.
- `--device` / `-d` (default `"auto"`, `main.py:304-309`) — `"auto"` is translated to `None` (`main.py:316`) so `load_model` picks CUDA when available, else CPU.

The command loads the model once, up front (`main.py:317-318`), then locates the frontend bundle at `muscriptor/web_dist` relative to the installed package (`main.py:319`) and passes it to `create_app` **only if that directory actually exists** (`main.py:320`). Finally it hands the constructed app to `uvicorn.run(fastapi_app, host=host, port=port)` (`main.py:321`). Note what is *not* configured: no `--workers`, no reload, no explicit worker count. `uvicorn.run` with a concrete app object runs a single process with a single event loop. That single-process assumption is load-bearing for the concurrency model below — the `transcribe_lock` is an in-process `threading.Lock`, so it only serializes requests within one process.

Unlike the `transcribe` subcommand, `serve` does **not** force the model to float32 (`main.py:207` does that only for the CLI path). The served model keeps whatever dtype `load_model` produced, and on CUDA runs under the float16 autocast configured in `_build_model` (`transcription_model.py:204-206`).

## `create_app`: construction, state, and routing

`create_app(model, web_dir=None)` (`server.py:68`) builds and returns a `FastAPI` instance titled `"muscriptor"`. There is no application factory beyond this, no lifespan handler, and — verified by grep across `muscriptor/` — **no CORS middleware and no middleware of any kind**. A browser served from a different origin than the API would be blocked by the same-origin policy; in the shipped configuration the frontend is served from the same origin (see [static serving](#serving-the-frontend-static-bundle)), so no CORS is needed.

All mutable server state lives in closure variables captured by the route handlers, not on the app object:

- `model` — the single shared `TranscriptionModel`, captured directly. It is never copied or re-instantiated per request; every `/transcribe` call drives the same model object (`server.py:169`).
- `transcribe_lock = threading.Lock()` (`server.py:71`) — the global serialization lock. One transcription at a time, process-wide.
- `lock_timeout_s = 60.0` (`server.py:72`) — how long a new request will wait to acquire the lock before giving up with 503.
- `cancel_guard = threading.Lock()` and `current_cancel: threading.Event | None` (`server.py:79-80`) — the preemption channel. `current_cancel` is the cancel flag of the run currently holding (or last to hold) the lock; `cancel_guard` protects reads/writes of it.

Routes, in registration order (order matters — see the static mount caveat):

| Method | Path | Handler | Purpose |
|---|---|---|---|
| GET | `/health` | `health` (`server.py:82`) | liveness probe, returns `{"status": "ok"}` |
| GET | `/instruments` | `list_instruments` (`server.py:86`) | the instrument-group name list |
| GET | `/soundfonts/MuseScore_General.sf3` | `soundfont` (`server.py:90`) | compressed soundfont for the browser synth |
| POST | `/transcribe` | `transcribe` (`server.py:100`) | the SSE transcription stream |
| POST | `/auralize` | `auralize` (`server.py:214`) | server-side WAV rendering |
| — | `/` (mount) | `StaticFiles` (`server.py:274-277`) | the frontend bundle, if present |

The `StaticFiles` catch-all is mounted at `/` **last** (`server.py:277`), and only when `web_dir` is given and resolves to a real directory (`server.py:274-276`). Starlette matches routes in registration order and returns the first match, so the five explicit API routes above are checked before the `/`-mount ever gets a chance to swallow the request. If you add a new API route, it must be registered *before* line 274 or the static mount will shadow it.

## `GET /health` and `GET /instruments`

`health` (`server.py:82-84`) is a trivial `async` handler returning `{"status": "ok"}` — no model touch, no lock, always fast. Use it for readiness/liveness checks.

`list_instruments` (`server.py:86-88`) returns `{"instruments": [...]}` where the list is `list(MT3_FULL_PLUS_GROUP_NAMES.keys())`. `MT3_FULL_PLUS_GROUP_NAMES` (`muscriptor/tokenizer/mt3.py:118`) is a `dict[str, int]` mapping each instrument-group name (`"acoustic_piano"`, `"electric_piano"`, …) to its group id, so the response is the ordered list of names the frontend offers as constraint checkboxes and that `POST /transcribe` validates against. This is the same dictionary the CLI's `list-instruments` command iterates (`main.py:324-328`).

## `POST /transcribe`: the SSE stream

This is the core endpoint. Its signature (`server.py:100-104`) declares a `multipart/form-data` body:

- `file: UploadFile` (`File()`) — the audio upload. Field name is literally `file`. Any format libsndfile can read is accepted (WAV, mp3, flac, ogg, m4a, …); dispatch is by content, not filename.
- `instruments: list[str]` (`Form(default_factory=list)`) — zero or more instrument-group names as repeated form fields. Empty by default. When non-empty, it is a hard decoding constraint (every other instrument is masked out during generation).

### Request handling before streaming

The handler is `async`, and everything up to the point where streaming begins runs on the event loop:

1. **Read the upload** — `data = await file.read()` (`server.py:105`) pulls the whole file into memory. There is no size limit and no streaming-to-disk; a large upload is fully buffered.
2. **Decode** (`server.py:112-121`). PCM WAV is decoded by the stdlib reader `_read_wav_file` (byte-for-byte identical to the CLI's WAV path); on `wave.Error`/`EOFError` it falls back to `_read_non_wav_file` (soundfile/libsndfile). If *that* also fails, the upload is undecodable and the endpoint raises `HTTPException(400)` with the filename and the underlying error — a client fault, deliberately not surfaced as a 500. Both readers accept an `io.BytesIO`, so nothing is written to disk. The decoded result is `(wav, sr)` where `wav` is a `[C, T]` float32 tensor at the file's native sample rate — the server does **not** resample; it hands the model the tuple and lets `TranscriptionModel._load_wav` downmix to mono and resample to 16 kHz (`transcription_model.py:603-619`). Tests `test_transcribe_passes_tensor_not_path` and `test_transcribe_accepts_non_wav_audio` pin exactly this: the model receives an in-memory `(tensor, sr)` tuple with the sample rate preserved (16000 for the WAV fixture, 22050 for the FLAC fixture).
3. **Validate instruments** (`server.py:123-128`). Any name not in `MT3_FULL_PLUS_GROUP_NAMES` yields `HTTPException(400)` listing the unknown names. Unlike the CLI, the server accepts **exact names only** — no abbreviation resolution (`resolve_instrument_names` is a CLI-side convenience; the server assumes the frontend sends canonical names).
4. **Acquire the lock with preemption** (`server.py:134-153`) — described in [Concurrency](#concurrency-model-the-preemption-lock) below.

Only after the lock is held does the handler return a `StreamingResponse` (`server.py:207-212`) and streaming begin. That means the 4xx errors above happen *before* any bytes of the event stream are sent, so the client sees a clean `400` status. Errors *after* streaming starts cannot change the status code (it is already `200`) — see the gotchas.

### The generator and the wire format

The response body is produced by the synchronous generator `gen()` (`server.py:162-205`). Starlette iterates a sync generator in a threadpool, so `model.transcribe(...)` — which is CPU/GPU-bound and blocking — runs off the event loop; the loop stays free to accept the next request and drive its preemption signaling. `gen()` calls:

```python
model.transcribe((wav, sr), instruments=instruments or None,
                 batch_size=1, no_eos_is_ok=True)
```

`batch_size=1` (`server.py:172`) is deliberate and specific to the server: it makes each 5-second chunk's notes stream out as soon as that chunk is generated, rather than being buffered until a whole batch of chunks completes (the CLI defaults to 4 on GPU). `no_eos_is_ok=True` (`server.py:173`) downgrades a runaway chunk that never emits EOS to a warning instead of aborting the stream. `instruments or None` passes `None` when the list is empty, matching the model's "decode anything" mode.

Each item the model yields is serialized to exactly one SSE frame of the form `data: <compact-json>\n\n`:

| `type` | JSON fields | Python source | When |
|---|---|---|---|
| `progress` | `completed` (int), `total` (int) | `ProgressEvent` (`server.py:180-191`) | once up front (`completed=0`, learns `total`), then once per finished chunk |
| `start` | `pitch` (int), `start_time` (float s), `index` (int), `instrument` (str) | `NoteStartEvent` via `event_to_dict` (`server.py:58-60`) | a note onset |
| `end` | `end_time` (float s), `start_event_index` (int) | `NoteEndEvent` via `event_to_dict` (`server.py:61-65`) | a note offset; `start_event_index` matches an earlier `start`'s `index` |
| `midi` | `data` (str, base64 of the `.mid` bytes) | `events_to_midi_bytes` (`server.py:200-203`) | exactly once, last, unless preempted |

Example frames from a two-chunk transcription:

```
data: {"type": "progress", "completed": 0, "total": 2}

data: {"type": "start", "pitch": 60, "start_time": 0.0, "index": 0, "instrument": "acoustic_piano"}

data: {"type": "end", "end_time": 0.5, "start_event_index": 0}

data: {"type": "progress", "completed": 1, "total": 2}

data: {"type": "midi", "data": "TVRoZAAAAAYAAQ..."}
```

A subtle but important detail: the SSE `type` is carried **inside the JSON payload**, not on an SSE `event:` line. There is no `event:` field and no `id:` field — every frame is a default (`message`) SSE event whose `data` is JSON. A browser `EventSource` would dispatch them all as `message` and the client discriminates on the JSON `type`. The frontend therefore parses `data:` lines and switches on `type` rather than registering named event listeners.

`ProgressEvent`s are forwarded to the client but explicitly excluded from the `events` list (`server.py:180-192`): only `start`/`end` events are appended (`server.py:192`), and the terminal MIDI is built from that list via `model.events_to_midi_bytes(iter(events))` (`server.py:200`). This reuses the *exact* CLI serialization path, so the `.mid` bytes streamed by the server are identical to what `muscriptor transcribe -f midi` writes. `test_transcribe_forwards_progress` verifies both halves: progress frames appear in the stream, and the list handed to `events_to_midi_bytes` contains no `ProgressEvent`.

Ordering guarantees are inherited wholesale from `transcribe()` (`transcription_model.py:304-405`): within a chunk, events are temporally ordered; all of chunk N's events precede any of chunk N+1's; every `start` is followed later by exactly one matching `end`. The terminal `midi` frame always comes last because it is emitted after the generator loop finishes (`server.py:195-203`). An empty transcription (no notes) still emits the `midi` frame — `test_transcribe_empty_stream` pins this: the stream is a single `midi` event.

```mermaid
sequenceDiagram
    participant C as Client (browser)
    participant EP as async /transcribe<br/>(event loop)
    participant L as transcribe_lock
    participant G as gen() + model.transcribe<br/>(threadpool)

    C->>EP: POST multipart (file, instruments[])
    EP->>EP: await file.read()
    EP->>EP: decode WAV / soundfile
    Note over EP: 400 if undecodable or unknown instrument<br/>(before any stream bytes)
    EP->>L: set current_cancel (preempt), acquire via to_thread (≤60s)
    alt lock not acquired within 60s
        EP-->>C: 503 server busy
    else acquired
        L-->>EP: acquired
        EP->>EP: cancel = new Event(); current_cancel = cancel
        EP-->>C: 200 text/event-stream
        loop each event from model.transcribe
            G->>G: if cancel.is_set(): stop (newer request won)
            G-->>C: data: {progress|start|end}\n\n
        end
        G-->>C: data: {"type":"midi","data": base64}\n\n
        G->>L: release_lock() in finally
    end
```

## Concurrency model: the preemption lock

This is the part operators must understand. Transcriptions are **serialized process-wide** by `transcribe_lock`, and a newer request **preempts** an in-flight one rather than queueing behind it to completion. The mechanism (`server.py:134-160`):

**Acquire loop** (`server.py:134-150`). Before acquiring, and on every ~1-second retry, the handler sets `current_cancel` — the cancel flag of whoever currently holds the lock — so the running transcription stops at its next event boundary instead of transcribing to completion for a client that has moved on. It then tries `transcribe_lock.acquire(True, min(1.0, remaining))` via `asyncio.to_thread` (so the blocking acquire does not stall the event loop). If it cannot acquire within `lock_timeout_s = 60.0` total, it raises `HTTPException(503, "server busy: another transcription is in progress")`.

**Take ownership** (`server.py:151-153`). Once acquired, the handler creates a fresh `cancel = threading.Event()` and publishes it as `current_cancel` under `cancel_guard`. From here, any *newer* request's acquire loop will set *this* cancel event.

**Cancellation checks** (`server.py:178-179`, `198-199`). Inside `gen()`, `cancel.is_set()` is checked before yielding each event and again before building the terminal MIDI. When set, `gen()` `return`s — closing the `model.transcribe` generator — and the `finally` releases the lock, at most one chunk after the signal. A preempted stream therefore ends **without** a `midi` frame.

The invariant this produces: *a transcription will not run to completion while another request is waiting for the lock.* The re-signaling each second means even a run that started while a competitor was already waiting (a third request that beat it to the lock) gets cancelled — so the last request standing is the one that completes. It is "newest wins" in the sense of "whoever is still waiting cancels whoever is running," not a strict wall-clock arrival order (acquisition among racing threads is not FIFO).

**Lock-release safety.** The lock must be released exactly once, from whichever cleanup path runs first. `_make_release_once` (`server.py:37-55`) wraps `transcribe_lock.release()` in an idempotent, thread-safe callable (guarded by its own inner lock and a `released` flag) so a double release can never raise `RuntimeError`. Two paths call it: `gen()`'s `finally` (`server.py:204-205`), covering normal completion, mid-stream errors, and disconnects once iteration has started; and the `StreamingResponse(background=BackgroundTask(release_lock))` (`server.py:210`), covering the case where the client disconnects *before* the generator is ever iterated — in which case `gen()`'s `finally` would never run and the lock would leak forever. This dual-path release is the reason the idempotent wrapper exists.

Practical consequences for operators:

- **One transcription at a time per process.** Throughput is one stream; concurrency is achieved (if needed) by running multiple processes behind a load balancer, each with its own lock and its own model copy (and its own GPU memory).
- **Cancellation must not rely on TCP disconnects.** The comment at `server.py:75-78` is explicit: aborts do not always reach the server (port forwards / proxies that keep the upstream connection open after the browser aborts), so the explicit cancel-event handshake — not connection teardown — is what stops stale work.
- **A burst of requests degrades gracefully to the newest.** Ten rapid clicks don't queue ten transcriptions; each new request cancels the previous, and only the last completes. Others receive a truncated stream (no `midi`) or, if they wait more than 60 s for the lock, a 503.

## `POST /auralize`: server-side WAV rendering

`auralize` (`server.py:214-272`) renders a transcription to audio server-side (as opposed to the browser's own synth). Multipart inputs:

- `midi: UploadFile` (`File()`) — the `.mid` to render. Required.
- `audio: UploadFile | None` (`File()`, default `None`) — the original source audio, required only for `mode="mix"`.
- `mode: str` (`Form()`, default `"mix"`) — `"mix"` or `"synth"`.

Validation (`server.py:228-233`): an unknown `mode` → 400; `mode="mix"` without an `audio` upload → 400. The heavy imports (`auralize`, `synthesize` from `utils.auralization`) are done lazily inside the handler (`server.py:225-226`) so the FluidSynth/soundfile machinery is only pulled in when this endpoint is actually hit.

The handler stages everything through named temp files created with `delete=False` (`server.py:238-262`): the uploaded MIDI to a `.mid`, an output `.wav`, and (for `mix`) the uploaded audio to a temp file whose suffix mirrors the upload's extension (`server.py:252`, defaulting to `.wav`). It then calls either `synthesize(midi_path, output_path)` (mono, synthesis only) or `do_auralize(midi_path, original_audio_path, output_path)` (stereo mix), reads the resulting WAV bytes back into memory, and returns them as `Response(content=wav_bytes, media_type="audio/wav")` (`server.py:272`). **The server always returns WAV** — there is no mp3 path here (the CLI's `--auralize` can emit mp3 by extension, but the endpoint hardcodes a `.wav` temp and `audio/wav`). Any exception during rendering is caught and re-raised as `HTTPException(500, str(e))` (`server.py:265-266`), and a `finally` unlinks every temp file that still exists (`server.py:267-270`), so temp hygiene holds even on failure.

## Soundfonts (`soundfonts.py` and the `/soundfonts` route)

`muscriptor/soundfonts.py` (19 lines) is pure configuration: two module constants naming the soundfonts, with documented SHA-256 hashes.

- `SF2_URL = "hf://MuScriptor/assets/MuseScore_General.sf2"` (`soundfonts.py:14`) — the full ~215 MB soundfont, rendered **server-side by FluidSynth** for `/auralize` and the CLI's `--auralize`.
- `SF3_URL = "hf://MuScriptor/assets/MuseScore_General.sf3"` (`soundfonts.py:19`) — a ~38 MB Vorbis-compressed build of the same soundfont, served to the **browser's** in-page `spessasynth_lib` synthesizer.

Neither ships in the wheel. Both are fetched lazily on first use through `download_if_necessary` (`muscriptor/utils/download.py:39`) and cached. Because both are `hf://` URLs, they land in the **HuggingFace hub cache** (via `hf_hub_download`), i.e. `~/.cache/huggingface/hub/`, *not* `~/.cache/muscriptor/` (that directory is only used for `http(s)://` downloads — `download.py:68-87`). The upstream origin is MuseScore's distribution mirror, re-hosted at `hf://MuScriptor/assets` under the MIT license (`soundfonts.py:1-9`).

The only soundfont-serving route is `GET /soundfonts/MuseScore_General.sf3` (`server.py:90-98`). On first request it downloads `SF3_URL` in a worker thread — `await asyncio.to_thread(download_if_necessary, SF3_URL)` — so the (large, one-time) fetch does not block the event loop, then returns a `FileResponse` of the cached file with `media_type="application/octet-stream"`. Subsequent requests hit the cache and return immediately. There is **no** route that serves the `.sf2` — the full soundfont never leaves the server; only the compressed `.sf3` is sent to the browser.

## Auralization internals (`utils/auralization.py`)

This module shells out to the `fluidsynth` binary and blends the result with the original audio. Everything here runs at **44100 Hz** (`_SAMPLE_RATE = 44100`, `auralization.py:27`) — distinct from the model's 16 kHz. There are two public entry points, `synthesize` and `auralize`, plus helpers.

**Soundfont resolution** (`_resolve_soundfont`, `auralization.py:36-48`). With no explicit path: prefer a pre-downloaded `MuseScore_General.sf2` at the repo root (`_LOCAL_SOUNDFONT`, `auralization.py:26` — kept for checkouts / Docker images that bundle one), else download `SF2_URL`. An explicit path that doesn't exist raises `FileNotFoundError` with a remediation message.

**FluidSynth invocation** (`_synthesize_midi`, `auralization.py:51-84`). Renders MIDI → mono float32 at 44100 Hz. It writes to a temp `.wav`, then runs:

```python
subprocess.run(["fluidsynth", "-ni", "-F", synth_tmp, "-r", "44100",
                str(soundfont_path), str(midi_path)], capture_output=True)
```

`-ni` = non-interactive / no MIDI input; `-F` = fast-render to file and exit; `-r` = sample rate. The ordering is load-bearing and commented (`auralization.py:60-62`): **options must precede the positional soundfont/MIDI arguments**, because FluidSynth ≥ 2.5 silently ignores trailing options (exits 0 while writing no output). A non-zero exit raises `RuntimeError` with the captured stderr (`auralization.py:72-77`). The rendered WAV is read back with soundfile and downmixed to mono if FluidSynth produced stereo (`auralization.py:78-81`). The temp file is always removed in a `finally` (`auralization.py:82-84`).

**`synthesize`** (`auralization.py:87-107`) is the `mode="synth"` path: resolve soundfont → render → `sf.write(output_path, synth_audio, 44100)`. Mono output, no original audio involved.

**`auralize`** (`auralization.py:110-156`) builds the stereo check-mix. Steps:

1. Render the MIDI to mono via `_synthesize_midi` (`auralization.py:138`).
2. Load the original audio as mono at 44100 Hz via `_load_mono_44k` → `load_audio(path, target_sr=44100)` (`auralization.py:141`, `30-33`).
3. **Length-match by zero-padding both to the longer length** (`auralization.py:143-146`) — the two signals are aligned from sample 0; whichever is shorter is padded at the end. There is no time-alignment beyond both starting at t=0.
4. **RMS-normalize the synthesis to the original's loudness** (`auralization.py:148-152`): scale `synth *= rms_orig / rms_synth`, guarded by `rms_synth > 1e-8` to avoid dividing by silence. The original channel is left untouched.
5. **Assemble `[T, 2]`** with `np.stack([original_audio, synth_audio], axis=1)` — **L = original, R = synthesis** — and `sf.write` it as a WAV at 44100 Hz (`auralization.py:154-156`).

The output is a stereo WAV where the left ear is the source recording and the right ear is the model's transcription rendered through the soundfont — an A/B "did it hear the notes right" check.

## Serving the frontend (static bundle)

If `web_dir` is passed and resolves to a directory, `create_app` mounts it at `/` with `StaticFiles(directory=web_path, html=True)` (`server.py:274-277`). `html=True` makes it an SPA-style mount: directory requests serve `index.html`, and the frontend's client-side routing takes over. As noted above, this mount is registered last so the API routes win; it is a no-op when the `web_dist` bundle is absent (a source checkout that hasn't built the frontend), in which case the server is API-only. The condition is checked twice — once in `serve` (`main.py:320`) and again in `create_app` (`server.py:276`) — so passing a non-directory `web_dir` degrades to API-only rather than erroring.

## Gotchas & invariants

- **Single process, single lock.** `transcribe_lock` is an in-process `threading.Lock` and `serve` runs uvicorn with one worker. Serialization and preemption hold *within a process only*. Scaling out means multiple processes, each with an independent lock and its own model/GPU copy.
- **Preemption ≠ TCP disconnect.** Stale transcriptions are cancelled by the explicit `current_cancel` event handshake, not by connection teardown, precisely because aborts don't always propagate through proxies/port-forwards (`server.py:75-78`).
- **A preempted or errored stream has no `midi` frame.** The terminal `midi` event is the client's only reliable "done" signal. If `cancel.is_set()` fires (`server.py:198`) or `model.transcribe` raises mid-iteration, the stream simply ends without it. Because the HTTP status is already `200` once streaming starts, a mid-stream error **cannot** become a 4xx/5xx — only pre-stream failures (decode, unknown instrument) return clean error codes. Clients must treat "stream closed without a `midi` frame" as failure.
- **503 vs 400 vs silent truncation** are three distinct failure modes: 503 = couldn't get the lock in 60 s; 400 = undecodable audio or unknown instrument (pre-stream); truncated stream = preempted or errored mid-generation.
- **The SSE `type` is in the JSON, not the SSE `event:` line.** All frames are default `message` events; discriminate on the JSON `type` field.
- **No resampling on the server.** `/transcribe` hands the model the raw decoded `(tensor, sr)`; the model owns downmix + 16 kHz resample. Tests pin that the original sample rate reaches the model unchanged.
- **Whole-file buffering.** Both `/transcribe` and `/auralize` `await …read()` the full upload into memory with no size cap — a deployment concern for large files or hostile clients.
- **`event_to_dict` is duplicated.** `server.py:58-65` and `main.py:39-46` are byte-identical; the server's docstring even cross-references `main._event_to_dict`. Changing the wire shape means editing both.
- **Soundfont cache location.** `hf://` soundfonts cache to the HuggingFace hub cache, not `~/.cache/muscriptor/`. `auralization.py`'s docstrings say "cached under `~/.cache/muscriptor/`," which is imprecise for the `.sf2` (an `hf://` URL); `soundfonts.py:1-9` states it correctly.
- **Missing `fluidsynth` raises `FileNotFoundError`, not `RuntimeError`.** `_synthesize_midi`'s docstring promises `RuntimeError` "if fluidsynth is not available," but a missing binary makes `subprocess.run` raise `FileNotFoundError` (the `RuntimeError` path is only for a non-zero exit). In `/auralize` both are caught by the blanket `except Exception` and become a 500; on the CLI a missing binary surfaces as an uncaught `FileNotFoundError`.
- **FluidSynth argument order is not cosmetic.** Options before positionals, or FluidSynth ≥ 2.5 exits 0 and writes nothing (`auralization.py:60-62`).
- **RMS matching has no limiter.** Scaling the synthesis to the original's RMS can push its peaks past full scale; there is no clip guard before `sf.write`, so loud transcriptions of quiet-RMS originals can distort in the right channel.

## Cross-references

- `01-pipeline.md` — the end-to-end transcription pipeline the server drives.
- `02-model-core.md` — `LMModel.generate`, the token stream the SSE events are decoded from.
- `03-conditioning-audio.md` — audio loading/resampling (`load_audio`, `_read_wav_file`, `_read_non_wav_file`) and mel conditioning shared with the server's decode path.
- `04-tokenizer.md` — `MT3_FULL_PLUS_GROUP_NAMES`, instrument groups, and the `forbidden_token_ids` constraint the `instruments` field triggers.
- `05-events-decoding.md` — `NoteStartEvent` / `NoteEndEvent` / `ProgressEvent`, `decode_model_tokens`, and `events_to_midi_bytes`, i.e. the objects serialized over the wire.
- `06-cli-packaging.md` — the `serve` command's Typer wiring, `--auralize`, `download_if_necessary`, and `web_dist` packaging.
- `08-tests.md` — `tests/test_server.py`, the hermetic (fake-transcriber) tests that pin the SSE wire format.
