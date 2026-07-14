# CLI, Entry Points, Model Download & Packaging

This chapter covers the thin outer shell of muscriptor: the command-line
interface a user actually runs (`muscriptor transcribe …`), the module and
console entry points that reach it, the utility that turns a `--model` value
into a local weights file (downloading and caching as needed), and the
`pyproject.toml` that declares dependencies, the console script, and how the
wheel is built. Everything here is glue: it parses arguments, resolves a few
values, loads a `TranscriptionModel` (chapter 02) and drives its `transcribe`
generator (chapter 01), then serializes the resulting event stream (chapter 05)
to MIDI, JSON, or JSONL. The heavy lifting lives elsewhere; this layer's job is
to expose it cleanly, keep stdout uncontaminated when a user pipes output, and
fail with a readable message instead of a traceback when a download needs
authentication.

## Entry points: `main`, `__main__`, `__init__`

There are three ways into the same code:

- **Console script.** `pyproject.toml:64-65` declares
  `[project.scripts] muscriptor = "muscriptor.main:main"`. Installing the
  package (or `uvx muscriptor …`) puts a `muscriptor` executable on PATH that
  calls `muscriptor.main.main`.
- **`python -m muscriptor`.** `muscriptor/__main__.py` is three lines —
  `from muscriptor.main import main` then `main()` — so the module form runs the
  identical entry point.
- **`main()` itself** (`muscriptor/main.py:331-332`) is a one-line wrapper,
  `def main(): app()`, that invokes the Typer application. Keeping `main` as a
  named function (rather than pointing the console script at `app` directly) is
  what lets both the console script and `__main__.py` share one target.

The package's public API is declared in `muscriptor/__init__.py`:

```python
__all__ = ["TranscriptionModel", "Note", "NoteStartEvent", "NoteEndEvent"]
```

That is the entire supported import surface for library users:
`TranscriptionModel` (the model wrapper), `Note` (the tokenizer's note
dataclass), and the two note events. Note what is *absent*: there is **no
`__version__`** anywhere in the package (the version string lives only in
`pyproject.toml:7`, `0.2.2a1`), and `ProgressEvent` is not re-exported even
though `transcribe` yields it — library consumers who want progress anchors must
import it from `muscriptor.events` directly.

## The Typer application

`muscriptor/main.py:20` builds the app:

```python
app = typer.Typer(add_completion=False, help="muscriptor — audio-to-MIDI transcription")
```

`add_completion=False` suppresses Typer's default `--install-completion` /
`--show-completion` options, so the only subcommands are the three registered
below. Each is a function decorated with `@app.command()`; Typer derives the
command name from the function name, converting underscores to hyphens (so
`def list_instruments` becomes the `list-instruments` command).

| Command | Function | Purpose |
| --- | --- | --- |
| `transcribe` | `main.py:49` | Audio file → MIDI / JSON / JSONL |
| `serve` | `main.py:289` | Run the HTTP + web-UI server |
| `list-instruments` | `main.py:324` | Print the `--instruments` vocabulary |

## `transcribe` — options

The command signature runs `muscriptor/main.py:49-169`. It takes one positional
argument and a large set of options:

| Option | Flags | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| audio file | *(positional)* | `Path` | required | Input audio (wav, mp3, flac, …) |
| output | `--output` / `-o` | `Path` | derived | Output path; `-` = stdout |
| format | `--format` / `-f` | `midi`/`json`/`jsonl` | `midi` | Output serialization (case-insensitive) |
| notes | `--notes` | flag | `False` | Echo decoded events to stderr |
| sampling | `--sampling` | flag | `False` | Temperature sampling vs. greedy |
| temperature | `--temperature` / `-t` | `float` | `1.0` | Sampling temperature (only with `--sampling`) |
| cfg coefficient | `--cfg-coef` | `float` | `1.0` | Classifier-free-guidance strength |
| model | `--model` / `-m` | `str` | `None` | Size keyword, local path, or `hf://` / `http(s)://` URL |
| device | `--device` / `-d` | `str` | `"auto"` | `auto`/`cpu`/`cuda`/`cuda:0`/… |
| batch size | `--batch-size` / `-b` | `int` | `None` | Generation batch size |
| strict EOS | `--strict-eos` | flag | `False` | Error (not warn) on missing EOS |
| beam size | `--beam-size` | `int` | `1` | Beam width (1 = greedy/sampling) |
| auralize | `--auralize` | `Path` | `None` | Write stereo original-vs-synthesis audio |
| soundfont | `--soundfont` | `Path` | `None` | `.sf2` SoundFont for auralization |
| instruments | `--instruments` | `str` | `None` | Comma-separated instrument allow-list |

A precision note on the short flags: `-t` is bound to `--temperature`, and
`--sampling` is a *separate* boolean with no short form. Temperature is only
consulted when `--sampling` is set (greedy decoding ignores it), but the CLI
always forwards both to `transcribe`, so passing `-t 0.8` without `--sampling`
is silently inert.

`--model` accepts a size keyword (`small`/`medium`/`large`), a local
`.safetensors` path, or an `hf://` / `http(s)://` URL. `None` (the default) is
resolved downstream to the `medium` HuggingFace variant by
`TranscriptionModel._resolve_source`. `--batch-size None` is likewise resolved
downstream: `transcribe` picks 4 on CUDA and 1 on CPU
(`transcription_model.py:332-333`).

### Execution flow

The body of `transcribe` runs a fixed sequence; the ordering matters for both
correctness and where errors surface.

1. **Resolve `--instruments`** (`main.py:171-183`). If given, the string is
   split on commas, blank tokens dropped, and the remainder passed to
   `resolve_instrument_names` (`muscriptor/tokenizer/mt3.py:174`). That function
   does forgiving matching — exact (case-insensitive), then unique substring
   (`"timp"` → `"timpani"`), then a `difflib` "did you mean …?" suggestion for a
   miss. A `ValueError` (ambiguous or unknown) is caught and reported as
   `Error: {e}. Run 'muscriptor list-instruments' to see available names.` on
   **stderr**, then `raise typer.Exit(1)`. On success the resolved canonical
   names are echoed as `Instruments: …` to stderr. The resolved list — not the
   raw user tokens — is what reaches the model.
2. **Existence check** (`main.py:185-187`). If the audio file does not exist,
   print `Error: file not found: …` to stderr and exit 1. (This checks only the
   audio file, never the weights — see Gotchas.)
3. **Detect stdout** (`main.py:189`):
   `is_stdout = output is not None and str(output) == "-"`.
4. **Derive the default output path** (`main.py:191-197`). When `--output` is
   omitted, the suffix is chosen by format and applied with `with_suffix`:

   ```python
   suffix = {OutputFormat.midi: ".mid", OutputFormat.json: ".json",
             OutputFormat.jsonl: ".jsonl"}[format]
   output = audio_file.with_suffix(suffix)
   ```

   `Path.with_suffix` **replaces** the final extension, so `song.wav` →
   `song.mid`. This means the default output can silently sit next to (or
   overwrite) an existing sibling with that name.
5. **Map the device** (`main.py:199`):
   `_device = None if device == "auto" else device`. The sentinel `"auto"` maps
   to `None`, which `load_model` turns into CUDA-if-available-else-CPU.
6. **Load the model** (`main.py:203-204`). `Loading model…` is echoed to
   **stderr** (`err=True`), then `_load_model(model_path, _device)` runs.
7. **The fp32 cast** (`main.py:205-207`) — see the dedicated section below.
8. **Validate `--auralize`** (`main.py:211-213`). If `--auralize` is set but the
   format is not `midi`, print `Error: --auralize requires --format midi` and
   exit 1. This check runs *after* the model is already loaded.
9. **Build the transcribe kwargs** (`main.py:215-224`) and dispatch on format.

The keyword bundle passed into the model is:

```python
kwargs = dict(
    audio=audio_file, use_sampling=sampling, temperature=temperature,
    cfg_coef=cfg_coef, instruments=instrument_names, batch_size=batch_size,
    no_eos_is_ok=not strict_eos, beam_size=beam_size,
)
```

Two translations happen here. `--strict-eos` is inverted into
`no_eos_is_ok=not strict_eos`: by default (`strict_eos=False`) a chunk that
never emits EOS within the generation budget only warns and keeps its notes;
with `--strict-eos` the model raises `RuntimeError`
(`transcription_model.py:500-505`). And `instruments=instrument_names` carries
the *resolved* names, not the raw string. Options that are purely
CLI-side — `output`, `format`, `notes`, `auralize`, `soundfont` — are not in the
bundle.

### Output formats

The three formats diverge in how they consume and serialize the event stream.
All three drop `ProgressEvent` — those coarse chunk anchors are advisory and
never appear in output.

**`midi`** (`main.py:226-248`) calls `model.transcribe_to_midi(**kwargs)`, which
internally reassembles notes and returns a `.mid` file as `bytes`. With
`-o -`, the bytes go straight to `sys.stdout.buffer` (binary) and are flushed;
otherwise `output.write_bytes(...)` writes the file and `Saved MIDI to …` is
echoed to stderr. `--notes` in this branch does *not* print events (the MIDI
path never materializes them); it prints a hint to re-run with `--format json`.
If `--auralize` is set (and not writing to stdout), the branch lazily imports
`muscriptor.utils.auralization.auralize` and calls it with the just-written MIDI
file, the original audio, the auralization path, and the optional soundfont
(deep coverage in 07-server-web.md).

**`jsonl`** (`main.py:249-270`) streams one compact JSON object per line,
flushing after each so a downstream consumer (file tail or stdout pipe) sees
events live:

```python
for e in model.transcribe(**kwargs):
    if isinstance(e, ProgressEvent):
        continue
    sink.write(json.dumps(_event_to_dict(e)) + "\n")
    sink.flush()
```

The sink is `sys.stdout` for `-o -` (never closed) or `output.open("w")`
otherwise (closed in a `finally`). `--notes` additionally echoes `str(e)` to
stderr per event.

**`json`** (`main.py:271-286`) is the buffered counterpart: it materializes the
whole stream into a list, then writes a single pretty-printed array
(`json.dumps([...], indent=2)`). It is not live — nothing is emitted until
transcription finishes. With `-o -` the array plus a trailing newline goes to
stdout; otherwise `output.write_text(...)` and `Saved JSON to …` to stderr.
`--notes` prints every event to stderr after the write.

The essential contrast: **jsonl** is a live, per-line, flushed stream of compact
objects; **json** is one indented array produced only after the full stream is
collected in memory.

### The stdout/stderr contract

The command treats stdout as reserved for machine-readable output whenever
`-o -` is used, and routes *everything* chatty — banners, timing, `Saved … to`,
the `Instruments:` line, `--notes` output — to stderr via `typer.echo(...,
err=True)` or explicit `file=sys.stderr` prints inside the model. `tests/test_cli.py`
locks this down: `test_jsonl_stdout_has_no_chatter` asserts every non-empty
stdout line parses as JSON, and `test_progress_messages_go_to_stderr` asserts
the `Loading model` / `Transcribing` banners appear on stderr and never on
stdout. This is what makes `muscriptor transcribe in.wav -f jsonl -o - | jq …`
work cleanly.

### Event serialization: `_event_to_dict`

`main.py:39-46` converts a note event to a JSON-ready dict:

```python
def _event_to_dict(ev):
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {"type": "end", "end_time": ev.end_time,
            "start_event_index": ev.start_event_index}
```

Start events are flattened with `dataclasses.asdict`, giving
`{type, pitch, start_time, index, instrument}`. End events are built by hand and
deliberately *not* run through `asdict`: `NoteEndEvent` holds a reference to its
`start_event` (a nested `NoteStartEvent`), so `asdict` would recursively inline
the entire start event. Instead the end dict carries only `end_time` and
`start_event_index` (a property on `NoteEndEvent`,
`muscriptor/events.py:33-35`, returning `self.start_event.index`), so a consumer
matches an end to its start by index. This exact serialization is duplicated as
the public `event_to_dict` in `muscriptor/server.py:58-65` — the server's SSE
events share the same wire shape (the server docstring points back at
`muscriptor.main._event_to_dict` as the reference).

### `_load_model` and clean download failures

`main.py:23-30` wraps `TranscriptionModel.load_model`:

```python
def _load_model(model_path, device):
    try:
        return TranscriptionModel.load_model(weights_path=model_path, device=device)
    except ModelDownloadError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
```

Only `ModelDownloadError` (raised by the download utility for problems the user
must fix, chiefly missing HuggingFace auth) is caught. Its message — already
composed to be shown verbatim — is printed to stderr and the process exits 1
with **no traceback**. Any other exception (a corrupt checkpoint, a missing
*local* weights file, a CUDA OOM) propagates normally and shows a traceback.

## The surprising fp32 cast

Immediately after loading, `transcribe` does something the `serve` path does
not (`main.py:205-207`):

```python
import torch
model._model = model._model.to(torch.float32)
```

`load_model` has already built the `LMModel` on the target device and loaded the
checkpoint's weights (`transcription_model.py:286-294`). This line reaches into
the private `_model` attribute and casts every parameter and buffer of that
inner module to fp32, regardless of the dtype the safetensors file stored. Its
purpose is to guarantee fp32 *master weights* for numerical stability.

Crucially, this does **not** force fp32 compute on CUDA. `_build_model` attaches
a `TorchAutocast(enabled=True, device_type="cuda", dtype=torch.float16)` to the
model when the device is CUDA (`transcription_model.py:204-206`), and generation
runs inside that autocast. Under autocast the heavy ops (matmuls/linears) still
downcast their inputs to fp16 on the fly, so the model keeps fp16 matmul
throughput while holding fp32 parameters and running the autocast-exempt ops
(normalization, softmax, embeddings) in fp32. On CPU there is no autocast, so
after the cast everything runs fp32. The autocast mechanics are covered in
02-model-core.md.

The `import torch` on `main.py:205` is a local rebind, not a lazy-load
optimization: `torch` is already imported transitively when `main.py` imports
`TranscriptionModel` at module load. `tests/test_cli.py` models this cast with a
`_FakeInner.to()` that returns `self`, confirming the CLI calls `.to(...)` on
the inner model object.

**The `serve` path does not perform this cast** (verified in `main.py:289-321`
and `muscriptor/server.py`; neither touches `model._model.dtype`). So if a
published checkpoint ships fp16 weights, the CLI `transcribe` runs it with fp32
parameters while the server runs it with fp16 parameters — a genuine, if subtle,
divergence between the two front ends.

## `serve` — options and behavior

`main.py:289-321`. The server command has four options:

| Option | Flags | Type | Default |
| --- | --- | --- | --- |
| host | `--host` | `str` | `"127.0.0.1"` |
| port | `--port` | `int` | `8222` |
| model | `--model` / `-m` | `str` | `None` |
| device | `--device` / `-d` | `str` | `"auto"` |

It resolves the device exactly like `transcribe` (`"auto"` → `None`), loads the
model through the same `_load_model` (so gated-repo failures are equally clean),
then imports `uvicorn` and `muscriptor.server.create_app` **lazily** inside the
function — a plain `muscriptor transcribe` never pays the FastAPI/uvicorn import
cost. The web assets directory is computed as
`Path(__file__).resolve().parent / "web_dist"` and passed to `create_app` only
if it is a real directory (otherwise `None`, and the server runs API-only).
Finally `uvicorn.run(fastapi_app, host=host, port=port)` blocks.

One inconsistency worth flagging: the serve command's `Loading model…` banner
(`main.py:317`) is echoed **without** `err=True`, so it goes to *stdout* — unlike
the transcribe command, whose identical banner (`main.py:203`) is on stderr. The
server itself (routes, SSE streaming, concurrency, auralization endpoints) is
covered in 07-server-web.md.

## `list-instruments`

`main.py:324-328` is the simplest command: no options, it iterates
`MT3_FULL_PLUS_GROUP_NAMES` (`muscriptor/tokenizer/mt3.py:118-154`) and echoes
each key on its own line to stdout. That dict maps 35 human-readable group names
(`acoustic_piano`, `electric_piano`, … `synth_pad`, `drums`) to the model's
internal group IDs. These names are exactly the vocabulary the `--instruments`
option accepts, and the same set the web UI's `/instruments` endpoint returns.

## `utils/download.py` — resolving weights to a local file

`download_if_necessary(url)` (`muscriptor/utils/download.py:39`) is the single
funnel that turns any `--model`-derived source into a local `Path`. The cache
root is `_CACHE_DIR = Path.home() / ".cache" / "muscriptor"`
(`download.py:16`). There are three source forms, distinguished by prefix:

**`hf://<org>/<name>/<path/in/repo>`** (`download.py:53-66`). The string after
`hf://` is split into `org`, `name`, and the in-repo filename; `repo_id` is
`org/name`; the file is fetched with `hf_hub_download` (which uses
HuggingFace's own cache under `~/.cache/huggingface`, *not* `_CACHE_DIR`). Error
translation is the point of this branch:

```python
except (GatedRepoError, RepositoryNotFoundError) as e:
    raise ModelDownloadError(_auth_help(repo_id)) from e
except HfHubHTTPError as e:
    if getattr(e.response, "status_code", None) in (401, 403):
        raise ModelDownloadError(_auth_help(repo_id)) from e
    raise
```

Both the typed gated/not-found errors and a raw 401/403 HTTP error become a
`ModelDownloadError` carrying `_auth_help` (`download.py:25-36`) — a multi-line
message telling the user to accept the model license on the hub page, then
either `uvx hf auth login` or set `HF_TOKEN`. Any other HTTP error re-raises
unchanged. `tests/test_download.py` pins this: both `GatedRepoError` and
`RepositoryNotFoundError` must surface a `ModelDownloadError` whose text mentions
`hf auth login`, `HF_TOKEN`, and the repo URL.

**`http://` / `https://`** (`download.py:68-87`). The cache dir is created; the
destination filename is the URL's last path segment (query string stripped),
**prefixed with the first 8 hex chars of the URL's SHA-256** so two different
URLs that happen to share a basename don't collide on one cache file. If the
destination already exists it is returned immediately. Otherwise the download is
atomic: it writes to a per-process temp file `…​.part<pid>`, then `os.replace`s
it onto the final name, with a `finally` that unlinks the temp
(`missing_ok=True`). This guarantees an interrupted or concurrent download can
never leave a partial file where a complete one is expected.

**Local path** (`download.py:89-93`). Anything else is treated as an existing
local file: `Path(url)` is checked with `.exists()`, returning it if present or
raising a plain `FileNotFoundError` if not. Note that this is *not* a
`ModelDownloadError`, so `_load_model` does not catch it — a bad `--model
/no/such.safetensors` surfaces as a traceback, not a clean message.

`download_companion(url, filename)` (`download.py:96-111`) is a best-effort
sibling fetch used by the model loader to grab a `config.json` next to the
weights. It only acts on `hf://` URLs (returns `None` for anything else),
`hf_hub_download`s the requested filename from the same repo, and swallows
`EntryNotFoundError` / `HfHubHTTPError` into `None` so a missing, gated, or
offline config lets `_resolve_config` fall back to other detection schemes
(repo-name regex, legacy filename tag) rather than failing the whole load.

## `pyproject.toml` — dependencies, script, packaging

**Build backend.** `hatchling` (`pyproject.toml:1-3`). Version is a static
`0.2.2a1` (`:7`); `requires-python = ">=3.10"` (`:10`); MIT-licensed.

**Console script.** `[project.scripts] muscriptor = "muscriptor.main:main"`
(`:64-65`) — the entry point discussed above.

**Dependencies with platform markers** (`:25-44`). Most pins are ordinary
(`einops`, `mido`, `safetensors`, `huggingface_hub`, `typer`, `fastapi`,
`uvicorn[standard]`, `httpx`, `python-multipart`, `soundfile`). The interesting
ones are `torch` and `numpy`, each split by an environment marker for Intel
macs:

```toml
"torch>=2.0 ; sys_platform != 'darwin' or platform_machine != 'x86_64'",
"torch>=2.0,<2.3 ; sys_platform == 'darwin' and platform_machine == 'x86_64'",
"numpy>=1.24 ; sys_platform != 'darwin' or platform_machine != 'x86_64'",
"numpy>=1.24,<2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'",
```

The in-file comment explains why: PyTorch stopped publishing Intel-mac
(`darwin` `x86_64`) wheels after 2.2.2, which itself only ships wheels up to
Python 3.12 and was compiled against NumPy 1.x. So on an Intel mac the resolver
must cap `torch < 2.3` and `numpy < 2` and run under Python ≤ 3.12 (the README
suggests `uvx --python 3.12 muscriptor`); every other platform gets the
unconstrained `torch >= 2.0` / `numpy >= 1.24`.

**Windows CUDA pin** (`:49-57`). PyPI only publishes CPU-only torch wheels for
Windows, so a plain install there yields a torch that can't see the GPU. This
block redirects the resolver:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

On Windows, `uv` pulls torch from PyTorch's official CUDA 12.8 index rather than
PyPI; `explicit = true` means that index is used *only* for packages that name
it (here, torch), not consulted for general resolution. Other platforms are
untouched — Linux PyPI wheels already bundle CUDA. (This is a `uv`-specific
mechanism; pip does not read `[tool.uv.*]`.)

**Wheel/sdist build config** (`:67-73`).

```toml
[tool.hatch.build]
artifacts = ["muscriptor/web_dist"]

[tool.hatch.build.targets.wheel]
packages = ["muscriptor"]
```

`muscriptor/web_dist` is the compiled frontend (built by `pnpm build` in
`web/`). It is gitignored, so it would normally be excluded from the sdist and
wheel; listing it under `artifacts` forces it into the built distributions so
that `uvx muscriptor serve` has a UI to mount. `packages = ["muscriptor"]` names
the importable package. This is the packaging counterpart to `serve`'s runtime
`web_dist.is_dir()` check: the wheel ships the directory, and the server mounts
it if present.

**Dev group** (`:75-79`). `[dependency-groups] dev` lists `pre-commit` and
`pytest` — the tooling for the test suite (chapter 08) and pre-commit hooks, not
installed for normal users.

## Gotchas & invariants

- **stdout is sacred under `-o -`.** Every human-facing message uses
  `err=True` (or `file=sys.stderr`). Breaking this would corrupt piped
  output and fail `tests/test_cli.py`.
- **`serve`'s `Loading model…` goes to stdout**, not stderr — the lone
  exception to the rule above (`main.py:317` omits `err=True`).
- **Default output can overwrite siblings.** `audio_file.with_suffix(...)`
  *replaces* the extension, so `song.wav` → `song.mid`, and writes
  (`write_bytes`/`write_text`/`open("w")`) never check for an existing file.
- **The audio-file existence check does not cover weights.** A missing local
  `--model` path raises `FileNotFoundError` (a traceback), because that error is
  not a `ModelDownloadError` and `_load_model` only catches the latter. Only the
  audio file gets the clean `file not found` message.
- **`--auralize` is validated after the model loads** (`main.py:211-213`), so a
  `--auralize x.wav -f json` mistake pays the full model-load cost before
  erroring.
- **`--auralize` is silently skipped with `-o -`.** The auralization branch is
  guarded by `not is_stdout` (`main.py:238`) because it reads the MIDI back from
  the on-disk output file, which does not exist when streaming to stdout. No
  error, no auralization.
- **`--soundfont` without `--auralize` does nothing.** It is only read inside
  the auralize branch (`main.py:246`).
- **`--temperature` without `--sampling` does nothing** — temperature only
  applies under sampling, but is always forwarded.
- **`--notes` is format-dependent.** It prints events to stderr for `json`/
  `jsonl`, but for `midi` only prints a hint to re-run with `--format json`.
- **fp32 cast asymmetry.** CLI `transcribe` casts inner weights to fp32;
  `serve` does not. On CUDA both still compute matmuls in fp16 via autocast, but
  a fp16 checkpoint runs with different parameter precision on the two paths.
- **`ProgressEvent` is filtered from all outputs** and is not part of the
  package's public `__all__`.
- **HTTP cache filenames are hash-prefixed and downloads are atomic.** Two URLs
  sharing a basename never collide; an interrupted download never leaves a
  partial file at the final path (`…​.part<pid>` + `os.replace`).
- **`hf://` weights use HuggingFace's own cache**, not `~/.cache/muscriptor/`;
  only `http(s)://` downloads land under `_CACHE_DIR`.
- **No `__version__`.** The version exists only in `pyproject.toml`; there is no
  runtime accessor in the package.
- **`web_dist` ships only because it is an explicit hatch artifact.** It is
  gitignored; without the `artifacts` entry the wheel would have no UI.

## Cross-references

- **01-pipeline.md** — the end-to-end transcribe pipeline this CLI drives.
- **02-model-core.md** — `LMModel`, `TorchAutocast`, and how the fp32 weight
  cast interacts with fp16 CUDA compute.
- **03-conditioning-audio.md** — audio loading and conditioning behind
  `transcribe`.
- **04-tokenizer.md** — `MT3_FULL_PLUS_GROUP_NAMES`, `resolve_instrument_names`,
  `forbidden_token_ids` (the `--instruments` machinery).
- **05-events-decoding.md** — `NoteStartEvent` / `NoteEndEvent` /
  `ProgressEvent` and the decode stream that `_event_to_dict` serializes.
- **07-server-web.md** — the `serve` command's FastAPI app, SSE streaming, and
  the auralization handoff.
- **08-tests.md** — `tests/test_cli.py` (stdout/stderr contract) and
  `tests/test_download.py` (gated-repo error translation).
