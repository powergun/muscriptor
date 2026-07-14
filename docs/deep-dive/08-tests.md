# The test suite

The `tests/` directory is muscriptor's executable specification of its own behavior. Almost all of it runs on CPU with no model weights and no network: the decoder, tokenizer, sampling utilities, CLI, and HTTP server are each exercised against synthetic inputs or fakes, so `uv run pytest` is fast and hermetic on a clean checkout. A separate band of *integration* tests loads a real safetensors checkpoint and runs inference end to end; those self-skip when the weights (or the reference song) are absent, which is the default on any machine that has not downloaded them. The suite's center of gravity is the streaming decode path — the machinery that turns a flat stream of model token indices into matched `NoteStart`/`NoteEnd` events — because that is where the package does its most intricate work and where a vocab or arithmetic change is most likely to break something silently.

The count is roughly 130 test functions across twelve files. `test_notes.py` (26) and `test_integration.py` (24) are the largest; the integration file is the only one that is normally skipped. This chapter documents every file, the shared fixtures, and the test-only encoder in `tests/encode_helpers.py` that lets the decoder be tested without ever running the model.

## How to run the suite

The project uses `uv` and declares `pytest>=9.0.3` in its dev dependency group (`pyproject.toml:75-79`). The canonical invocation is:

```
uv run pytest              # whole suite (integration tests skip without weights)
uv run pytest -v           # per-test names + SKIPPED reasons
uv run pytest tests/test_events.py       # one file
uv run pytest tests/test_integration.py -v   # only the model-backed tests
```

There is **no** `[tool.pytest.ini_options]` table in `pyproject.toml` and no `pytest.ini`/`setup.cfg`, so pytest runs with defaults: it discovers `tests/` by the `test_*.py` convention and imports the package under test from the working tree. `tests/` is an importable package (`tests/__init__.py` exists, though empty), which is why `test_events.py:21` can do `from tests.encode_helpers import encode_note_events` — the helper is shared code, not a test module.

### Skip and gating conditions

Gating is done entirely through fixtures that call `pytest.skip()`, **not** through markers. `tests/conftest.py:12-13` registers an `integration` marker via `config.addinivalue_line`, but no test is actually decorated with `@pytest.mark.integration` — the only `@pytest.mark` in the suite is the `parametrize` in `tests/test_download.py:10`. So `pytest -m "not integration"` does *not* deselect the model-backed tests; they are instead skipped from inside their fixtures:

- **Weights-dependent tests** (`test_integration.py`, and the `transcription_model` fixture in `conftest.py:16-23`) skip when `WEIGHTS_PATH` does not exist. `WEIGHTS_PATH` is hard-coded to `muscriptor_weights_01684fbb_350.safetensors` in the repo root (`conftest.py:6-8`); on a checkout without that file every test that requests the `transcription_model` fixture reports `SKIPPED (Weights not found …)`.
- **Real-audio tests** additionally need a specific WAV. `SONG_PATH` is hard-coded to `/home/simon/audio/filling_the_void.wav` (`conftest.py:9`), an absolute path on the original author's machine. The `song_clip` fixture (`test_integration.py:38-46`) and `test_transcribe_song_from_file_path` (`test_integration.py:320-326`) skip when it is missing, so those never run for anyone else even when the weights are present.

No test reaches the network. `test_download.py` monkeypatches `hf_hub_download` so the HuggingFace error paths are simulated, never dialed; the server tests use a mock model and never download the soundfont; the conditioner/transformer/sampling tests build tiny random-weight modules in-process.

### Rough runtimes

The hermetic tests are effectively instantaneous — tiny tensors, one- or two-layer models, no I/O — so the non-integration suite completes in a few seconds. The heaviest hermetic construction is the `tiny_model` fixture (`test_strict_instruments.py:86-101`): a one-layer, dim-16 `LMModel` decoded for eight steps. The integration tests are, per their own module docstring (`test_integration.py:1-7`), "slow (model is large) but run on CPU with short audio clips" — they load the large checkpoint once (session-scoped fixture) and transcribe 5- and 10-second clips.

## Shared fixtures — `conftest.py`

`conftest.py` is deliberately small (23 lines). It defines two module-level path constants (`WEIGHTS_PATH`, `SONG_PATH`), registers the unused `integration` marker, and provides exactly one fixture:

- **`transcription_model`** (`conftest.py:16-23`), `scope="session"`. Skips if `WEIGHTS_PATH` is missing; otherwise imports `TranscriptionModel` lazily and returns `TranscriptionModel.load_model(weights_path=WEIGHTS_PATH, device="cpu")`. Session scope means the multi-hundred-megabyte checkpoint is loaded once for the entire run and shared by every integration test. `SONG_PATH` is exported for import by `test_integration.py` (`from .conftest import SONG_PATH`) but is not itself a fixture.

All other fixtures are local to the file that uses them (e.g. `fake_audio`/`patched_model` in `test_cli.py`, `silence`/`noise`/`song_clip` in `test_integration.py`, `tokenizer`/`tiny_model` in `test_strict_instruments.py`).

## The synthetic encoder — `tests/encode_helpers.py`

This 229-line module is the single most important piece of test infrastructure, and it exists for a specific architectural reason spelled out in its own docstring (`encode_helpers.py:1-8`): the shipped package only ever decodes **forward**, from token indices straight to streamed events (`muscriptor.events.decode_model_tokens`). The reverse direction — building tokens from note events — and the older per-chunk *note-list* decode were part of the original YourMT3-style tokenizer but are not used at runtime. Rather than leave that code in the package as dead weight, it lives here as a test fixture. That gives the tests two things: an **encoder** to manufacture the exact token streams a model would emit, and an **independent decoder** to cross-check the arithmetic. Because the encoder mirrors the vocab layout by hand, any drift between it and `build_event_vocab` is itself a test failure.

The API, in dependency order:

- **`note_event2event(note_events, tie_note_events=None, start_time=0.0, frame_rate=100) -> list[Event]`** (`encode_helpers.py:23-86`). The core encoder. It sorts note events by `(tick, is_drum, program, velocity, pitch)`, emits a **tie prologue** (`program`/`pitch` pairs for notes sustained from a previous chunk) terminated by a single `tie` token, then walks the events emitting `shift` tokens (absolute tick minus `start_tick`, only when time advances), `program`/`velocity` state tokens (only when the running state changes), and `pitch`/`drum` value tokens. Drums are special-cased: velocity is forced to `1` and the pitch is emitted as a `drum` token, with no offset. This is the mirror image of the decoder's state machine — it encodes the same run-length assumptions (program/velocity are sticky, shifts are monotonic deltas from the chunk start) that the decoder relies on.

- **`encode_index_map(max_shift_steps) -> dict[(type, value), int]`** (`encode_helpers.py:89-93`). The inverse of `build_event_vocab`: it enumerates the canonical vocab and maps each `(type, value)` back to its index. Its docstring names it "the inverse of `build_event_vocab`", and that is exactly the invariant `test_notes.py` checks by round-tripping through it.

- **`encode_note_events(note_events, max_shift_steps, tie_note_events=None, start_time=0.0, frame_rate=100) -> list[int]`** (`encode_helpers.py:96-106`). The one-call encoder tests actually use: `note_event2event` to get `Event`s, then `encode_index_map` to turn each into its integer token id. This is what lets a test say "a note on at 0.1s and off at 0.5s" and get back the precise `list[int]` the model would have produced.

- **`event2note_event(events, start_time=0.0, frame_rate=100)`** (`encode_helpers.py:109-211`). The test-only *note-list* decoder (distinct from the streaming `decode_model_tokens`). It parses the tie prologue into `TieNoteEvent`s, then replays shifts/programs/velocities/pitches into `NoteEvent`s, accumulating a `Counter` of typed error strings ("Err/Negative shift", "Err/Note off without note on", …) instead of raising. Returns `(note_events, tie_note_events, last_activity, err_cnt)`. Tests assert `not err` to prove a clean round-trip.

- **`decode_tokens(tokens, vocab, start_time=0.0, frame_rate=100)`** (`encode_helpers.py:214-229`). Thin wrapper: maps each token index through `vocab` (raising `ValueError` on an out-of-range index — the behavior `test_notes.py:164` pins) and delegates to `event2note_event`.

The vocab constants it mirrors come straight from `muscriptor.tokenizer.notes`: it imports `DRUM_PROGRAM`, `Event`, `NoteEvent`, `TieNoteEvent`, `build_event_vocab`, and the sort helpers (`encode_helpers.py:12-20`). It hard-codes no token indices — everything is derived from `build_event_vocab(max_shift_steps)` — which is precisely why it stays correct as long as the layout function does.

## Per-file walkthrough

### `test_notes.py` — tokenizer primitives and round-trips

The broadest unit file (26 tests). It exercises `muscriptor/tokenizer/notes.py` together with the test-only encoder, in five clusters:

- **Sorting** (`test_notes.py:33-53`): `sort_notes` orders by onset, `sort_note_events` by time.
- **`validate_notes`** (`:60-75`): a sub-`MINIMUM_NOTE_DURATION_SEC` note is lengthened, an inverted (onset > offset) note is fixed, a `None`-onset note is dropped.
- **`trim_overlapping_notes`** (`:83-109`): same-pitch overlaps are trimmed so the earlier note's offset meets the later's onset; different pitches are left alone. `test_trim_overlapping_notes_different_pitch` is the instructive one — it proves trimming is per-`(program, pitch, is_drum)` channel, not global.
- **Vocab layout** (`:117-167`). `test_event_vocab_num_tokens` (`:117`) is the canary for the whole tokenizer: it asserts `len(build_event_vocab(206)) == 3 + 206 + 128 + 2 + 1 + 130 + 128` — special tokens, then shift, pitch, velocity, tie, program (0–129, i.e. 130 values), drum. Change any range and this fails first. The `roundtrip_*` tests confirm `encode_index_map` and `build_event_vocab` are mutual inverses for shift/pitch/velocity/program/drum; `test_decode_out_of_range_index_raises` (`:164`) pins the `ValueError` on a bad index.
- **Round-trips** (`:183-296`). `test_event_roundtrip_piano_note` and `test_encode_decode_roundtrip` are the most instructive: encode two `NoteEvent`s → tokens → decode, assert `not err` and that onset/offset times survive to within 1e-6. `test_note2note_event_roundtrip` (`:278`) checks that one `Note` expands to exactly two `NoteEvent`s (onset velocity 1, offset velocity 0). These lock the tokenizer's time quantization (round to `frame_rate` ticks) and its program/velocity run-length encoding.

### `test_events.py` — the streaming decoder

Twelve hermetic tests for `muscriptor.events.decode_model_tokens`, the actual runtime decode path. The technique is the payoff of `encode_helpers`: a private `_decode(*chunks)` helper (`test_events.py:44-62`) takes `(note_events, tie_note_events, seek_time, next_seek_time)` tuples, and for each one yields a `ChunkBoundary` followed by `encode_note_events(...)` tokens, then runs the whole interleaved stream through `decode_model_tokens`. `_MAX_SHIFT_STEPS = 1001` (`:24`) is chosen to match the real model's shift range so within-chunk shifts encode without saturating. No model, no audio.

What it locks in:

- **Basic matching** (`:70-111`): a single note yields a `NoteStart`/`NoteEnd` pair with the right pitch, times and `instrument` (from the injected `instrument_for_program`); note indices are unique and monotonic (`:83`); an orphan note-off produces nothing (`:96`); a re-trigger of an already-open note closes the previous instance before opening the new one (`test_retrigger_closes_previous`, `:101`).
- **Drums** (`:119-129`): a drum token emits a start/end pair whose end is `onset + MINIMUM_NOTE_DURATION_SEC` and whose instrument is the literal `"drums"`.
- **Chunk stitching** (`:136-179`) — the subtle part. `test_note_sustains_across_chunk_via_tie` proves a note left open at a chunk boundary survives when the next chunk's tie prologue re-declares `(program, pitch)`; `test_unsustained_note_closes_at_chunk_boundary` proves that without a tie it is closed at the boundary `seek_time`; `test_tie_for_unknown_note_is_ignored` proves a tie for a note that was never open is harmless.
- **Windowing & flush** (`:186-210`): events past `next_seek_time` are dropped (`test_events_past_next_seek_time_are_filtered`), and at end-of-stream any still-open note is closed with the minimum-duration fallback.
- **Global contract** (`test_every_start_has_exactly_one_end`, `:218`): a synthetic three-chunk session (with a cross-chunk tie and a trailing drum) asserts `len(starts) == len(ends)`, that the index sets match, and that indices are unique — the guarantee `events.py` advertises in its module docstring.

### `test_transcription_model.py` — the token-stream interleaver

Six tests for `TranscriptionModel._generate_token_stream`, the generator that pulls one token-per-chunk-per-timestep out of the model and re-serializes it into whole chunks in order. The fake (`test_transcription_model.py:21-53`) is clever: `generate()` yields one row per timestep from a scripted `batches` list and appends to a shared `pulled` list, so a test can assert not just *what* comes out but *how far generation had progressed* when it did. `EOS = 99` stands in for the tokenizer's EOS id.

- **Streaming eagerness** (`:61-87`): `test_first_chunk_streams_before_the_batch_finishes` drives a 2-chunk batch where chunk 0 ends at row 2 but chunk 1 not until row 4, and asserts chunk 0's tokens surface after only two timesteps — the decoder does not wait for the whole batch. The `ChunkBoundary` is emitted before any generation (`len(pulled) == 0`).
- **Ordering & buffering** (`:94-139`): `test_later_chunk_finishing_first_is_buffered_until_its_turn` (`:111`) is the most instructive — chunk 1 hits EOS before chunk 0, and the test proves chunk 1's tokens are held back and only flushed after chunk 0 completes, preserving global order. `test_chunks_across_multiple_batches_stay_in_order` checks the `batch_size=1` path (one `generate()` call per chunk) and its per-chunk `ProgressEvent` anchors.
- **Missing EOS** (`:147-167`): with `no_eos_is_ok=False` a chunk that never emits EOS raises `RuntimeError("… did not emit EOS")`; with `no_eos_is_ok=True` it downgrades to a `RuntimeWarning` and still emits the tokens. This pins the `--strict-eos` CLI semantics.

### `test_strict_instruments.py` — instrument masking across all decode paths

Eleven tests, three layers, covering the "forbid every instrument not listed" feature (`pyproject.toml`/README call it a hard constraint). A module-scoped `tokenizer` fixture builds a real `MT3Tokenizer("MT3_FULL_PLUS", max_shift_steps=1001)` (`:20-22`).

- **`forbidden_token_ids`** (`:34-77`): `test_forbidden_ids_never_touch_timing_or_special_tokens` proves the mask only ever touches `program`/`drum` tokens — never PAD/EOS/UNK/shift/pitch/velocity/tie. `test_forbidden_ids_keep_only_listed_programs` proves only the *representative* (first) program of each listed group is allowed. `test_forbidden_ids_mask_drums_unless_listed` and `..._drums_only_masks_every_program` pin the drum rule. `test_forbidden_ids_rejects_unknown_names` checks the `ValueError` on a bogus name.
- **`LMModel.generate` masking** (`:86-147`) — the reason this file matters. A `tiny_model` fixture (one layer, dim 16, `CARD=16`) is generated for 8 steps with a `forbidden_tokens` list that leaves only one or two ids legal, across **all three decode paths**: greedy (`:109`), sampling at temp 2.0 (`:118`), and **beam search** with `beam_size=2` (`:128`). Each asserts the output set is a subset of the allowed ids. `test_generate_unmasked_uses_full_vocabulary` (`:141`) is the control: the same high-temperature sampling *without* a mask spreads over more than two tokens, proving the masked assertions actually bite. This is the only place beam search executes anywhere in the suite.
- **Automatic activation** (`:158-189`): `_forbidden_tokens_used_by_transcribe` fakes out `_load_wav`/`_build_conditions`/`_generate_token_stream` and captures the last positional argument handed to `_generate_token_stream` (which is `forbidden_tokens`). It proves that passing `instruments=["violin"]` produces exactly `tokenizer.forbidden_token_ids(["violin"])` and that passing `None` produces `None` — there is no separate flag; giving `instruments` *is* the switch.

### `test_sampling.py` — `utils/sampling.py`

Eleven tests over the pure sampling helpers, all shape- and support-focused (no distributional assertions beyond concentration):

- `length_to_mask` (`:14-41`): the basic mask, an explicit `max_len`, and the all-zero-lengths edge case (which still returns a width-1 mask because `final_length` is floored at 1).
- `multinomial` (`:43-53`): 1-D and batched shapes.
- `sample_top_k` (`:55-69`): `test_sample_top_k_only_top_tokens` concentrates mass on ids 0–4 and, over 30 draws, asserts nothing outside the top-k is ever sampled.
- `sample_top_p` (`:72-81`): shape and in-vocab support.
- `sample_stratified` (`:84-100`): `test_sample_stratified_returns_special_when_mass_concentrated` puts almost all mass on a "special" token and asserts it is picked ~always — the mechanism the model uses to bias toward/away from a reserved token.

### `test_conditioners.py` — `modules/conditioners.py` (CPU only)

Ten tests. Constructs small `MelSpectrogramConditioner` (`_make_mel_conditioner`, `:67-75`) and `ClassConditioner` (`:109-112`) instances directly:

- **`ConditioningAttributes`** (`:21-43`): defaults are empty dicts; `__getitem__` maps `"text"`/`"wav"` to the right dict; the `*_attributes` properties expose the keys.
- **`nullify_wav`** (`:51-59`): zeros the waveform and sets length 0 — the CFG null branch.
- **`MelSpectrogramConditioner`** (`:78-101`): forward output projects to `output_dim` and returns a `(embed, mask)` pair of the right batch shape; the mask dtype is one of bool/float32/int32. This is also the only coverage of `modules/mel_spectrogram.py`, exercised transitively.
- **`ClassConditioner`** (`:115-127`): string class indices `["3","7"]` embed to `(2,1,16)`; `None` inputs (the null class) still produce a batch-2 embedding.
- **`ConditioningProvider`** (`:135-149`) and **`nullify_all_conditions`** (`:157-168`): the provider tokenizes+forwards a full attribute list; nullify zeros the wav length and sets text to `None` while proving the original is untouched (deepcopy semantics).

### `test_transformer.py` — `modules/transformer.py` + `modules/streaming.py`

Six tests, small tensors:

- **`create_sin_embedding`** (`:17-34`): output shape for even dims, and that different positions produce different embeddings.
- **`StreamingTransformer`** (`:48-86`): a 2-layer, dim-32 model returns `[B,T,D]` unchanged in a batched forward. `test_streaming_transformer_streaming_mode` (`:57`) is the instructive case — it feeds a 6-token sequence one timestep at a time using `init_states`/`increment_steps` from `modules/streaming.py` and concatenates the per-step outputs, checking the streaming (KV-cached) path yields the same shape as a full forward. This is the only direct coverage of the streaming-state helpers, though it checks shape, not numerical equivalence to the non-streaming path.

### `test_midi.py` — `utils/midi.py`

Seven tests using a fixed three-note sample (two piano notes plus a drum hit, `:12-17`) and `mido` to validate output. They cover `notes_to_midi` (returns a `MidiFile` with tracks, honors a custom tempo, handles the empty-note list) and `save_midi` (writes a non-empty, re-loadable file; accepts both `Path` and `str`). Because `notes_to_midi` delegates through `note2note_event` and `note_event2midi` (`utils/midi.py:19-27`), this file is the **only** non-integration coverage of `note_event2midi` in `tokenizer/notes.py` — the multi-track writer with its channel-allocation logic (programs on channels 0–8/10–15, drums on 9).

### `test_cli.py` — the Typer CLI

Five tests. The whole point is the stdout/stderr contract: with `-o -` the JSONL/JSON output must be *pure* on stdout while every "Loading model…"/"Transcribing…"/"Saved …" banner goes to stderr. It stays hermetic via `patched_model` (`:61-64`), which monkeypatches `muscriptor.main.TranscriptionModel` with a `_FakeModel` (`:25-47`) whose `transcribe()` yields a scripted two-note stream and whose `_FakeInner.to(...)` absorbs the CLI's `model._model = model._model.to(torch.float32)` cast. `fake_audio` (`:50-58`) writes a tiny real WAV so the file-exists check passes. Driven with Typer's `CliRunner`.

- `test_jsonl_to_stdout_is_pure_jsonl` (`:67`) parses every stdout line as JSON and checks the exact dict shapes.
- `test_jsonl_stdout_has_no_chatter` / `test_progress_messages_go_to_stderr` (`:90`, `:106`) assert banners appear only on stderr.
- `test_jsonl_to_file_keeps_progress_on_stderr` (`:120`) checks that writing to a file leaves stdout empty and the file byte-exact.
- `test_instruments_passed_to_model` (`:154`) proves `--instruments violin,drums` reaches the model as `["violin","drums"]` (the fake records `last_kwargs`).

### `test_server.py` — the FastAPI SSE server

Ten tests using FastAPI's `TestClient`. The model is faked with `create_autospec(TranscriptionModel, instance=True)` (`make_model`, `:22-32`) — autospec matters because the server does `isinstance` checks and calls specific methods, and the spec keeps the mock's signatures honest while stubbing `transcribe` and `events_to_midi_bytes` return values. A `_parse_sse` helper (`:58-67`) splits the `text/event-stream` body on blank lines and parses each `data: <json>` payload.

- `test_event_to_dict_start_and_end` (`:70`): pins the wire shapes of `server.event_to_dict` (identical to `main._event_to_dict`).
- `test_transcribe_streams_sse_events` (`:87`): note events stream as SSE, followed by a trailing `{"type":"midi","data": <base64>}` event.
- `test_transcribe_forwards_progress` (`:117`): `ProgressEvent`s surface as their own `progress` SSE type **but** are kept out of the list passed to `events_to_midi_bytes` (checked via `call_args`).
- Audio decoding: `test_transcribe_passes_tensor_not_path` (`:164`) proves the server hands the model an in-memory `(tensor, sr)` tuple; `test_transcribe_accepts_non_wav_audio` (`:193`) encodes a FLAC in memory via `soundfile` and proves the non-WAV path decodes it (sample rate preserved, not resampled). This is the suite's only exercise of `utils/audio.py`'s `_read_wav_file`/`_read_non_wav_file` outside integration.
- Error handling: a missing `file` field → 422 (`:158`), an undecodable WAV or a mystery MP3 → 400 (`:183`, `:213`, using `raise_server_exceptions=False`).
- `test_transcribe_passes_instruments` (`:223`): form field `instruments` reaches the model as a list.

Untested server surface: the `/auralize`, `/health`, `/instruments`, and `/soundfonts` endpoints, and the entire cancellation/lock-preemption machinery (`_make_release_once`, the preempt-and-wait loop) have no tests.

### `test_download.py` — friendly auth errors

One parametrized test (`:10-11`, over `GatedRepoError` and `RepositoryNotFoundError`) for `utils/download.py`. It monkeypatches `hf_hub_download` to raise, then asserts `download_if_necessary("hf://…")` re-raises as a `ModelDownloadError` whose message contains the actionable hints: `hf auth login`, `HF_TOKEN`, and the repo's license URL. Only this one gated-repo branch is covered; the HTTP-URL caching branch, the `download_companion` helper, and the local-file branch are untested.

### `test_integration.py` — real checkpoint, end to end

Twenty-four tests, all gated on the session-scoped `transcription_model` fixture (skipped without weights). Module-scoped `silence`/`noise` fixtures (`:27-35`) build 5-second CPU waveforms; `song_clip` (`:38-46`) needs the real WAV and skips otherwise. Coverage here is exactly the surface the hermetic tests fake out:

- **Model state** (`:54-67`): loaded, in `eval()`, on CPU.
- **`_load_wav`** (`:74-97`): 1-D→[1,T], 2-D pass-through, stereo→mono, and 44.1k→16k resampling (`test_load_wav_resamples`, `:93`) — the only place `utils/resample.py` and `utils/audio.py`'s resample path actually run.
- **`_build_conditions`** (`:104-119`): the condition dict has `self_wav`/`instrument_group`/`dataset_name`, and the wav tensor is `[1,1,T]` with the right length.
- **`transcribe`** (`:141-165`, `:246-265`): the event stream obeys the start/end invariants (`_assert_start_end_invariants`, `:131-138`), fields are in range, and greedy/sampling/`cfg_coef=0` all run.
- **`transcribe_to_midi`** (`:171-184`): returns non-empty bytes that `mido` can parse.
- **CLI end to end** (`test_cli_json_output`, `:187`): monkeypatches `load_model` to return the session model and drives the real Typer app to write and re-parse a JSON event file.
- **Real song** (`:273-327`): non-empty events, note validity, MIDI output, and `test_transcribe_emits_progress_anchors` (`:286`) — which asserts the `ProgressEvent` sequence is exactly `[0, 1, …, total]` with a constant `total`.

## Coverage map

| Source area | Test file(s) | Depth |
| --- | --- | --- |
| `events.decode_model_tokens` (streaming decoder) | `test_events.py` (+ `encode_helpers.py`) | Deep — 12 hermetic cases incl. ties, windowing, retrigger |
| `transcription_model._generate_token_stream` (interleave/buffer/EOS) | `test_transcription_model.py` | Deep — ordering + streaming eagerness |
| `tokenizer/notes.py` vocab + note/event round-trips | `test_notes.py` (+ `encode_helpers.py`) | Deep |
| `tokenizer/notes.note_event2midi` (multi-track writer) | `test_midi.py` (transitive via `notes_to_midi`) | Good |
| `tokenizer/mt3.forbidden_token_ids` | `test_strict_instruments.py` | Deep |
| `models/lm.generate` masking — greedy/sampling/beam | `test_strict_instruments.py` (tiny model) | Good for masking; beam correctness itself unchecked |
| `transcription_model.transcribe` forbidden-token wiring | `test_strict_instruments.py` | Good (faked internals) |
| `modules/conditioners.py` | `test_conditioners.py` | Good (CPU, shapes) |
| `modules/transformer.py` + `modules/streaming.py` | `test_transformer.py` | Good (shape parity, not numeric) |
| `modules/mel_spectrogram.py` | `test_conditioners.py` | Indirect only |
| `utils/sampling.py` | `test_sampling.py` | Good |
| `utils/midi.py` | `test_midi.py` | Good |
| `utils/download.py` (gated-repo auth error) | `test_download.py` | Thin — one branch; HTTP/local/companion untested |
| `main.py` CLI (stdout/stderr split, formats) | `test_cli.py` | Good (fake model) |
| `server.py` `/transcribe` SSE + audio decode + errors | `test_server.py` | Good (mock model) |
| `utils/audio.py` `_read_wav_file`/`_read_non_wav_file` | `test_server.py` | Indirect |
| `transcription_model` load/transcribe end-to-end, `_load_wav`, `_build_conditions` | `test_integration.py` | Deep **but gated on weights** |
| `utils/resample.py` | `test_integration.py` only | None off-weights |
| `utils/auralization.py`; `soundfonts.py` | — | **None** |
| server `/auralize`, `/health`, `/instruments`, `/soundfonts`, lock/cancel | — | **None** |
| `transcription_model` load helpers (`_resolve_config`, `_remap_single_codebook_keys`, config.json parsing) | — | **None** (indirectly via integration load) |
| `models/lm.generate` CFG-doubling / prompt / beam KV-reorder | `test_integration.py` (CFG only, gated) | Thin |

Honest summary of gaps: the decode/tokenize core is very well covered without weights; the *load* path, resampling, and any real inference only run when the checkpoint is present. `auralization.py` (FluidSynth rendering) and `soundfonts.py` have no tests at all, and neither does the server's concurrency-control code — the trickiest untested surface in the repo, since its correctness depends on lock ordering and cancellation that a mock-model test never triggers.

## CI & tooling

- **CI**: the only GitHub Actions workflow is `.github/workflows/pypi.yml`, and it does **not run the test suite**. It triggers on GitHub release creation, builds the web frontend (`pnpm build` in `web/`), runs `uv build`, and publishes to PyPI via trusted publishing (`id-token: write`). There is no test/lint job on push or PR.
- **pre-commit** (`.pre-commit-config.yaml`): `ruff-check --fix`, `ruff-format`, and `uv-lock`. These are formatting/lint/lockfile hooks — again, no pytest hook — so tests are run manually (or not gated at all in the automated flow).

## Gotchas & invariants

- **Change the vocab, and `test_notes.py:117` fails first.** `test_event_vocab_num_tokens` hard-codes the term-by-term size `3 + 206 + 128 + 2 + 1 + 130 + 128`. Any edit to `build_event_vocab`'s ranges (adding an event type, changing the program count from 130, etc.) must update that assertion and the mirrored layout in `encode_helpers.py`, or a cascade of round-trip tests breaks. The decoder tests in `test_events.py` also assume `_MAX_SHIFT_STEPS = 1001` matches the model's shift range.
- **The `integration` marker is a decoy.** It is registered (`conftest.py:12-13`) but never applied, so `-m "not integration"` deselects nothing. Real gating is `pytest.skip()` inside the `transcription_model`, `song_clip`, and related fixtures. To run model-backed tests you must place `muscriptor_weights_01684fbb_350.safetensors` in the repo root (the exact filename in `conftest.py:6-8`), and the legacy `01684fbb` tag maps to the *large* config.
- **`SONG_PATH` is machine-specific.** `/home/simon/audio/filling_the_void.wav` (`conftest.py:9`) will never exist off the author's box, so the real-audio integration tests skip even when weights are present. There is no environment-variable override.
- **Nothing here is flaky in the network sense** — every external dependency (HuggingFace, soundfont download, model inference) is faked or skipped. The sampling tests that draw randomly (`test_sampling.py`, `test_strict_instruments.py`) either seed the RNG (`torch.manual_seed`) or assert only on support/concentration, so they are deterministic in practice; a change to `LMModel` init or the sampling math could still perturb the unseeded high-temperature control test (`test_generate_unmasked_uses_full_vocabulary`).
- **Beam search executes in exactly one test.** `test_beam_search_never_emits_forbidden_tokens` is the only path that runs `LMModel.generate` with `beam_size > 1`, and it only checks the *masking* invariant — the beam ranking, KV-cache reordering, and length normalization are never asserted for correctness.
- **The two `event_to_dict` functions must stay in lockstep.** `main._event_to_dict` and `server.event_to_dict` produce identical wire shapes; `test_cli.py` and `test_server.py` each pin their own copy, so a change to one without the other is caught on one side only.

## Cross-references

- `05-events-decoding.md` — the `decode_model_tokens` state machine that `test_events.py` drives, and the `encode_helpers.py` counterpart encoder.
- `04-tokenizer.md` — `build_event_vocab`, `MT3Tokenizer`, and `forbidden_token_ids`, verified by `test_notes.py` and `test_strict_instruments.py`.
- `01-pipeline.md` and `02-model-core.md` — `transcribe`, `_generate_token_stream`, and `LMModel.generate`, whose contracts `test_transcription_model.py` and the integration tests lock in.
- `03-conditioning-audio.md` — the conditioners, mel spectrogram, and audio/resample utilities behind `test_conditioners.py`, `test_transformer.py`, and the integration `_load_wav` tests.
- `06-cli-packaging.md` — the Typer CLI and packaging (`pyproject.toml`, CI) surfaced by `test_cli.py`.
- `07-server-web.md` — the FastAPI server and SSE protocol tested by `test_server.py`.
