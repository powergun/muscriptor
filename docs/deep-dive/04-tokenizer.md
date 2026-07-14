# The MT3-style tokenizer and the Note domain model

The model does not emit MIDI, notes, or times directly — it emits a stream of
integer token ids from a small, fixed vocabulary. This chapter documents the
two modules that define what those integers *mean* and the plain-data types the
rest of the pipeline reasons about: `muscriptor/tokenizer/notes.py`, which owns
the `Note`/`NoteEvent` domain model and the token vocabulary table, and
`muscriptor/tokenizer/mt3.py`, which owns instrument grouping, name resolution,
and the `MT3Tokenizer` that ties the vocabulary to a model. The token scheme is
adapted from MT3 / YourMT3+ (see the module docstrings), but everything below
describes what *this* code does, not the paper.

A useful orientation before the details: the `MT3Tokenizer` object is
deliberately thin. It has **no `encode` and no `decode` method**. At inference
it does exactly three things — build the vocabulary table (`_vocab`), expose a
few scalars (`eos_id`, `frame_rate`, `num_tokens`) and the instrument grouping
(`group_program_map`), and compute the set of token ids to forbid when the user
constrains instruments (`forbidden_token_ids`). The actual token → event decode
lives in `muscriptor/events.py` (`decode_model_tokens`), which reads
`tokenizer._vocab`; the reverse note → token *encode* path exists only as test
scaffolding in `tests/encode_helpers.py` and is not shipped in the package.

## The event vocabulary

`build_event_vocab(max_shift_steps)` (`muscriptor/tokenizer/notes.py:115`) is
the single source of truth for the token layout. It builds an ordered list of
`Event(type, value)` records; the list index *is* the token id, so `vocab[i]`
is the event the model means when it emits integer `i`. The layout is a fixed
concatenation of contiguous ranges (`muscriptor/tokenizer/notes.py:121`):

```python
ranges = (
    [EventRange(token, 0, 0) for token in SPECIAL_TOKENS]     # PAD, EOS, UNK
    + [EventRange("shift", 0, max_shift_steps - 1)]
    + [EventRange("pitch", 0, 127),
       EventRange("velocity", 0, 1),
       EventRange("tie", 0, 0),
       EventRange("program", 0, 129),
       EventRange("drum", 0, 127)]
)
```

Each `EventRange` expands to `max_value - min_value + 1` consecutive ids (the
`max_value` is inclusive, `muscriptor/tokenizer/notes.py:44`). `SPECIAL_TOKENS`
is the 3-tuple `("PAD", "EOS", "UNK")` and occupies the first three ids
(`muscriptor/tokenizer/notes.py:112`).

The shipped `MT3Tokenizer` is always built with `max_shift_steps=1001`
(`muscriptor/transcription_model.py:296`), which yields a **1393-token**
vocabulary. Verified id ranges for that configuration:

| Token ids | Event `type` | `value` range | Count | Meaning |
|-----------|--------------|---------------|-------|---------|
| 0 | `PAD` | 0 | 1 | Padding (never emitted as output) |
| 1 | `EOS` | 0 | 1 | End of a chunk's sequence — this id is `eos_id` |
| 2 | `UNK` | 0 | 1 | Unknown / reserved (unused by the decoder) |
| 3–1003 | `shift` | 0–1000 | 1001 | Time offset within the chunk, in 10 ms steps |
| 1004–1131 | `pitch` | 0–127 | 128 | MIDI note number of a melodic note on/off |
| 1132–1133 | `velocity` | 0–1 | 2 | Note-**off** (0) vs note-**on** (1) — *not* loudness |
| 1134 | `tie` | 0 | 1 | Terminator of the per-chunk *tie prologue* |
| 1135–1264 | `program` | 0–129 | 130 | Instrument (GM program) for the notes that follow |
| 1265–1392 | `drum` | 0–127 | 128 | A drum onset whose `value` is the GM drum pitch |

`test_event_vocab_num_tokens` (`tests/test_notes.py:117`) pins this arithmetic
for the `max_shift_steps=206` case: `3 + 206 + 128 + 2 + 1 + 130 + 128`. In
general the vocabulary size is `max_shift_steps + 392`.

### Vocabulary size vs. model `card`

The vocabulary size (`MT3Tokenizer.num_tokens`, set at
`muscriptor/tokenizer/mt3.py:228`) is **1393** for every model size, because the
tokenizer is constructed identically regardless of which checkpoint is loaded.
The model's output width is a separate number, `card`, taken from the per-size
config (`muscriptor/transcription_model.py:103`):

- `small`: `card = 1393` — exactly equal to the vocabulary.
- `medium` / `large`: `card = 1395` — two more than the vocabulary.

The output head is `nn.Linear(dim, card)` (`muscriptor/models/lm.py:133`), so
the medium/large models can in principle emit ids `1393` and `1394`, which have
**no entry in the 1393-element vocabulary**. The input embedding is sized
`card + 1` (`muscriptor/models/lm.py:117`) because the generation loop primes
itself with an `initial_token_id == card` (`muscriptor/models/lm.py:141`); that
extra row is the BOS slot, not a decodable token. In practice the model is
trained never to produce `1393`/`1394`, so `decode_model_tokens` never indexes
them; see Gotchas for why an out-of-range id would be fatal if one ever
appeared.

### Time resolution: `frame_rate` and `shift`

`frame_rate` defaults to 100 and is stored on the tokenizer
(`muscriptor/tokenizer/mt3.py:226`); the decoder reads it back
(`muscriptor/transcription_model.py:404`). 100 frames per second means each
`shift` step is **10 ms**. A `shift` token is *absolute within the chunk*, not
incremental: the decoder computes `tick_state = start_tick + event.value` where
`start_tick = round(seek_time * frame_rate)`
(`muscriptor/events.py:127`, `muscriptor/events.py:170`), so a `shift` of `N`
places the following events at `seek_time + N/100` seconds. `shift` value `0` is
a no-op (the decoder only moves time when `event.value > 0`,
`muscriptor/events.py:169`). With `max_shift_steps=1001` the representable range
is 0–1000 steps = 0–10.00 s, comfortably more than the 5.0 s chunk length
(`muscriptor/transcription_model.py:89`), so a chunk never runs out of time
resolution.

### What "velocity" is (and what is not stored)

The `velocity` range holds only **two** values, 0 and 1, and they encode
note-off vs note-on, not loudness. The README states this directly: "this
tokenizer does not preserve velocity (loudness) — only onset/offset timing,
pitch, and instrument are recovered" (`README.md:210`). During decode,
`velocity_state > 0` starts a note and `velocity == 0` closes the currently open
`(program, pitch)` note (`muscriptor/events.py:192`, `muscriptor/events.py:190`).
When the pipeline later writes MIDI it stamps every note with a constant
velocity of 100 (`muscriptor/utils/midi.py:11`). So there is no dynamics
information anywhere in the token stream — the vocabulary simply has no slot for
it.

## `MT3Tokenizer`

`MT3Tokenizer.__init__` (`muscriptor/tokenizer/mt3.py:217`) takes three
arguments:

- `instrument_vocabulary` (default `"FULL"`) — selects the program-grouping
  scheme; production code always passes `"MT3_FULL_PLUS"`
  (`muscriptor/transcription_model.py:297`).
- `max_shift_steps` (default 1001) — the size of the `shift` range, passed
  straight to `build_event_vocab`.
- `frame_rate` (default 100) — stored for the decoder's tick↔seconds math.

It computes and stores:

- `group_program_map` — `get_group_program_map(instrument_vocabulary,
  misc_programs="SINGLETON_GROUPS", is_mt3=True)` (`:223`). Note the fixed
  `misc_programs`/`is_mt3` — callers cannot change them.
- `frame_rate` (`:226`), `_vocab` (`:227`, the `build_event_vocab` list),
  `num_tokens` (`:228`, `len(self._vocab)` = 1393),
  and `eos_id` (`:229`, `SPECIAL_TOKENS.index("EOS")` = **1**).

Every one of these is consumed by `TranscriptionModel`:
`tokenizer._vocab` and `tokenizer.frame_rate` are handed to
`decode_model_tokens` (`muscriptor/transcription_model.py:402`);
`tokenizer.eos_id` becomes the generator's `early_stop_on_token`
(`muscriptor/transcription_model.py:442`, `:469`); `group_program_map` drives
the program↔name mapping (below); and `forbidden_token_ids` produces the decode
mask. There is no `encode`/`decode` method on the class — those verbs live
elsewhere as described in the intro.

### `forbidden_token_ids` — the hard instrument constraint

`forbidden_token_ids(instruments)` (`muscriptor/tokenizer/mt3.py:233`) is the
enforcement half of the `--instruments` feature. Given a list of *exact*
MT3_FULL_PLUS group names, it returns the token ids that must be masked to
`-inf` so the model can never emit any instrument outside the allowed set. The
logic:

1. Validate names against `MT3_FULL_PLUS_GROUP_NAMES`, raising `ValueError`
   listing the unknown ones (`:246`).
2. `allow_drums = "drums" in names` (`:252`).
3. Build `allowed_programs`: for each non-drum name, look up its group id and
   take `group_program_map[gid][0]` — the group's **representative (first)
   program** (`:256`). This mirrors the decode-side convention exactly (below),
   so the one allowed program per group is precisely the one the model emits for
   that group.
4. Walk the whole vocabulary and forbid every `program` token whose value is not
   in `allowed_programs`, plus every `drum` token unless `allow_drums`
   (`:263`–`:268`).

Everything else — `shift`, `pitch`, `velocity`, `tie`, and the three special
tokens — is **never** forbidden. `tests/test_strict_instruments.py` pins each of
these rules: timing/special tokens are untouched
(`test_forbidden_ids_never_touch_timing_or_special_tokens`), only the listed
groups' representative programs survive
(`test_forbidden_ids_keep_only_listed_programs`), drums are masked unless listed
(`test_forbidden_ids_mask_drums_unless_listed`), and `["drums"]` alone forbids
every program token (`test_forbidden_ids_drums_only_masks_every_program`). The
returned ids are converted to a tensor and passed through generation to
`LMModel`, which sets `logits[:, forbidden_tokens] = -inf`
(`muscriptor/transcription_model.py:342`, `muscriptor/models/lm.py:224`). The
constraint is automatic: supplying `instruments` activates it, omitting it masks
nothing (`test_instruments_given_activates_forbidden_tokens` /
`test_no_instruments_means_no_forbidden_tokens`).

## Instrument grouping

`get_group_program_map` (`muscriptor/tokenizer/mt3.py:19`) maps a group id to
the list of General MIDI program numbers that belong to it, for one of several
named schemes. The production scheme is `"MT3_FULL_PLUS"`
(`:44`–`:82`): 36 explicitly defined groups (ids 0–35) that fold the 0–95 GM
programs plus 100/101 into musically meaningful buckets — e.g. group 0 is
`[0, 1, 3, 6, 7]` (acoustic-piano-like programs), group 9 is `[40]` (violin),
group 15 is `[48, 49, 44, 45]` (string ensemble). Within each group the **first
listed program is the representative**, which is the whole grouping's load-
bearing convention (see below and `_build_instrument_for_program`).

After the explicit groups, the function reconciles the remaining programs
(`:97`–`:109`). With the tokenizer's fixed `is_mt3=True`, `not_assigned` is the
set of `range(128)` programs not placed by any explicit group; `include_drums`
is false, so `DRUM_PROGRAM` is *not* added. `misc_programs="SINGLETON_GROUPS"`
then appends one singleton group per leftover program (`:105`). For
MT3_FULL_PLUS that leftover set is programs `96–99` and `102–127` (30 of them),
producing group ids `36`–`65`. So the shipped `group_program_map` has **66
groups total**: 36 curated + 30 singletons.

### Human-readable names: `MT3_FULL_PLUS_GROUP_NAMES`

`MT3_FULL_PLUS_GROUP_NAMES` (`muscriptor/tokenizer/mt3.py:118`) is an ordered
dict of **35** user-facing names → group id, e.g. `"acoustic_piano": 0`,
`"violin": 9`, `"flutes": 31`, `"synth_pad": 33`, and `"drums": 36`. It names
the 34 curated melodic groups 0–33 and the drum group, but deliberately does
*not* name groups 34/35 (programs 100/101) or any singleton group. The comment
at `:113` records the contract: the numeric ids index the model's learned
program groups and must not change; only the strings may. Anything the model
decodes into an unnamed group surfaces as `program_<n>` (see next section). This
dict is what the CLI's `list-instruments` command prints
(`muscriptor/main.py:327`) and what the `--instruments` option resolves against.

### Programs → readable names: the representative convention

`_build_instrument_for_program` (`muscriptor/transcription_model.py:224`) builds
the program → name lookup the decoder attaches to every note. It inverts the
grouping through the representative-program rule
(`muscriptor/transcription_model.py:233`):

```python
for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
    if gid in group_map and group_map[gid]:
        program_to_name[group_map[gid][0]] = name
```

Because the model only ever emits a group's *first* program, mapping that one
representative back to the group name recovers the instrument. The returned
`lookup(program)` short-circuits `DRUM_PROGRAM` to `"drums"` and falls back to
`f"program_{program}"` for anything unnamed
(`muscriptor/transcription_model.py:238`). `_program_for_instrument`
(`muscriptor/transcription_model.py:586`) is the inverse used when rebuilding
MIDI, memoizing the same name → representative-program map and special-casing
`program_<n>` strings back to their integer.

### Name resolution: strict vs. loose

Two functions turn user input into the group names the constraint machinery
expects:

- `instrument_group_from_names(names)` (`muscriptor/tokenizer/mt3.py:157`) is
  **strict**: every name must be a verbatim key of `MT3_FULL_PLUS_GROUP_NAMES`,
  else `ValueError`. It returns a space-joined string of the group *ids* (e.g.
  `"9 36"`), which is the advisory `instrument_group` conditioning text fed to
  the model's class conditioner (`muscriptor/transcription_model.py:337`). This
  is the soft hint; `forbidden_token_ids` is the hard counterpart.
- `resolve_instrument_names(tokens)` (`muscriptor/tokenizer/mt3.py:174`) is the
  **loose, CLI-facing** resolver. Each token is lower-cased and stripped
  (`:184`); an exact name passes through; otherwise it is treated as a substring
  and matched against all group names. A single substring hit resolves
  (`"timp"` → `"timpani"`); multiple hits raise a `ValueError` listing the
  candidates (`:191`); zero hits raise a `ValueError` with up to three
  `difflib`-ranked spelling suggestions, comparing the token against each full
  name *and* its underscore-separated words so a typo like `"pinao"` still
  surfaces `"acoustic_piano"` (`:199`–`:212`). The CLI calls this before
  transcription and prints the resolved names (`muscriptor/main.py:175`); the
  library `transcribe` API expects already-exact names and calls only the strict
  path (`muscriptor/transcription_model.py:335`).

## The `Note` domain model

`notes.py` defines the plain dataclasses the rest of the system passes around.

`Note` (`muscriptor/tokenizer/notes.py:16`) is the durational unit — one
sounding note:

- `is_drum: bool`
- `program: int` — GM program 0–127, or `128` (`DRUM_PROGRAM`) for drums
- `onset: float`, `offset: float` — seconds (`offset == onset` nominally for
  drums, though decode gives drums a tiny non-zero duration; see below)
- `pitch: int` — MIDI note number 0–127

`DRUM_PROGRAM = 128` (`muscriptor/tokenizer/notes.py:12`) is the sentinel that
distinguishes a drum `Note` from a program-0 piano note, and
`MINIMUM_NOTE_DURATION_SEC = 0.01` (`:13`) is the floor duration used
throughout.

Three transient event types support conversion but are not part of the public
output:

- `NoteEvent` (`:25`) — a single on/off *edge*: `time`, `velocity` (1 = onset,
  0 = offset; drums have only an onset), `program`, `pitch`, `is_drum`.
- `TieNoteEvent` (`:34`) — a `(program, pitch)` pair carried across a chunk
  boundary, representing a note held open from the previous chunk.
- `Event` (`:47`) and `EventRange` (`:40`) — the vocabulary primitives described
  above.

### Sorting helpers

`sort_notes` (`:53`) orders `Note`s in place by
`(onset, is_drum, program, pitch, offset)`; `sort_note_events` (`:58`) orders
`NoteEvent`s by `(time, is_drum, program, velocity, pitch)`; and
`sort_tie_note_events` (`:65`) orders `TieNoteEvent`s by `(program, pitch)`.
These deterministic keys make the downstream trimming and MIDI-writing passes
reproducible. All three no-op on an empty list.

### Note hygiene: `validate_notes` and `trim_overlapping_notes`

`validate_notes(notes, minimum_offset=0.01, fix=True)`
(`muscriptor/tokenizer/notes.py:70`) is a cleanup pass over timing only — it
never inspects `pitch` or `program` ranges, so it does **not** clamp or drop
out-of-range pitches. The body is an `if`/`elif` chain, so **at most one** rule
fires per note (`:76`–`:85`):

1. `onset is None` → drop the note from the list.
2. else `offset is None` → set `offset = onset + minimum_offset`.
3. else `onset > offset` (inverted) → `offset = max(offset, onset + minimum_offset)`.
4. else non-drum and `offset - onset < 0.01` → `offset = onset + minimum_offset`.

Rule 4 is guarded on `is_drum is False`, so short *drum* notes are left alone.
All fixes are gated on `fix=True` (the only mode used in the package); with
`fix=False` the function is effectively a no-op that returns the list unchanged.
`tests/test_notes.py` exercises the short-duration fix, the inverted fix, and
the `None`-onset drop (`:60`–`:75`).

`trim_overlapping_notes(notes, sort=True)`
(`muscriptor/tokenizer/notes.py:89`) enforces monophony *per channel*, where a
channel is a distinct `(program, pitch, is_drum)` triple (`:93`). Within each
channel it sorts by onset and, whenever a note's offset extends past the next
note's onset, truncates the earlier offset down to that onset
(`:101`–`:103`); notes left with `onset >= offset` are then dropped (`:104`).
Different pitches or different programs never interact, so genuine chords are
preserved — `test_trim_overlapping_notes_different_pitch` confirms two
overlapping notes of different pitch both survive. The pass **mutates** the
input `Note` objects' `offset` fields. `sort=True` re-sorts the survivors with
`sort_notes`.

Both functions run back-to-back in the inference MIDI path:
`events_to_midi_bytes` calls `validate_notes(notes, fix=True)` then
`trim_overlapping_notes(notes, sort=True)` before serializing
(`muscriptor/transcription_model.py:579`), matching the legacy decoder's cleanup
so the MIDI bytes stay stable across refactors.

### `NoteEvent` conversions

- `note2note_event(notes)` (`muscriptor/tokenizer/notes.py:243`) explodes each
  `Note` into its edge events: an onset `NoteEvent` (velocity 1) always, plus an
  offset `NoteEvent` (velocity 0) for non-drums (`:251`). Drums get only an
  onset. It is used by `notes_to_midi` (`muscriptor/utils/midi.py:19`) and
  re-exported from the package (`muscriptor/tokenizer/__init__.py`). Note the
  legacy line `if note.program == 1024: note.is_drum = True` (`:246`): the drum
  sentinel used elsewhere is `128`, not `1024`, so this branch never fires for
  notes produced by this codebase — it is dead legacy compatibility.
- `note_event2midi(note_events, ...)` (`muscriptor/tokenizer/notes.py:259`)
  serializes edges into a **type-1 (multi-track)** mido `MidiFile`, one named
  track per program so DAWs that split imports by track (Ableton) keep
  instruments apart. It synthesizes drum note-offs 0.01 s after each drum onset
  (`:283`–`:295`), assigns programs to channels 0–8 then 10–15 in first-
  appearance order (drums always on channel 9), and names each track from the
  optional `program_names` map, falling back to `"program <n>"` / `"drums"`.
  This is the concrete MIDI writer behind `notes_to_midi` /
  `TranscriptionModel.transcribe_to_midi`.
- `note_event2note(note_events, ...)` (`muscriptor/tokenizer/notes.py:139`)
  reassembles `Note`s from a flat `NoteEvent` list, pairing onsets with offsets
  through an `active_note_events` map keyed by `(program, pitch)` and applying
  `TieNoteEvent`s for cross-segment holds (`:152`, `:189`). It also shortens
  over-long notes, optionally runs `validate_notes`/`trim_overlapping_notes`,
  and returns a `Counter` of decode errors. **This function is test-only** — the
  shipped inference path reassembles notes incrementally in
  `decode_model_tokens` (`muscriptor/events.py`) and never calls it; its only
  callers are in `tests/test_notes.py`.

## Gotchas & invariants

- **`num_tokens` (1393) is not `card` (1395 for medium/large).** The vocabulary
  table is the same size for every checkpoint; only `small` has `card` equal to
  it. Medium/large models have two output logits (ids 1393, 1394) with no
  vocabulary entry. `decode_model_tokens` does a bare `vocab[item]`
  (`muscriptor/events.py:137`), so if a model ever emitted one of those ids it
  would raise `IndexError` rather than degrade gracefully. Training keeps this
  from happening; there is no defensive clamp.
- **`velocity` is note-on/off, not loudness.** Only values 0 and 1 exist, and
  all output MIDI is written at a constant velocity of 100. No dynamics survive
  transcription.
- **`shift` is chunk-relative and absolute within the chunk**, not incremental.
  Decoding adds it to `start_tick`, and a `shift` of 0 is ignored. Each step is
  `1/frame_rate` s = 10 ms at the default 100 Hz.
- **A group's first program is its identity.** Encoding (implied by training),
  `forbidden_token_ids`, and `_build_instrument_for_program` all rely on
  `group_program_map[gid][0]`. If the order of programs within a group in
  `get_group_program_map` ever changed, the allowed-program mask and the decoded
  instrument name would both shift.
- **"drums" (group id 36) collides with the first singleton group.** With the
  tokenizer's fixed `is_mt3=True` / `include_drums=False`, the first auto-
  generated singleton group is id 36 = `[96]`, but `MT3_FULL_PLUS_GROUP_NAMES`
  also assigns `"drums"` the id 36. Consequently the reverse map assigns
  **program 96 the name `"drums"`** (verified: `program_to_name[96] == "drums"`).
  Real drums are unaffected because they are decoded via the separate `drum`
  token type and the `DRUM_PROGRAM` short-circuit in `lookup`, never via a
  program token. The only observable effect is a latent mislabel: if the model
  ever decoded a `program` token with value 96 (GM "FX 1 (rain)"), that note
  would be tagged `"drums"` and then, because `events_to_midi_bytes` keys drums
  off the string `ev.instrument == "drums"`
  (`muscriptor/transcription_model.py:557`), be written to the MIDI as a drum
  hit. In normal use the model does not emit program 96, so this stays dormant.
- **`program` range extends past GM.** The `program` event covers values 0–129
  (130 slots), two beyond the 0–127 GM range. Value 128 equals `DRUM_PROGRAM`
  and would decode to `"drums"` via the `lookup` short-circuit; value 129 is
  unused and would surface as `"program_129"`. Neither is expected from a
  trained model.
- **Groups 34/35 and every singleton are nameless.** Programs 100 and 101, and
  singleton programs 97–99/102–127, have no entry in
  `MT3_FULL_PLUS_GROUP_NAMES`; the model can still decode them, and they surface
  as `program_<n>` in the event stream and as `"program <n>"` MIDI track names.
- **`validate_notes` only touches timing.** It does not validate `pitch` or
  `program`; out-of-range values pass through untouched. Its `fix=False` mode is
  never used in the package (and would break on a `None` onset, since the
  inverted-note comparison would then compare `None`).
- **`MT3Tokenizer` has no encode/decode methods.** Decode is in
  `muscriptor/events.py`; the note↔token encode path is test-only scaffolding in
  `tests/encode_helpers.py`. `note_event2note` in `notes.py` is likewise
  test-only.

## Cross-references

- `05-events-decoding.md` — `decode_model_tokens`, the tie prologue, and the
  chunk state machine that consumes `tokenizer._vocab` and `frame_rate`.
- `02-model-core.md` — `LMModel`, `card` vs. `num_tokens`, `initial_token_id`,
  and how `forbidden_tokens` masks logits.
- `03-conditioning-audio.md` — the `instrument_group` class conditioner fed by
  `instrument_group_from_names`.
- `06-cli-packaging.md` — the `--instruments` / `list-instruments` CLI surface
  and `resolve_instrument_names`.
- `01-pipeline.md` — where `validate_notes` / `trim_overlapping_notes` and
  `notes_to_midi` sit in the end-to-end transcription flow.
- `08-tests.md` — `tests/test_notes.py`, `tests/test_strict_instruments.py`, and
  the `tests/encode_helpers.py` encode/decode scaffolding.
