# LazyPredictGUI

A desktop app for turning a labeled dataset into a ranked comparison of
~30 scikit-learn classifiers (via [lazypredict](https://github.com/shankarpandala/lazypredict))
and a confusion matrix for whichever model you pick - no notebook, no
boilerplate.

It's a **generic tool** - it doesn't know or care where your data came from.
It ships with one built-in importer for wolfSentry mic-classification
recordings (background noise / dog barking / people speaking), but the core
pipeline (split -> leaderboard -> confusion matrix) works on any CSV with
feature columns and a label column.

## Install & run

```sh
./run.sh
```

First run creates a `.venv` here and installs `requirements.txt` into it
(PySide6, pandas, numpy, scikit-learn, lazypredict, matplotlib, librosa,
sounddevice). Later runs reuse it. The window opens with three tabs:
**Import & Label**, **Split & Leaderboard**, **Confusion Matrix**.

## Quick start with the bundled sample data

`test_data/` has three synthetic CSVs (`quiet.csv`, `dog.csv`, `speech.csv`)
and a script that regenerates them (`generate_synthetic_data.py`) - not real
audio, just distinct-enough fake feature distributions to try the whole
pipeline end to end before you have real recordings:

1. **Import & Label** tab -> "Load CSV..." -> pick `test_data/quiet.csv`.
   It already has a `label` column, so it's detected automatically.
2. Repeat for `dog.csv` and `speech.csv`, or just concatenate all three into
   one CSV yourself first (`generate_synthetic_data.py` doesn't do this for
   you - the "Load CSV..." path expects one already-combined file, see
   below).
3. **Split & Leaderboard** tab -> leave the defaults -> "Run LazyPredict".
4. **Confusion Matrix** tab -> pick a model from the dropdown -> "Evaluate
   on Test Set".

## The two ways to get data in

### A) Load a CSV directly (generic path)

Any CSV with numeric feature columns and one label column. Use this if your
data already came from somewhere else (a spreadsheet, another tool, an
already-combined export).

- Click **"Load CSV..."** in the Import & Label tab.
- The label column is auto-detected: a column literally named `label` if
  one exists, otherwise the last column.
- If it picked the wrong one, pick the right column from the **"Label
  column"** dropdown and click **"Use this column"**.

### B) Import wolfSentry mic sessions and label them here

If you're building the background/dog/speech mic dataset, each wolfSentry
capture session is a single `.bin` file (~470KB, ~30 seconds of audio) that
you pull off the device with wolfSentry's own
`host/pull_mic_dataset.py [serial-port]` script (run via `host/.venv/bin/
python`, not bare system python3 - it needs pyserial/xmodem) - that just
moves the bytes, it doesn't decode or label anything. Bring the resulting
file here:

1. Decide the granularity **before** importing (see below), then
   **"Import session..."** -> pick the `.bin` file. It's added as a new row
   in the session table (columns: Session, Label, Examples) and
   auto-selected.
2. Select a row, click **"Play"** to listen to the full ~30-second clip
   (long enough to confidently tell what it is - that's the whole reason
   sessions are 30s, not shorter). **"Stop"** cuts playback off early.
3. Pick a label in the combo box next to "Label selected session" and click
   **"Assign"** - it now shows up immediately in that row's **Label**
   column. The combo starts with `background`/`dog`/`speech`/`discard` as
   suggestions, but it's editable - type any class name you want (`2` or
   `20` classes both work fine, nothing here is hardcoded to 3).
4. Imported a file you don't want? Select its row and click **"Remove"** -
   drops it from the table and from what "Combine & Continue" will use.
5. Repeat for every session you've collected. Import several sessions of
   the same class (e.g. `dog_session1.bin`, `dog_session2.bin`,
   `dog_session3.bin`) and label them all `dog` - they'll be pooled
   together as one `dog` group, not split individually.
6. The running count under the table shows how many **examples** (not
   files) you've labeled per class so far - see the Examples column/
   granularity note below for why those can differ.
7. When you've got enough (see "How much data do I need?" below), click
   **"Combine & Continue"** - shows a summary of what got pooled (counts
   per class) and switches you straight to the Split & Leaderboard tab.

**Example granularity - the "Split into 1-second sub-clips" checkbox**
(next to "Import session...", applies to the *next* import only, not
retroactive - check it before importing a given file, or remove and
re-import to change an already-imported one's mode):

- **Off (default)**: 1 session = 1 example. Uses the on-device Goertzel
  summary (12 bands + RMS/ZCR/peak, computed on the device itself while
  capturing) plus MFCC computed here over the full ~30s. Needs many
  separate session recordings per class before a split means anything -
  see the volume table below.
- **On**: 1 session = up to 30 examples, one per already-captured
  1-second audio chunk. Gets real, usable data out of just a couple of
  recordings (2 sessions -> 60 examples), but **no Goertzel columns** for
  these rows - the device only computes Goertzel over the whole 30s
  session, not per second, so these examples only carry MFCC + PC-computed
  RMS/ZCR/peak. The Examples column shows the actual count either way.

Whole-session and sub-clip-mode sessions can be mixed in the same combined
dataset - rows missing a feature (e.g. no Goertzel columns from a sub-clip
row) just come out blank/imputed downstream, nothing breaks.

You don't have to pick Goertzel or MFCC up front either - the next tab lets
you try either, or both, and see which one LazyPredict actually ranks
higher for your data.

### How much data do I need?

Depends on which granularity you're using (see above):

**Whole-session mode** (1 example = 1 recording) - rough starting points:

| tier | background | dog | speech | total sessions |
|---|---|---|---|---|
| minimum viable | 40 | 20 | 20 | 80 |
| good | 100 | 60 | 60 | 220 |
| solid | 200 | 120 | 120 | 440 |

**Sub-clip mode** (1 example = 1 second) - the same 30x multiplier applies
in reverse: 2-3 recordings per class (60-90 examples) is already enough for
a real, non-degenerate 70/15/15 split and leaderboard, though more
recordings (not just longer ones) still helps with variety.

**The hard floor, either mode**: fewer than ~3 total examples and the
split can't even produce three non-empty buckets - the app will warn
instead of crashing, but there's nothing meaningful to rank with that
little data (confirmed on real hardware: 2 whole-session examples -> only
13/~30 models could even fit, and all scored 0% accuracy, since a single
training example can't generalize to anything). Background needs more
examples than the other classes either way - it's the "open world" of
everything else your device isn't dog or speech (room tone, HVAC, traffic,
wind, other animals, silence), so it needs more variety of conditions to
avoid the model just memorizing one room. More/varied recordings across
different times/locations beats fewer/longer ones, for every class.

## Split & Leaderboard

- **Feature set** checkboxes: pick Goertzel, MFCC, or both. This is
  per-run, so it's cheap to try both ways and compare leaderboards.
- **Train/Val/Test %** spinboxes, default 70/15/15. Must sum to 100 - the
  Run button is disabled otherwise.
- **Run LazyPredict** fits ~30 scikit-learn models against your train/val
  split and shows a sortable leaderboard (Accuracy, Balanced Accuracy, ROC
  AUC, F1, Time Taken). This can take anywhere from a few seconds to a
  couple minutes depending on dataset size - the app stays responsive while
  it runs.

Why validation, not test, feeds the leaderboard: the held-out test set is
never touched until you've actually picked a model, so the leaderboard
can't accidentally "cheat" by tuning to the same data you'll use for the
real final check.

## Help menu

**Help -> About LazyPredictGUI** has license/usage terms and a link if you
want to support the project.

## Confusion Matrix

Pick any model from the leaderboard, click **"Evaluate on Test Set"**. This
retrains that one model on train+validation combined and checks it against
the test set it's never seen - showing accuracy, weighted F1, and the
confusion matrix itself (rows = actual class, columns = predicted).

## Files, if you're poking around the code

```
lazypredictgui/
  gui.py                     Three tabs + MainWindow
  mic_session_reader.py      Decodes wolfSentry's session .bin format
  mic_features.py            MFCC extraction (the reference spec for a future on-chip port)
  dataset.py                 Pools labeled sessions (whole-session or 1s sub-clip granularity) into one combined table
  data.py                    CSV loading, label-column detection, train/val/test split
  workers.py                 Background threads for LazyPredict + model retraining
  confusion_matrix_widget.py Matplotlib confusion matrix widget
test_data/
  generate_synthetic_data.py Regenerates the sample CSVs
```
