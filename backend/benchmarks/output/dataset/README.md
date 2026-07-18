# STT Benchmark Dataset

## Running the benchmark

```bash
cd backend
PYTHONPATH=. .venv/Scripts/python.exe benchmarks/stt_benchmark.py --model small
```

Output:
- `benchmarks/output/benchmark_results.csv` — per-utterance metrics
- `benchmarks/output/benchmark_report.md` — human-readable report

## Adding real recordings

The benchmark runs in two modes:

### Mode 1: Synthetic (default)

If no WAV files exist in `benchmarks/output/dataset/`, the benchmark
auto-generates sine-wave audio using NumPy.  This is useful for
**latency testing** (VAD, pipeline timing) but **not for accuracy**
(sine waves produce no meaningful transcription).

### Mode 2: Real recordings (recommended for accuracy)

To measure WER/CER, place real speech recordings in this folder.

**Requirements:**
- Format: WAV (16-bit PCM, mono)
- Sample rate: 16000 Hz
- Filename: `{utterance_id}.wav` (matches the ID in `stt_benchmark.py`)

**Current dataset:**

| ID | Category | Expected text |
|----|----------|---------------|
| s01 | short_command | Open Chrome |
| s02 | short_command | Close Chrome |
| s03 | short_command | Mute |
| s04 | short_command | Stop |
| s05 | short_command | Volume up |
| s06 | short_command | Volume down |
| s07 | short_command | Analyze screen |
| s08 | short_command | What time is it |
| s09 | short_command | Take a screenshot |
| s10 | short_command | Lock the screen |
| l01 | long_command | Analyze my current screen and summarize what I'm working on |
| l02 | long_command | Can you explain the code currently open in Visual Studio Code |
| l03 | long_command | Search the web for the latest AI news |
| l04 | long_command | Open the file manager and navigate to the documents folder |
| l05 | long_command | Create a new folder on the desktop called project files |
| v01 | quiet_speech | Open Chrome |
| v02 | fast_speech | Close Chrome |
| v03 | slow_speech | Mute |
| v04 | accent | Volume up |
| v05 | long_pause | Analyze screen |

## Recording tips

1. Use a consistent microphone and distance for all recordings
2. Record at 16kHz mono WAV (ffmpeg: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`)
3. Name files exactly `{utterance_id}.wav` (e.g. `s01.wav`)
4. Place all WAV files in this `dataset/` directory
5. Run the benchmark — it will skip synthesis for existing files

## Variation categories

When recording variations, keep the expected text the same but vary:
- **quiet_speech**: Whisper at ~10cm distance
- **fast_speech**: Normal speed but compressed phrasing
- **slow_speech**: Elongated, deliberate speech
- **accent**: Natural accent variation
- **long_pause**: Insert 2-3 second pauses mid-utterance
