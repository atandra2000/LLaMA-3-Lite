"""Phase 3 gates: the music bed mix and its loudness tooling.

T3.1 keeps the bed level, the duck and the two-pass loudnorm in
build_full_video.py. T3.4 adds tools/verify_video.py. The pure parsers —
loudnorm JSON, ebur128, SRT cue counting — are the steps worth pinning; the
rest of the gate shells out to ffmpeg and is covered by running it.
"""
import numpy as np
import pytest

from build_full_video import parse_loudnorm_json
from tools.verify_video import count_srt_cues, parse_ebur128, speech_over_gaps


def test_parse_loudnorm_json_extracts_trailing_block():
    stderr = (
        "[Parsed_loudnorm_0 @ 0x1] some warning\n"
        "{\n"
        '\t"input_i" : "-28.12",\n'
        '\t"input_tp" : "-6.01",\n'
        '\t"input_lra" : "4.20",\n'
        '\t"input_thresh" : "-38.20",\n'
        '\t"output_i" : "-16.00",\n'
        '\t"target_offset" : "0.10"\n'
        "}\n"
        "    \n"
        "[out#0/null @ 0x1] video:0kB audio:0kB\n"
    )
    got = parse_loudnorm_json(stderr)
    assert got["input_i"] == "-28.12"
    assert got["target_offset"] == "0.10"


def test_parse_loudnorm_json_raises_without_block():
    with pytest.raises(RuntimeError):
        parse_loudnorm_json("ffmpeg version n6.0\nno stats here\n")


# T3.4: the release gate's two pure parsers.
def test_parse_ebur128_takes_the_summary_not_the_frame_log():
    blob = (
        "[Parsed_ebur128_0 @ 0x1] t: 0.0249859    TARGET:-23 LUFS    "
        "M: -70.5 S: -70.5     I: -70.5 LUFS    LRA:  0.0 LU\n"
        "[Parsed_ebur128_0 @ 0x1] t: 1.0249859   TARGET:-23 LUFS    "
        "M: -16.2 S: -16.1     I: -16.4 LUFS    LRA:  8.1 LU\n"
        "\n"
        "  Integrated loudness:\n"
        "    I:         -16.0 LUFS\n"
        "    Threshold: -26.9 LUFS\n"
        "\n"
        "  True peak:\n"
        "    Peak:       -1.4 dBFS\n"
    )
    integrated, peak = parse_ebur128(blob)
    assert integrated == -16.0
    assert peak == -1.4


def test_parse_ebur128_raises_without_summary():
    with pytest.raises(RuntimeError):
        parse_ebur128("ffmpeg version n6.0\nno summary here\n")


def test_count_srt_cues_reads_arrow_lines_only(tmp_path):
    f = tmp_path / "x.srt"
    f.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nA large language model.\n\n"
        "2\n00:00:02,200 --> 00:00:04,000\nOne function.\n",
        encoding="utf-8")
    assert count_srt_cues(f) == 2


# The voice-in-mix gate: a build can drop the narration and still measure in
# spec on loudness, so this compares the speech spans against the gaps.
_ORDER = ["sec01"]
_SECTIONS = {"sec01": {"sentences": {
    "s01.1": {"start": 0.0, "end": 1.0},
    "s01.2": {"start": 2.0, "end": 3.0},
}}}
_CHAPS = [{"start": 0.0, "end": 4.0}]


def _tone_with(speech_gain: float) -> np.ndarray:
    """4 s at 8 kHz: tones only inside the sentence spans, off elsewhere."""
    x = np.zeros(4 * 8000, dtype=np.float32)
    for a, b in ((0.0, 1.0), (2.0, 3.0)):
        i = int(a * 8000)
        x[i:i + 8000] = speech_gain * np.sin(2 * np.pi * 440
                                             * np.arange(8000) / 8000)
    return x


def test_speech_over_gaps_sees_the_voice():
    delta = speech_over_gaps(_tone_with(0.5), _CHAPS, 0.0, _ORDER, _SECTIONS)
    assert delta > 20.0


def test_speech_over_gaps_reports_a_dropped_voice():
    # Voice gone, bed left playing through the gaps: 8 dB quieter in speech.
    x = np.full(4 * 8000, 0.4, dtype=np.float32) * np.sin(
        2 * np.pi * 440 * np.arange(32000) / 8000)
    for a, b in ((0.0, 1.0), (2.0, 3.0)):
        x[int(a * 8000):int(b * 8000)] = 0.0
    delta = speech_over_gaps(x, _CHAPS, 0.0, _ORDER, _SECTIONS)
    assert delta < 0.0
