"""Phase 3 gates: the music bed mix and its loudness tooling.

T3.1 keeps the bed level, the duck and the two-pass loudnorm in
build_full_video.py. T3.4 adds tools/verify_video.py. The pure parsers —
loudnorm JSON, ebur128, SRT cue counting — are the steps worth pinning; the
rest of the gate shells out to ffmpeg and is covered by running it.
"""
import pytest

from build_full_video import parse_loudnorm_json
from tools.verify_video import count_srt_cues, parse_ebur128


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
