"""Phase 3 gates: the music bed mix and its loudness tooling.

T3.1 keeps the bed level, the duck and the two-pass loudnorm in
build_full_video.py. The JSON parse is the one pure step worth pinning.
"""
import pytest

from build_full_video import parse_loudnorm_json


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
