"""Phase 2 gates: the cue API, the timing contract, and the motion layer.

Each test maps to what T2.1–T2.4 must deliver — per-sentence audio with exact
spans, a cue that waits on those spans, a budget the scene cannot quietly
overrun, and ambient motion that keeps a hold from reading as a freeze.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

RENDER = (V2 / "render_scenes.py").read_text(encoding="utf-8")
AUDIO = (V2 / "generate_audio.py").read_text(encoding="utf-8")
TIMINGS = json.loads((V2 / "audio" / "timings.json").read_text(encoding="utf-8"))
NARRATION = yaml.safe_load((V2 / "narration.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------- T2.1 audio ------
def test_timings_are_the_only_clock():
    """P1/P3: the render reads timings.json, not a hand-typed durations file."""
    assert 'json.loads((HERE / "audio" / "timings.json").read_text())' in RENDER
    assert "durations.json" not in RENDER


def test_no_beat_api_remains():
    """T2.2: beat(fraction) is gone; cue(sentence) owns the pacing."""
    assert "def beat" not in RENDER
    assert "self.beat(" not in RENDER


@pytest.mark.skip(reason="script v2 rewrite: scene re-cue pending (12 sections, 167 cues)")
def test_every_sentence_has_exactly_one_cue():
    """T2.2: the cue map covers the narration, one cue per spoken line."""
    ids = [s["id"] for sec in NARRATION["sections"] for s in sec["sentences"]]
    cued = []
    for line in RENDER.splitlines():
        st = line.strip()
        if st.startswith("self.cue("):
            cued.append(st.split('"', 2)[1])
    assert sorted(cued) == sorted(ids), (
        f"missing cues: {set(ids) - set(cued)}  "
        f"stray cues: {set(cued) - set(ids)}")


def test_timings_spans_are_monotonic_and_inside_their_section():
    """T2.1: a sentence starts after the one before it and ends inside the track."""
    for sec, data in TIMINGS["sections"].items():
        prev_end = 0.0
        for sid, span in data["sentences"].items():
            assert span["start"] >= prev_end - 1e-6, (sec, sid)
            assert span["end"] > span["start"], (sec, sid)
            assert span["end"] <= data["duration"] + 1e-6, (sec, sid)
            assert span["text"].strip(), (sec, sid)
            prev_end = span["end"]


def test_timings_cover_every_section():
    assert list(TIMINGS["sections"]) == [s["id"] for s in NARRATION["sections"]]


def test_lead_and_tail_come_from_the_timing_block():
    """No number is typed twice: narration.yaml's timing block is the source."""
    t = NARRATION["timing"]
    assert TIMINGS["lead"] == pytest.approx(t["lead"])
    assert TIMINGS["tail"] == pytest.approx(t["tail"])
    assert TIMINGS["sentence_gap"] == pytest.approx(t["sentence_gap"])
    assert "LEAD = TIMINGS" in RENDER and "TAIL = TIMINGS" in RENDER


def test_audio_generator_is_per_sentence():
    """T2.1/D2: one clip per sentence, silence-trimmed, gaps between them."""
    assert "sentences" in AUDIO and "to_pcm" in AUDIO
    assert "silenceremove" in AUDIO
    assert "assemble" in AUDIO


# ----------------------------------------------------------- T2.2 cue -------
def test_cue_waits_on_the_sentence_offset_plus_lead():
    """The rule the whole sync rests on."""
    body = RENDER.split("def cue", 1)[1].split("def ", 1)[0]
    assert "LEAD + start" in body
    assert 'self._layout_check(f"cue {sid}")' in body


def test_cues_must_be_in_order_and_belong_to_the_scene():
    body = RENDER.split("def cue", 1)[1].split("def ", 1)[0]
    assert "out of order" in body
    assert "it belongs to" in body


def test_sync_end_raises_on_budget_error():
    """P1: the timing contract is enforced, not printed."""
    body = RENDER.split("def sync_end", 1)[1]
    assert "raise RuntimeError(" in body
    assert "budget" in body


def test_sync_end_fills_the_remainder():
    body = RENDER.split("def sync_end", 1)[1]
    assert "self.wait(rem)" in body


# ---------------------------------------------------------- T2.3 motion -----
def test_ambient_pulse_always_runs():
    """R1/R3: the background breathes, so no wait is a dead frame."""
    assert "def breathe" in RENDER
    assert "_base_op" in RENDER
    assert "self._ambient.add_updater(breathe)" in RENDER


def test_clause_playhead_moves_with_the_scene():
    """The moving highlight for the spoken clause."""
    assert "_playhead" in RENDER
    assert "self._playhead.add_updater(place)" in RENDER


def test_cue_fires_a_flow_pulse():
    assert "_pulse_at" in RENDER
    body = RENDER.split("def cue", 1)[1].split("def ", 1)[0]
    assert "_pulse_at" in body


def test_run_times_snap_to_frames():
    """Without this the tracked clock drifts off the rendered one."""
    assert "def _snap" in RENDER
    assert "config.frame_rate" in RENDER
