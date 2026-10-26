# Release pipeline for the LLaMA-3-Lite architecture explainer.
#
#   make check     fast static gate: facts, script lint, doc sync, tests
#   make layout    480p render of every scene with the safe-rect guard
#   make audio     re-record the narration and rebuild audio/timings.json
#   make scenes    1080p30 render of every scene
#   make video     scenes + assemble the master
#   make verify    the release gate (tools/verify_video.py)
#   make all       check + audio + video + verify
#
# Scene classes are discovered from render_scenes.py, so adding a section
# does not mean editing this file. PLAN.md §6 also names `make sheet` and a
# `build_timeline.py`; neither is written yet.

PY := .venv/bin/python
V2 := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# One manim class per section, in file order: Section01 Section02 ...
SCENES := $(shell $(PY) -c "import re;print(' '.join(re.findall(r'^class (Section\\d+)\\(Section\\)', open('render_scenes.py').read(), re.M)))")

.PHONY: all check layout audio scenes video verify facts clean

all: check audio video verify

check:
	$(PY) tools/check_facts.py
	$(PY) tools/lint_script.py --strict
	$(PY) tools/build_script_doc.py --check
	$(PY) -m pytest tests -q

facts:
	$(PY) tools/build_facts.py
	$(PY) tools/check_facts.py

layout:
	mkdir -p /tmp/explainer-layout
	@for s in $(SCENES); do \
		echo "=== $$s ==="; \
		$(PY) render_scenes.py $$s --check_layout -ql --format=mp4 \
			> /tmp/explainer-layout/$$s.log 2>&1 || exit 1; \
	done
	@grep -h 'layout:' /tmp/explainer-layout/*.log

audio:
	$(PY) generate_audio.py

scenes:
	@for s in $(SCENES); do \
		echo "=== $$s ==="; \
		$(PY) render_scenes.py $$s -qm -r 1920,1080 --format=mp4 || exit 1; \
	done

video: scenes
	$(PY) build_full_video.py

verify:
	$(PY) tools/verify_video.py

clean:
	rm -f concat_list.txt _chapters.txt _mix.wav _narration.m4a _video_silent.mp4
