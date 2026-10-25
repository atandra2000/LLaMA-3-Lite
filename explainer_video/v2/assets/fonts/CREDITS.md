# Bundled fonts

Both families are registered at import by `render_scenes.py`
(`manimpango.register_font`) so a render does not depend on the host's
macOS-only Avenir Next / DIN Alternate. Set `USE_SYSTEM_FONTS=1` to fall back
to those—used only to diff against the v2 baseline.

| Family | Files | Role | Weights drawn from |
|---|---|---|---|
| **Inter** | `Inter-Regular.ttf`, `Inter-Medium.ttf`, `Inter-SemiBold.ttf`, `Inter-Bold.ttf` | display / body (`FONT`) | Regular, Medium, SemiBold, Bold |
| **Barlow Semi Condensed** | `BarlowSemiCondensed-Regular.ttf`, `BarlowSemiCondensed-Medium.ttf`, `BarlowSemiCondensed-Bold.ttf` | tracked technical labels (`DIN`) | Regular, Medium, Bold |

## Sources and licences

- **Inter 4.1** — <https://github.com/rsms/inter/releases/tag/v4.1>, static
  instances from `extras/ttf/`. SIL Open Font License 1.1 — see
  `OFL-Inter.txt`.
- **Barlow Semi Condensed** — <https://github.com/google/fonts/tree/main/ofl/barlowsemicondensed>.
  SIL Open Font License 1.1 — see `OFL-BarlowSemiCondensed.txt`.

Both licences permit bundling and redistribution with the rendered work.
