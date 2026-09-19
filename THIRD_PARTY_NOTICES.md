# Third-Party Notices

This repository is MIT licensed (see [`LICENSE`](LICENSE)). The components below
are bundled third-party works and keep their own licenses.

## archify

- Path: `.claude/skills/archify/`
- Upstream: <https://github.com/tt-a1i/archify>
- License: MIT (see `.claude/skills/archify/LICENSE` and
  `.claude/skills/archify/THIRD_PARTY_NOTICES.md` for its own dependency notices)
- Vendored as diagram-rendering tooling. It is deliberately trimmed: the rendered
  `examples/*.html` showcases and the `test/` fixtures are not vendored because
  they are regenerable (`node scripts/render-examples.mjs`) and made up 64% of
  tracked bytes. See `CLAUDE.md` for the re-vendoring rule.
