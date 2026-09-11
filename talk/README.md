# dSABRE — general-audience talk

`dSABRE_talk.pptx` — 22 slides, 16:9, with speaker notes on every slide
(~3800 words of notes, roughly a 27-minute talk at a normal speaking pace).

Written for an audience with **no quantum computing background**: slides 3–10
build up qubits, circuits, routing, modular hardware, and teleportation before
the research contribution starts at slide 11.

Nothing outside this folder was modified.

## Structure

| # | Slide | Figure |
|---|-------|--------|
| 1 | Title | — |
| 2 | Roadmap: three questions | — |
| 3 | Bits, and what a qubit does differently | — |
| 4 | A quantum program is a circuit | `dag_example` (paper Fig. 1) |
| 5 | Qubits can only talk to their neighbours | `routing_onechip` (new) |
| 6 | SABRE: the routing heuristic everyone ships | — |
| 7 | Why quantum computers are going modular | `architectures` (paper Fig. 2) |
| 8 | The currency: an EPR pair | — |
| 9 | How a qubit crosses between chips | `teleport_protocol` (new) |
| 10 | Two ways to spend a pair (teledata / telegate) | — |
| 11 | **A teleport is not just an expensive SWAP** | `swap_vs_tele` (new) |
| 12 | And a full chip is a trap | — |
| 13 | Score the whole move, as one thing | — |
| 14 | One decision, worked through | `teleport_candidates` (paper Fig. 4) |
| 15 | The routing loop | `flowchart_talk` (paper Fig. 3) |
| 16 | Three rules that make getting stuck impossible | — |
| 17 | About half the entanglement, everywhere | `chart_headline` |
| 18 | Not an artefact of averaging | `chart_percircuit` |
| 19 | It keeps working as the machine grows | `chart_scale` |
| 20 | Which parts are actually doing the work? | `chart_ablation` |
| 21 | How to read these numbers honestly | — |
| 22 | Three things to take away | — |

Slide 11 is the conceptual pivot; if the audience already knows quantum
computing, starting there works.

## Rebuilding

```bash
python3 build_deck.py     # writes dSABRE_talk.pptx from img/
python3 preview.py        # renders preview/slide-NN.png + a QA report
```

`preview.py` exists because there is no LibreOffice on this machine. It re-reads
the saved `.pptx` through `python-pptx` and repaints every shape with PIL, using
the Calibri and Cambria files that ship inside `Microsoft PowerPoint.app`, so
text wrapping and overflow are measured with the metrics PowerPoint will use.
It reports text that overruns its box, shapes that leave the slide, and pictures
that intersect a card. A clean run prints `no geometry or overflow issues
detected`.

## Figures

`img/` holds the rendered PNGs the deck embeds. Four come from the paper and
seven are new; sources for the new ones are in `src/`.

**From the paper** (`../paper/fig/*.tex`, recompiled standalone at 400 dpi):
`architectures`, `dag_example`, `teleport_candidates`, and `flowchart_talk` —
the last being `flowchart_compact.tex` with its two `\ref{eq:…}` replaced by
plain words, since a standalone compile has no `.aux` to resolve them against
and they would otherwise print as `Eq. ??`.

**New for the talk** (`src/*.tex`, same TikZ style as the paper):
`teleport_protocol` (the three-step protocol), `swap_vs_tele` (the SWAP versus
teleport contrast behind slide 11), `routing_onechip` (single-chip routing).
Build any of them with:

```bash
sed 's/FIGNAME/teleport_protocol/' src/wrap.tex > build.tex && pdflatex build.tex
```

**Charts** (`src/charts.py`, `src/chart_scale.py`, matplotlib):
`chart_headline`, `chart_percircuit`, `chart_ablation`, `chart_scale`.
Keep the scale chart in `chart_scale.py` only — an earlier duplicate of it in
`charts.py` silently overwrote the good version and a stale figure shipped into
slide 19 before QA caught it.

## Where the numbers come from

Every figure on a slide is from the current revision of `../paper/dsabre.tex`:

- 50.8 / 62.0 / 49.1 % EPR reduction — Table `tab:main` geometric means
- 52.0 % ring, 41.5 % star — Sec. IV-C, heavy-hex
- 45/45 vs 40/45 instances, 48 s for 360-qubit QFT — Sec. IV-B, IV-C
- 1.9 s vs 4.7 s, exponent 0.97 (R² 0.95) — Sec. IV-C runtime
- +12.4 / +9.4 / +0.2 / +87.8 % ablation — Table `tab:mech`
- A/B/C worked example — Table `tab:example`
- 15.75 gates per teledata move, +93.5 % for telegate — Sec. VI
- pytket-dqc −56.1 / −13.8 / +15.3 %, unbuildable 4-of-6, 2-of-6, ≥6-of-9,
  −59.6 % under a three-gate lifetime cap — Table `tab:fair`, Sec. IV-E
- recovery: 0 reported routes at 25–64 q; 6, 10, 66 gates at 100/200/360 q;
  1074 transactions — Table `tab:tier2`

If the manuscript's numbers move, these slides need updating with it.
