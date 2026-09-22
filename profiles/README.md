# Fit profiles

A fit profile is the versioned contract that defines what an input is being
compared against. The generic scorer never invents this target from the input.

Required top-level fields:

- `profile_schema_version`: currently `1`.
- `id`, `version`, and `name`: stable profile identity.
- `instruction`: shared instruction sent with every input chunk.
- `criteria`: one or more Jev yes/no probability questions.
- `score_aggregation`: rule used to combine core criteria.

Each criterion contains an `id`, `role` (`core` or `auxiliary`), and
`instructions`. Core criteria may define a positive `weight`. Auxiliary
criteria may define a `threshold`, which produces a boolean flag without
changing the core fit score.

Supported evidence aggregation methods:

- `top_weighted`: combine the strongest distinct evidence units using
  `top_weights`.
- `mean`: average all distinct evidence units.
- `maximum`: use the strongest distinct evidence unit.

Supported core score aggregation methods:

- `weighted_mean`
- `weighted_geometric`
- `weighted_geometric_bottleneck`
- `minimum`

`digital_twin.json` is the first production profile and preserves the existing
digital-twin scoring behavior.

Score a UTF-8 text, Markdown, CSV, JSON, or JSONL file:

```powershell
python .\fit.py `
    --profile .\profiles\digital_twin.json `
    --input .\candidate.md `
    --name "Candidate" `
    --output .\candidate.fit.json `
    --responses-output .\candidate.jev.jsonl
```

Recompute the fit locally after changing only weights or aggregation:

```powershell
python .\fit.py `
    --profile .\profiles\digital_twin.json `
    --responses .\candidate.jev.jsonl `
    --name "Candidate" `
    --output .\candidate.fit.json
```

Use `--input - --yes` to read piped UTF-8 text from stdin. Live runs display
the number of Jev requests and require confirmation unless `--yes` is passed.
