# Fit profiles

A fit profile is a versioned scoring contract: it defines what “fit” means,
which questions Jev must answer from the evidence, and how those answers become
one score. The input supplies evidence; the profile supplies the target.

## Authoring resources

- [`HOW_TO_CREATE_PROFILES.md`](HOW_TO_CREATE_PROFILES.md): actionable guide
  for designing, validating, testing, and versioning a profile.
- [`profile.schema.json`](profile.schema.json): machine-readable JSON Schema for
  profile generation, editor validation, and CI checks.
- [`profile.template.json`](profile.template.json): minimal valid profile to
  copy and edit.
- [`profile_request.template.json`](profile_request.template.json): structured
  demand brief that a person or application can give to an AI generator.
- [`PROFILE_GENERATION_PROMPT.md`](PROFILE_GENERATION_PROMPT.md): canonical
  prompt for converting a demand brief into a profile JSON.
- [`digital_twin.json`](digital_twin.json): production example.
- [`gsd_patient_journey_mapping.json`](gsd_patient_journey_mapping.json):
  production profile for scoring and categorizing solutions that may support
  the glycogen-storage-disease patient journey.

## Quick start

Copy `profile.template.json`, replace every example value, then validate the
result locally (no API call is made):

```powershell
python .\fit.py `
    --profile .\profiles\my_profile.json `
    --validate-profile
```

Score a UTF-8 text, Markdown, CSV, JSON, or JSONL file:

```powershell
python .\fit.py `
    --profile .\profiles\my_profile.json `
    --input .\candidate.md `
    --name "Candidate" `
    --output .\candidate.fit.json `
    --responses-output .\candidate.jev.jsonl
```

Recompute the fit locally after changing only roles, thresholds, weights, or
aggregation:

```powershell
python .\fit.py `
    --profile .\profiles\my_profile.json `
    --responses .\candidate.jev.jsonl `
    --name "Candidate" `
    --output .\candidate.fit.json
```

Saved responses include a digest of the instructions and questions that
produced them. Changing criterion weights, roles, thresholds, or aggregation
does not require new Jev calls. Changing the shared instruction, criterion IDs,
or criterion instructions changes that digest and requires a new live run.

Use `--input - --yes` to read piped UTF-8 text from stdin. Live runs display
the number of Jev requests and require confirmation unless `--yes` is passed.
