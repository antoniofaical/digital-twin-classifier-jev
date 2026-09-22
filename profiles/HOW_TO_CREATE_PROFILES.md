# How to create a fit profile

This guide turns a scoring demand into a profile that is useful to a person,
valid for the CLI, and reproducible by an AI generator.

## 1. Define the decision before the criteria

Write down these five items first. If any is unclear, the score will also be
unclear.

| Item | Question to answer | Example |
| --- | --- | --- |
| Subject | What kind of thing receives a score? | A startup |
| Target | What does 100% fit mean? | Strong functional adherence to a digital twin |
| Evidence | What text may support the score? | Public website pages |
| Use | What decision will the score inform? | Research prioritization |
| Exclusions | What must not count as evidence? | Marketing labels without functionality |

The fit score is not an absolute probability that the subject “is” something.
It is the degree to which the supplied evidence satisfies this version of the
profile. A result should therefore always be interpreted with its profile ID,
version, and evidence set.

For repeatable or AI-assisted authoring, copy
`profile_request.template.json` and fill every field before designing criteria.

## 2. Convert the target into atomic criteria

Use three to seven `core` criteria in most profiles. A core criterion changes
the final score. Use `auxiliary` criteria for useful context or flags that must
not change the score.

Every criterion should:

1. ask one yes/no question;
2. be answerable from the supplied evidence;
3. describe observable evidence, not an opinion about overall quality;
4. measure one dimension that is not already covered by another criterion;
5. make a high probability unambiguously mean more fit.

Prefer:

> Does the evidence show measured data from the specific subject being used to
> initialize or personalize its corresponding model?

Avoid:

> Is the solution innovative, mature, integrated, and probably a good digital
> twin?

The second question is subjective, combines several dimensions, and cannot be
calibrated reliably.

Do not make a self-claim a core criterion. “The company calls this a digital
twin” is useful provenance, but it is not evidence that the required functions
exist. Represent it as an auxiliary criterion with an optional threshold.

## 3. Write the shared instruction

`instruction` applies to every input chunk and every criterion. It should state
the evidence policy, not repeat all criteria. A sound default is:

```text
Judge only capabilities explicitly supported by the supplied evidence. Do not
infer missing capabilities from branding, industry, or general plausibility.
```

Add domain-specific restrictions only when necessary. For example, a funding
fit profile might say that announced but uncommitted funding does not count.

## 4. Assign roles, weights, and thresholds

Each criterion needs:

- `id`: stable `snake_case` identifier;
- `role`: `core` or `auxiliary`;
- `instructions`: its atomic evidence question.

Core criteria may have a positive `weight`; the default is `1.0`. Start with
equal weights unless the decision definition clearly makes one dimension more
important. Weights express importance, not model confidence.

Auxiliary criteria may have a `threshold` from `0` to `1`. When present, the
result includes a boolean flag. The raw auxiliary score is always preserved.

## 5. Choose evidence aggregation

One input can be split into several chunks. Evidence aggregation produces one
score per criterion across those chunks.

| Method | Use when | Main risk |
| --- | --- | --- |
| `top_weighted` | A few independent passages can establish a capability | One strong claim may dominate if weights are too concentrated |
| `mean` | Every chunk is a comparable observation | Long irrelevant inputs dilute the score |
| `maximum` | One authoritative passage is sufficient | Most sensitive to isolated false positives |

For documents and crawled sites, begin with:

```json
{
  "method": "top_weighted",
  "top_weights": [0.6, 0.25, 0.15]
}
```

`top_weights` are applied to the strongest distinct evidence units. They are
normalized automatically and must be positive.

## 6. Choose score aggregation

Score aggregation combines core criterion scores into the final fit score.

| Method | Behavior | Appropriate target |
| --- | --- | --- |
| `weighted_mean` | Strong dimensions compensate for weak ones | Additive preferences |
| `weighted_geometric` | Penalizes weak dimensions | All dimensions matter |
| `minimum` | The weakest dimension determines fit | Strict conjunction / gate |
| `weighted_geometric_bottleneck` | Blends overall balance with the weakest dimension | Functional definitions with a meaningful bottleneck |

For a target that requires all capabilities but should not collapse entirely
because one criterion is uncertain, start with:

```json
{
  "method": "weighted_geometric_bottleneck",
  "geometric_weight": 0.6,
  "bottleneck_weight": 0.4
}
```

## 7. Create and validate the JSON

Copy `profile.template.json`, save it under `profiles/`, and validate it:

```powershell
Copy-Item .\profiles\profile.template.json .\profiles\my_profile.json

python .\fit.py `
    --profile .\profiles\my_profile.json `
    --validate-profile
```

The command checks the runtime contract and prints the `question_set_sha256`.
It does not read an API key or make a network call.

The JSON Schema in `profile.schema.json` is intended for IDEs, generators, and
CI. Runtime validation remains authoritative because it also checks rules such
as unique criterion IDs.

## 8. Run a small pilot before bulk scoring

Use inputs that cover at least:

- clear positive cases;
- clear negative cases;
- near-boundary cases;
- sparse or ambiguous evidence;
- misleading self-claims;
- more than one language, if production data is multilingual.

For a first pilot, 20 to 30 deliberately varied cases is enough to expose bad
questions. For calibration, use an independently labelled set, record human
disagreement, and select thresholds or weights against the downstream decision
metric rather than against intuition alone.

Example live run:

```powershell
python .\fit.py `
    --profile .\profiles\my_profile.json `
    --input .\examples\positive_case.md `
    --name "Positive case" `
    --output .\out\positive_case.fit.json `
    --responses-output .\out\positive_case.jev.jsonl
```

Inspect `criterion_scores`, `main_strength`, `main_gap`, and
`criterion_evidence`; do not assess only the final number.

## 9. Version safely

Use semantic versions for profiles and preserve old files or commits so results
remain reproducible.

| Change | Suggested bump | Can reuse saved Jev responses? |
| --- | --- | --- |
| Description or documentation only | patch | Yes |
| Core weights, auxiliary thresholds, or aggregation | minor | Yes |
| Criterion role | minor | Yes, if IDs and questions are unchanged |
| Shared instruction | major | No |
| Criterion ID, question, addition, or removal | major | No |

The CLI verifies this boundary with `question_set_sha256`. An offline replay is
rejected when saved responses came from different instructions or questions.

## AI-assisted generation

1. Fill `profile_request.template.json` with the demand.
2. Give an AI the completed request, `profile.schema.json`,
   `PROFILE_GENERATION_PROMPT.md`, and one relevant example such as
   `digital_twin.json`.
3. Require JSON-only output.
4. Save the output and run `--validate-profile`.
5. Have a domain owner review the criteria and evidence policy.
6. Pilot and calibrate before treating the score as decision-grade.

Schema validity proves that a profile is executable; it does not prove that the
construct being measured is valid. Human review and calibration are mandatory
for high-impact decisions.

## Release checklist

- [ ] The subject, target, evidence, use, and exclusions are explicit.
- [ ] There are three to seven non-overlapping core criteria, unless justified.
- [ ] Every question is atomic, evidence-grounded, and positively oriented.
- [ ] Self-claims and contextual signals are auxiliary.
- [ ] Weights reflect decision importance, not confidence.
- [ ] Evidence and score aggregation match the decision logic.
- [ ] `--validate-profile` succeeds.
- [ ] Positive, negative, boundary, sparse, and misleading cases were tested.
- [ ] A domain owner approved the profile.
- [ ] The profile version and calibration set are recorded.
