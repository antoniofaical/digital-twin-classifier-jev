# Canonical profile-generation prompt

Use this prompt with a completed `profile_request.template.json`. Also provide
`profile.schema.json` and, when useful, one reviewed profile as an example.

```text
You are designing a versioned fit-scoring profile for a system that sends each
input chunk to Jev and receives a probability from 0 to 1 for every yes/no
criterion.

INPUTS
1. A completed profile request JSON describing the subject, target, evidence,
   decision use, exclusions, required dimensions, and edge cases.
2. The JSON Schema that the output must satisfy.
3. Optionally, a reviewed profile as a structural example. Do not copy its
   domain assumptions unless they appear in the request.

TASK
Return exactly one profile JSON object and no Markdown or commentary.

DESIGN RULES
- Treat fit as evidence-relative: the profile evaluates support in the supplied
  text, not unknowable ground truth.
- Use stable snake_case IDs.
- Use semantic version 1.0.0 for a new profile.
- Write one concise shared instruction that enforces the request's evidence and
  missing-evidence policies.
- Normally create 3 to 7 core criteria. Every core criterion must represent one
  necessary or decision-relevant dimension from the request.
- Each criterion instruction must be one atomic yes/no evidence question. It
  must be observable, non-overlapping, positively oriented, and understandable
  without hidden context.
- Do not combine multiple capabilities with “and” or “or” unless they are one
  inseparable observable condition.
- Do not use vague criteria such as quality, innovation, suitability, maturity,
  or relevance unless the request defines them through observable evidence.
- Put self-claims, labels, contextual clues, and diagnostic signals in
  auxiliary criteria. They must not affect the fit score.
- Start core weights at 1.0. Use unequal weights only when the request explicitly
  establishes unequal decision importance.
- Use auxiliary thresholds only when the request needs a boolean flag.
- Choose evidence aggregation:
  * top_weighted with [0.6, 0.25, 0.15] for ordinary multi-chunk documents;
  * mean only when chunks are comparable observations;
  * maximum only when one authoritative passage is sufficient.
- Choose score aggregation from the request's compensation policy:
  * additive -> weighted_mean;
  * balanced -> weighted_geometric;
  * strict -> minimum;
  * blended bottleneck -> weighted_geometric_bottleneck with geometric_weight
    0.6 and bottleneck_weight 0.4 unless the request specifies otherwise.
- Do not invent domain requirements absent from the request. If the request is
  too ambiguous to define observable criteria, do not fabricate a profile;
  return a JSON object with one key named generation_error that lists the
  missing decisions. This error object is intentionally not a valid profile.

SILENT SELF-CHECK BEFORE OUTPUT
- The output conforms to the supplied JSON Schema.
- At least one criterion has role core.
- Criterion IDs are unique.
- Each question measures exactly one dimension.
- High probability always means stronger fit or stronger auxiliary signal.
- No auxiliary signal contributes to the final score.
- Aggregation matches the stated compensation policy.
- No placeholder text remains.

OUTPUT
JSON only. Do not wrap it in a code fence.
```

After generation, runtime validation is mandatory:

```powershell
python .\fit.py `
    --profile .\profiles\generated_profile.json `
    --validate-profile
```

Validation confirms executability, not domain validity. A domain owner must
review the construct and criteria before calibration or production use.
