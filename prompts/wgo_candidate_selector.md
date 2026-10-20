# WGO Candidate Selector Prompt

Source of truth for the reported V5.1 result: `SYSTEM` and `PROMPT` in
`scripts/wgo_bench/select_pred_segment_candidates.py`.

## Image Inputs

The selector receives three images in order: a five-frame PREV contact sheet,
a five-frame CURRENT contact sheet, and a five-frame NEXT contact sheet. It
also receives four text candidates. Gold labels are never provided.

## System Prompt

```text
You choose the best subtask label for an egocentric manipulation video segment. Return one concise imperative phrase. Do not narrate the actor. Reply with ONLY JSON: {"label": "...", "source": "A|B|C|D|corrected"}.
```

## User Prompt Template

```text
I will send PREV, CURRENT, and NEXT contact sheets for one predicted segment.
Choose the best label for the CURRENT segment only.

Candidate labels:
A raw-frame label: {raw_label}
B raw-prior neighbor label: {rawprior_label}
C segmentation-prior neighbor label: {seed_label}
D ffmpeg raw-frame label: {ffmpeg_label}

Candidate A is usually the most stable. Prefer A unless the CURRENT sheet clearly supports another candidate.
Use PREV and NEXT only to understand what changed at the boundaries.
If all candidates miss an obvious action, output a corrected short label.

Episode task, for context only:
{instruction}
```

`scripts/wgo_bench/run_wgo_candidate_selector.py` remains available as a
generic two-stage selector, but it is not the exact selector used for the
reported `0.1542` result.
