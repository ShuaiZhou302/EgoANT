# WGO S2 Full-Cover Segmentation Prompt

This is the local-refinement prompt used by the V5.1 HomER segmentation run.
The model receives one or more timestamped contact sheets for a short local
window and a coarse segment hypothesis for that window.

## System Prompt

```text
You segment egocentric manipulation videos into completed atomic events. Reply with ONLY valid JSON.
```

## User Prompt Template

```text
These contact sheets cover a SHORT local window of an egocentric manipulation video.
Each tile is sampled every {sample_sec:.2f}s. Yellow text is the timestamp in seconds.
Visible time range: [{t0:.2f}, {t1:.2f}] seconds.
Episode instruction (context only): {instruction}
Coarse hypothesis for this window (may be wrong on boundaries): {coarse_hint}

Re-segment ONLY this local window into completed atomic events.
Prefer shorter events (1-5s). Split if multiple state changes are visible.
Return ONLY JSON:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"..."}}]}}
```

## Inputs

- `sample_sec`: frame sampling interval used for contact-sheet tiles.
- `t0`, `t1`: visible time range covered by the current local window.
- `instruction`: episode-level task description, used only as context.
- `coarse_hint`: the coarse segment label and time span being refined.
- images: timestamped contact sheets ordered by time.

## Expected Output

```json
{
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 1.0,
      "subtask": "short completed action"
    }
  ]
}
```

The score reported in V5.1 uses IoU >= 0.75 with outer-boundary snap for
temporal matching, then a semantic judge over matched segment labels.
