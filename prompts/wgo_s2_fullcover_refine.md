# WGO S2 Full-Cover Refinement Prompt

Source of truth: this prompt was used by the V5.1 HomER S2 local-refinement
run. The local refinement job receives a coarse segment hypothesis and
timestamped contact sheets for only that local window.

## System Prompt

```text
You segment egocentric manipulation videos into completed atomic events. Reply with ONLY valid JSON.
```

## Image Inputs

The request appends the precomputed contact sheets whose time spans overlap the
current coarse segment. Each tile is ordered by time and contains a visible
timestamp in seconds. With `pad=0`, the declared refinement interval is not
expanded and returned boundaries are clipped to it. Because sheets are reused
at their original 20-tile granularity, the first and last sheet can still show
neighboring tiles outside that interval; this is part of the reported V5.1
implementation.

## User Prompt Template

```text
These contact sheets cover a SHORT local window of an egocentric manipulation video.
Each tile is sampled every {sample_sec:.2f}s. Yellow text is the timestamp in seconds.
Visible time range: [{t0:.2f}, {t1:.2f}] seconds.
Episode instruction (context only): {instruction}
Coarse hypothesis for this window (may be wrong on boundaries): {coarse_hint}

Re-segment this entire local window into completed atomic manipulation events.
Return a gap-free, non-overlapping timeline that covers the whole visible range from {t0:.2f} to {t1:.2f} seconds.
Prefer completed world-state changes: pick, place, open, close, pour, wipe, cut, fold, tighten, insert, remove.
Do NOT create segments for passive inspection, small repositioning, grasp adjustment, hovering, approach, retreat, or camera motion unless a visible object state is completed.
Do NOT split continuous drawing/cutting/wiping into stroke-level micro-actions unless the object/state target clearly changes.
If a long continuous manipulation is visible, keep it as one segment rather than many micro-segments.
Use timestamp boundaries and keep segments sorted.
Return ONLY JSON:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"..."}}]}}
```
