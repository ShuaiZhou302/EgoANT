# WGO V5.1 Coarse Contact-Sheet Segmentation Prompt

Source of truth for the reported V5.1 result: `SYSTEM` and
`USER_PROMPT_GEPA` in `scripts/wgo_bench/contact_sheet_segment.py`.

## System Prompt

```text
You segment egocentric manipulation videos into completed atomic events. Reply with ONLY valid JSON.
```

## Image Inputs

The user message is followed by all timestamped contact sheets for the episode.
Frames are sampled every `0.5` seconds, every tile is `224 x 224` pixels, and
20 tiles are packed into each sheet in a five-column grid. Yellow text on each
tile gives its timestamp. The images are sent in chronological order.

## User Prompt Template

```text
Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Context:
- Each tile is a frame sampled every {sample_sec:.2f}s. Yellow text on each tile is the timestamp in seconds.
- Tiles are ordered left-to-right, top-to-bottom; sheets are in time order.
- Visible time range in this call: [{t0:.2f}, {t1:.2f}] seconds.
- Episode instruction (context only): {instruction}

Return only JSON with this shape:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}}]}}

Rules:
- Segment only completed robot manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are okay only for fast pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
- start_sec/end_sec must be within [{t0:.2f}, {t1:.2f}], non-overlapping, sorted by time.
```
