# Caption Prompt

Source of truth: `SYSTEM_PROMPT`, `CAPTION_PROMPTS`, and `PROMPT_BODY` in
`scripts/annotate_video.py`.

## System Prompt

```text
You are a precise video-frame annotator for egocentric human-object interaction clips.
```

## Image Inputs

The caption stage sends images before the user text. The exact image set
depends on `--caption-input`:

- `raw`: 10 ordered raw RGB frames sampled uniformly from the candidate segment.
- `overlay`: 10 ordered frames sampled uniformly from the candidate segment, with
  a 2D hand trajectory overlay. The trajectory color encodes temporal progress.
- `raw-hand-crop`: 10 ordered raw frames, followed by one hand-focused crop grid
  from the same current segment.
- `temporal-collage`: three ordered grids: PAST context, CURRENT target, and
  FUTURE context. Each grid contains 6 raw frames in row-major order. The model
  must label only CURRENT.

## User Prompt: raw

```text
I will send you 10 ordered raw frames sampled uniformly from one candidate atomic action clip.

The frames come from an egocentric manipulation video and are ordered from earliest to latest. Respect the frame order.

Write one short imperative instruction for the visible physical action.

Focus on:
1. active hand or hands;
2. the main object being acted on or directly manipulated;
3. other involved objects, such as a container, target, tool, or support;
4. concrete action verb;
5. visible contact or grasp;
6. visible final state or final location.

Rules:
- Describe only visible facts from this clip.
- Do not use hidden intent or scene context to guess the action.
- The instruction must be an action command. Do not output prompt or meta text
  such as "Describe the action", "the clip shows", or "in this clip".
- Do not mention the reference image, frames, camera, or any meta notes.
- Do not address the viewer or use second-person words such as "you" or
  "your". Refer to the left hand, right hand, or both hands directly.
- Do not number items, do not add commentary, and do not use code fences.
- Name an object only when it is visibly involved in the action: touched by the hand, approached by the hand with clear intent, manipulated by a held tool, or visibly affected by a held tool.
- Before naming main_object, check whether its size is plausible for the
  visible container, tool, hand grasp, and motion.
- If an item is moved into or out of a container, main_object is the moved item;
  put the container in involved_objects.
- If a held tool acts on another item, main_object is the affected item; put the
  tool in involved_objects.
- Use a basic object noun for main_object.name.
- Do not include color, size, shape, material, or variety adjectives in main_object.name.
- If unsure of the exact object name, use a generic noun such as 'object', 'block', 'handle', or 'container'.
- If several objects participate, list them in involved_objects with short roles. Keep at most 4 objects.
- Avoid vague verbs. Prefer concrete verbs: approach, align, open the hand, close the hand, grasp, rotate, tilt, release, lift, lower, move forward/backward/left/right/up/down, push, pull, slide, insert, retract, place, hold.
- Do not use "transfer" in instruction, alternatives, or verb. Use the visible
  concrete action instead, such as move, lift, place, insert, remove, or pour.
- Return 3 to 5 alternatives. Alternatives must preserve the same meaning and must not add new facts.

Return only valid JSON:
{
  "instruction": "canonical one-sentence imperative instruction",
  "alternatives": ["3 to 5 equivalent descriptions"],
  "actor": "left hand | right hand | both hands | unknown",
  "verb": "main visible action verb",
  "main_object": {
    "name": "basic noun for the main object being acted on | unknown",
    "name_candidates": [{"name": "possible object name", "confidence": 0.0}],
    "confidence": 0.0
  },
  "involved_objects": [
    {"name": "basic noun", "role": "primary | container | target | tool | support | other"}
  ],
  "contact": "visible contact or grasp | unknown",
  "final_state": "visible final state/location | unknown",
  "confidence": 0.0,
  "needs_review": false
}
```

Other caption-input modes use the same body with a different opening paragraph
describing the image inputs listed above.
