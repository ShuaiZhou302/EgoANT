# Merge Judge Prompt

Source of truth: `SYSTEM_PROMPT` and `JUDGE_PROMPT` in
`scripts/merge_judge_segments.py`.

## System Prompt

```text
You are a precise video-frame annotator for egocentric human-object interaction clips.
```

## Image Inputs

When `--include-middle-frame` is enabled, the request appends one middle-frame
image for each source segment in the current judge window. The text prompt adds
lines of the form:

```text
Middle frame images:
Image 1 corresponds to source segment <seg_id>.
Image 2 corresponds to source segment <seg_id>.
...
```

The image kind is controlled by `--frame-kind raw|overlay`.

## User Prompt Template

The fixed judge instruction is:

```text
You are given an overlapping window of ordered candidate atomic action annotations from one egocentric manipulation video.

Decide whether each adjacent boundary should be merged.

Merge only when:
- the two descriptions refer to the same visible continuous subtask;
- the main object is the same, or the names are visually compatible aliases for the same object;
- the descriptions read as consecutive parts of one short subtask;
- hand/tool contact, motion, or object state changes are temporally coherent;
- no clear reset to a different subtask.
- A `pure_static_candidate` normally should not remain an atomic action by
  itself. Attach it to exactly one neighboring subtask when the images and
  shared objects support continuity:
  - attach left when it is the settling or maintained final state of the left action;
  - attach right when it is the grasp/stabilization preparation for the right action;
  - if both sides are different actions, choose only the better-supported side;
  - never use one static segment to bridge two otherwise distinct actions.

Keep split when:
- either segment has instruction `N/A`;
- the main object clearly changes to a different object;
- the next segment starts an unrelated action or a new repeated trial;
- do not merge across a completed action cycle;
- the left segment already reaches a visible final state and the right segment
  starts a new transfer or repetition;
- the segments describe incompatible outcomes;
- the images or annotations are too ambiguous to support merge.
- Preserve a hold as a separate action when it is visibly task-relevant, such
  as deliberately stabilizing multiple objects or a tool, and is not merely a
  repeated description of the neighboring action.

Object consistency:
- Use images, contact, final_state, main_object, involved_objects, and main_object.name_candidates together.
- Do not split only because one segment uses a generic noun and the next uses a more specific noun, if they visually refer to the same manipulated object.
- Do not merge only because two segments happen near the same container or scene.
- Do not invent object names.

Return only valid JSON:
{
  "boundary_decisions": [
    {
      "left_seg_id": 0,
      "right_seg_id": 1,
      "merge": false,
      "reason": "brief evidence-based reason",
      "confidence": 0.0
    }
  ]
}
```

At runtime the script appends the current window annotations and expected
adjacent boundary ids.
