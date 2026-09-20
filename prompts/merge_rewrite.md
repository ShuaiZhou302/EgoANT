# Merge Rewrite Prompt

Source of truth: `REWRITE_PROMPT` in `scripts/merge_judge_segments.py`.

## Image Inputs

The request appends one middle-frame image for each source segment in the
candidate group. The prompt text maps image order to source segment id.

## User Prompt Template

```text
You are given one candidate group of consecutive source segments. A first merge stage proposed that the group may form one visible atomic action.

Use the source annotations and ordered middle-frame images as evidence.

Approve the merge only if the complete group is one continuous visible subtask:
- same main object, or very similar names that visually could refer to the same object;
- the source instructions are consecutive parts of one short subtask;
- hand/tool contact, motion, or object state changes are temporally coherent;
- no clear reset to a different subtask.

If a source segment has motion_hint=`pure_static_candidate`, treat it as a
holding/stabilization phase rather than an independent motion. Include it in
the merged action when it prepares or maintains the neighboring manipulation.
Do not let it bridge two different actions. Preserve it separately only when
stabilization itself is visibly the task-relevant goal.

Reject the merge if the group crosses a completed action cycle, or if an earlier
segment already reaches a visible final state and a later segment starts a new
transfer or repetition.

If the group should not be merged, return merge=false and do not rewrite it.

If the group should be merged, write a new annotation for the whole group.
Do not merge a group that contains any source segment with instruction `N/A`.

Writing style:
- Do not address the viewer or use second-person words such as "you" or
  "your". Refer to the left hand, right hand, or both hands directly.
- Do not use "transfer" in instruction, alternatives, or verb. Use the visible
  concrete action instead, such as move, lift, place, insert, remove, or pour.
- A meaningful hold may remain the main action when the complete group visibly
  stabilizes objects. Do not replace every use of "hold" mechanically.

Object naming:
- Use a basic noun for main_object.name.
- Do not include color, size, shape, material, or variety adjectives in main_object.name.
- If unsure of the exact object name, use a generic noun such as 'object', 'block', 'handle', or 'container'.
- If multiple objects participate, list them in involved_objects with roles.
- Use source main_object.name_candidates only as supporting evidence.
- Keep alternate names in object_aliases. Do not put aliases into main_object.name.
- Do not invent an object name from scene context.

Return 3 to 5 alternative descriptions that preserve the same visible meaning.

Return only valid JSON:
{
  "merge": true,
  "reason": "brief evidence-based reason",
  "annotation": {
    "instruction": "one canonical imperative instruction",
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
    "object_aliases": ["alternate names from source segments"],
    "contact": "visible contact or grasp | unknown",
    "final_state": "visible final state/location | unknown",
    "confidence": 0.0,
    "needs_review": false
  }
}
```

At runtime the script appends the candidate source segments as JSON.
