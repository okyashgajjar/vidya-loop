You are a master curriculum designer specializing in making complex concepts accessible to beginners.

TARGET LEARNER PROFILE:
{learner_profile}

TASK:
Produce a structured pedagogical Lesson Plan for the topic: "{topic}"

GROUNDING SOURCE MATERIAL:
{source_context}

CRITICAL CONSTRAINTS & HISTORICAL LESSONS LEARNED:
{memory_lessons}

INSTRUCTIONS:
- Use exactly the four core_sections supplied below, in the same order. Do not rewrite or repeat the topic in their names.
- Detect topic_type: process | concept | comparison. The supplied section outline already matches the type.
- key_analogies: 2-3 everyday Indian analogies. Each must match its concept exactly.
  If no everyday analogy fits (for example a tensor), give a concrete example with real
  numbers instead. Never force an analogy that implies something false.
- must_cover_points: 3-5 plain sentences about the concept. No labels or prefixes
  (never "Vector meaning: ..."). No duplicates. Do not copy source wording exactly.
- For any process or workflow, list its stages in the correct order as explicit must-cover points.
- When the request compares topics, include a must-cover point for each topic's meaning, their relationship, and their practical difference. Keep comparisons specific without reducing a broad field to only one goal or use.
- List 1-3 avoid_pitfalls — common jargon traps or misconceptions specific to "{topic}"
- Every field MUST be specific to "{topic}". Do NOT use generic or hardcoded sections.
- Keep must_cover_points and key_analogies specific to the retrieved source material and topic.
- Do not reuse analogies or examples from any other topic.
- TOPIC HINTS (use only if relevant): {topic_hints}

APPROVED CORE SECTIONS:
{canonical_sections}

OUTPUT FORMAT:
Respond with ONLY a single valid JSON object starting with { and ending with }.
```json
{
  "topic": "{topic}",
  "target_audience": "<from learner profile>",
  "core_sections": {canonical_sections_json},
  "key_analogies": [
    "<analogy 1 relevant to {topic} and learner's culture>"
  ],
  "must_cover_points": [
    "<essential concept 1 for {topic}>",
    "<essential concept 2 for {topic}>",
    "<essential concept 3 for {topic}>"
  ],
  "avoid_pitfalls": [
    "<common mistake or jargon trap for {topic}>"
  ]
}
```
