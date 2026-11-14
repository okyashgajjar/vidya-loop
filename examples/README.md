# Examples

Reviewers may not be able to run the app live. These files document the expected behaviour of the pipeline.

| File | Description |
| --- | --- |
| `passing_lesson.md` | A lesson on Introduction to RAG that passes every native and judge check. |
| `error_injection_run.json` | Rejection log from a run with `--inject-error paraphrased_falsehood`. Shows the native tier passing (the paraphrased claim is not in the forbidden list) and the judge tier catching it via `JUDGE_ACCURACY_GROUNDING` with a quoted source-pack line. |
| `final_lesson.md` | The revised lesson after the judge rejection, showing the false claim removed and the corrected explanation grounded in sources. |
