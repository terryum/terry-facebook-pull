# CLAUDE.md

This project builds a "personal external memory system" from Facebook posts.

## 0. Project Context

- Input: Facebook post archive (JSON)
- Output: Obsidian markdown files
- Goal: Extract reusable thoughts, not store raw data

Important:
- Archive = raw memory
- Synthesized = reusable knowledge
- Only Synthesized is used for most AI tasks

---

## 1. Think Before Coding

Do not start coding immediately.

Before implementing:
- Clarify what stage you are working on:
  (parse / filter / classify / cluster / synthesize)
- State assumptions about input/output format
- If uncertain, ask instead of guessing

---

## 2. Simplicity First

- No over-engineering
- No generic frameworks
- No unnecessary abstractions

Each script should do ONE thing only.

---

## 3. Pipeline Discipline

Always follow this pipeline:

1. Parse (JSON → normalized records)
2. Filter (heuristic)
3. Classify (LLM)
4. Cluster (embedding)
5. Synthesize (LLM)
6. Export (markdown)

Do NOT mix stages.

---

## 4. Data Integrity

- Never modify raw Archive data
- Always write new files instead of overwriting
- Keep intermediate outputs for debugging

---

## 5. Cost Awareness

- Use cheap models (Haiku) for classification
- Use stronger models (Sonnet) only for synthesis
- Avoid unnecessary repeated LLM calls

---

## 6. Output Format Discipline

All outputs must be deterministic and structured.

Examples:
- JSON → strict schema
- Markdown → consistent frontmatter

---

## 7. Goal-Driven Execution

Each step must be verifiable:

Example:
- Filter → count before/after
- Cluster → number of clusters
- Synthesis → file count

---

## 8. Do Not Overreach

- Do not redesign Obsidian structure
- Do not introduce new storage systems
- Do not build a UI

Stay within scope.

---

## 9. Definition of Done

The system is complete when:
- Facebook posts are converted to markdown
- Meaningful posts are filtered
- Clusters are formed
- Synthesized notes are generated
- Files are usable in Obsidian
