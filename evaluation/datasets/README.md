# PaperLens evaluation datasets

Datasets use [schema-v1.json](schema-v1.json) and are loaded by
`evaluation.benchmark`. Every case needs a stable `case_id`, split, category,
query, and source annotation. `source.kind` is `real_document`,
`synthetic_fixture`, or `mocked_result`; these labels must describe where the
case actually came from.

For retrieval labels, set `retrieval_evaluation: true` and add
`relevant_evidence` entries with exact IDs and a granularity (`page`, `figure`,
`element`, `region`, or `chunk`). A page ID may be written as
`page:<document-id>:<one-based-page>`. Specific region questions should label
the child tile/region ID, not just its page or parent figure. Relevance grades
are positive integers; use larger grades for more useful evidence. Empty lists
are permitted to represent a query with no known relevant evidence and score
zero while remaining in denominators.

Keep expected facts, numeric values and citations in `expected_answer`,
`answer_evaluation`, or `extraction_evaluation`; never include them in query
text or the content visible to the retriever. For real documents, manually
inspect and cite the relevant source page/element in the annotation. Prefer
document-level development/test splits. The existing [answer fixtures](answer-fixtures.json)
are controlled mocked strings for evaluator checks, not real answers or model
output. No manually labelled real-document cases are currently included.

Run a labelled retrieval comparison only when the relevant document is already
ingested and its indexes are available. Text embeddings use the existing remote
embedding model and require explicit `--allow-remote-text-embeddings`; visual
evaluation runs Hugging Face in offline mode and reports missing cached weights
or indexes as failures/no-results. The runner does not download models or
start services.
