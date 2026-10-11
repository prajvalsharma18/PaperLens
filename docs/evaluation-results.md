# Evaluation results

Evaluation runs write timestamped JSON reports and Markdown summaries under
\`evaluation/reports/\`. See [the dataset annotation guide](../evaluation/datasets/README.md)
and [the README evaluation section](../README.md#reproducible-evaluation-runner)
for the runner and metric definitions.

The repository currently includes controlled answer strings and generated
table/chart fixtures. They exercise metric and runner behavior; they are not
manually labelled real-document ground truth. No local PDFs were found in the
repository, so real-document retrieval, extraction accuracy, and answer
grounding remain unevaluated. Do not interpret synthetic scores as general
accuracy.

The previous \`evaluation/benchmark.py\` RAGAS workflow invoked model generation
and RAGAS against a single already-retrieved context. That did not compare
retrievers. RAGAS remains available as a separate optional evaluator and is not
run by the offline benchmark.
