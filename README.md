# PaperLens

PaperLens extracts typed elements from PDFs, indexes searchable representations
in Qdrant, and links retrieved results to their original content in a local
document store. It supports text, headings, tables, extracted figures, optional
full-page image indexing, and optional OCR.

## Ingestion behavior

- **Text and headings:** Unstructured parses the PDF. Its `page` metadata is
  retained as a one-based page number.
- **Tables:** `text_as_html`, when supplied by Unstructured, is retained as
  `metadata.text_as_html` and used as the table's searchable content. HTML
  extraction is not guaranteed in Unstructured's `fast` strategy.
- **Figures:** When Poppler is available, Unstructured runs with `hi_res` and
  writes extracted figures into the document-scoped asset directory. Otherwise
  PaperLens uses `fast` for text extraction; full-page PDFium renders remain
  available as visual evidence.
- **Full pages:** Every page is rendered with `pypdfium2` at the configured
  scale. These images are saved as assets and linked from extracted elements
  and a document manifest; rendering by itself does not add pages to the vector
  index or call a vision model. Pass `--index-page-images` to index a page
  image only when that page has no extracted figure.
- **OCR:** Pass `--ocr` to OCR only pages without non-empty extracted text.
  OCR uses the Tesseract executable and records its text, page number, bounding
  box, and pixel coordinate convention. If Tesseract is unavailable, PaperLens
  reports that explicitly. OCR is not a requirement for ordinary text
  extraction.
- **Visual child chunks:** Pass `--chunk-visuals` to create deterministic,
  overlapping spatial crops for large image elements. The default tile size is
  1024 by 1024 pixels, overlap is 128 pixels, and splitting also occurs when
  either source dimension exceeds 1024 pixels. Child records and crop assets
  are stored in the document manifest and document-scoped asset directory;
  they are not added to Qdrant. The original image remains the indexed and
  retrieved parent. Page renders are eligible only when `--index-page-images`
  is also requested. These are spatial tiles, not semantic or query-aware
  chunks; they do not identify chart regions or label visual content.

### Metadata and assets

The existing `uuid` element key and legacy `metadata.page` field are preserved.
New elements also carry `metadata.element_id`, `document_id`, `source_file`,
`source_path`, `element_type`, and a SHA-256 `content_hash`. Unstructured
coordinates are retained unmodified in `metadata.coordinates`; a validated
`bbox` is additionally provided as `[left, top, right, bottom]` in that same
coordinate system, with `page_width`, `page_height`, and available coordinate
system attributes. When coordinates are absent, the page-render association
supplies pixel dimensions and top-left pixel coordinates; PDF point dimensions
are kept separately as `pdf_page_width` and `pdf_page_height`. No page number,
bounding box, coordinate origin, or unit is invented for Unstructured
coordinates when the parser does not supply it. OCR bounding boxes use
top-left origin and pixels.

Assets are kept under `DOC_STORE_PATH` (default `.\doc_store`) in
`assets\<document-key>\pages` and `assets\<document-key>\extracted`.
Extracted images are named by content hash within their document directory to
reuse identical assets. Image references stored under this root are relative
to it and resolved by PaperLens when used. The
`documents\<document-key>\manifest.json` file records the source path/hash and
page-render mapping. Raw element JSON files remain at the storage root, keyed
by their UUID, for compatibility with the existing retriever.

Generated document IDs are stable for the same source path and PDF contents.
Element and Qdrant point IDs are deterministic for repeated ingestion of the
same document identity. Re-ingestion overwrites corresponding raw element
files, page renders, and the manifest; it does not remove unrelated assets or
old Qdrant points created before deterministic point IDs were introduced.
Older Qdrant records are not automatically migrated or deleted. If a clean
re-index is required, back up data and manage the collection explicitly.

## Windows setup

Use Python 3.12 in PowerShell from the project directory:

```powershell
if (Test-Path .\.venv\Scripts\python.exe) {
    .\.venv\Scripts\python.exe --version
} else {
    py -3.12 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in a local `.env` file (copy `.env.example` only if `.env`
does not already exist). Ingestion indexing and querying make OpenAI API calls.
The automated unit tests mock those calls and need no API key.

Qdrant can be run locally with Docker Compose:

```powershell
docker compose up -d qdrant
```

Poppler is optional: without `pdftoppm` and `pdfinfo`, the partitioner uses
Unstructured's `fast` strategy and warns that table HTML and extracted figure
assets may be unavailable. To use its `hi_res` strategy, install Poppler and
open a new PowerShell session:

```powershell
winget install --id oschwartz10612.Poppler --exact
Get-Command pdftoppm, pdfinfo
```

OCR is separately optional and requires the Tesseract executable, not just a
Python package:

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
Get-Command tesseract
```

If an executable is installed outside `PATH`, add its actual `bin` directory
for the current PowerShell session, then verify it with `Get-Command`. PaperLens
does not install system executables or change the system execution policy.

## CLI

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py ingest .\path\to\document.pdf
.\.venv\Scripts\python.exe main.py ingest .\path\to\scan.pdf --ocr
.\.venv\Scripts\python.exe main.py ingest .\path\to\report.pdf --index-page-images
.\.venv\Scripts\python.exe main.py query "What does the revenue chart show?"
.\.venv\Scripts\python.exe main.py query "Summarize the report" --document-id "YOUR_DOCUMENT_ID"
```

`--render-scale` controls page image resolution and defaults to `1.5`.
`--ocr-language` selects the Tesseract language and defaults to `eng`. Use
`--chunk-visuals` to opt into image tiling, optionally adjusting it with
`--tile-width`, `--tile-height`, `--tile-overlap`, and `--min-image-size`.
For example:

```powershell
.\.venv\Scripts\python.exe main.py ingest .\path\to\report.pdf --chunk-visuals
```

Use `--help` for all supported options. The separate
`evaluation/benchmark.py` script is not a CLI subcommand.

## Query routing and page evidence

Query intent is planned deterministically from the wording; routing does not
make an LLM call. General questions continue through the existing Qdrant
similarity search and raw-element UUID lookup. Visual questions (figures,
tables, charts, diagrams, layouts, and similar terms) use that same semantic
path to find indexed descriptions and original elements.

Explicit one-based page references take a separate exact-evidence path. For
example:

```powershell
.\.venv\Scripts\python.exe main.py query "What is shown on page 4?" --document-id "YOUR_DOCUMENT_ID"
.\.venv\Scripts\python.exe main.py query "Compare the layouts on pages 1 and 3" --document-id "YOUR_DOCUMENT_ID"
```

PaperLens reads the selected document's existing provenance manifest and
loads the rendered image recorded for each requested page. Comparison pages
are passed to generation in the order requested and labelled by document and
page. The lookup also includes original stored elements whose `document_id`
and one-based `metadata.page` match. It does not ask Qdrant to choose pages or
index a page image merely to answer a direct page question. If a page is
outside the manifest's page count, its render is missing, or multiple
manifests make an unselected request ambiguous, the query fails with an
actionable message instead of substituting another page. Use `--document-id`
when more than one PDF has been ingested.

Semantic visual retrieval can only find content represented by existing
indexed text, captions, summaries, OCR, and figure records. It does not search
Phase 3 child tiles or identify a requested region unless the available
evidence itself makes that region clear. Numerical chart questions may send
the retrieved chart image to the vision-capable generator, but PaperLens does
not digitize chart data or guarantee exact values; estimates must be treated
as visual interpretation, not structured extraction.

### Direct visual retrieval

Phase 5 adds a separate CLIP image/text embedding path and a separate Qdrant
collection; it does not change the existing text embedding collection. After
ingestion, build the visual index explicitly:

```powershell
.\.venv\Scripts\python.exe main.py index-visuals --document-id "YOUR_DOCUMENT_ID"
```

This indexes persisted extracted figures, opted-in page images, and any child
tiles saved by `--chunk-visuals`. It does not re-parse the PDF or call OpenAI.
The first run downloads the configured Hugging Face model
(`VISUAL_EMBED_MODEL`, default `openai/clip-vit-base-patch32`) and loads it
locally; plan for model download, disk, memory, and CPU time. CPU is the default.
Set `VISUAL_DEVICE=cuda` only when a compatible CUDA-enabled PyTorch
installation and GPU are available. The vector collection name is
`VISUAL_COLLECTION_NAME` (default `rag_visuals`). It is independent from
`COLLECTION_NAME`; a dimension mismatch fails without changing the collection.

Visual questions use the existing text retrieval and the direct visual index,
then fuse rankings with reciprocal rank fusion (RRF). Text evidence and visual
evidence have separate limits; `VISUAL_EVIDENCE_LIMIT` controls the visual
limit (default `4`). Tile evidence is labelled as a crop relative to its
parent image, and valid parent-image context is also supplied. The generator
receives the selected image evidence alongside text. Explicit page-reference
questions still use the exact manifest/page path and bypass semantic and visual
search. General text questions do not load the visual model. An absent or
empty visual index falls back to text retrieval with a diagnostic.

The visual retrieval metric helper is deterministic and does not need Qdrant,
model weights, or API credentials. Given labelled rankings, it computes macro
Recall@k and MRR:

```python
from evaluation.retrieval_metrics import evaluate_retrieval

scores = evaluate_retrieval(
    rankings=[["figure-2", "table-1", "figure-1"]],
    relevant_ids=[{"figure-1", "figure-2"}],
    k_values=(1, 3, 5),
)
```

The automated metric tests use constructed rankings to verify the calculation;
they are not a retrieval-quality benchmark and do not demonstrate gains on
real documents. Evaluate representative labelled queries and report those
results separately. The existing `evaluation/benchmark.py` remains the
answer-generation comparison utility and may make external model calls.

## Tests and verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Renderer tests create synthetic text/vector PDFs and use the installed PDFium
library. Parser element tests use deterministic Unstructured element fixtures;
OCR tests mock the Tesseract process. The unit suite does not require Poppler,
Tesseract, Qdrant, or OpenAI credentials. It does not establish that every
scanned PDF, complex layout, or chart can be extracted accurately; validate
those cases against representative real documents and the corresponding
external tools before relying on them.
