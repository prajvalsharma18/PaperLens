"""Render PDF pages to document-scoped image assets using PDFium."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from ingestion.assets import document_asset_key, storage_root


class PageRenderError(RuntimeError):
    """Raised when a PDF cannot be rendered to page assets."""


def render_pdf_pages(
    pdf_path: str,
    document_id: str,
    scale: float = 1.5,
    pages: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Render selected one-based pages and return paths relative to DOC_STORE_PATH."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Render scale must be a finite number greater than zero.")

    source = Path(pdf_path).expanduser().resolve()
    root = storage_root()
    page_dir = root / "assets" / document_asset_key(document_id) / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    document = None
    try:
        document = pdfium.PdfDocument(str(source))
        page_count = len(document)
        if page_count < 1:
            raise PageRenderError(f"The PDF contains no renderable pages: {source}")
        if pages is not None and any(page < 1 or page > page_count for page in pages):
            raise PageRenderError(
                f"Requested page is outside the PDF's 1-{page_count} page range."
            )

        page_numbers = sorted(pages) if pages is not None else range(1, page_count + 1)
        rendered: list[dict[str, Any]] = []
        for page_number in page_numbers:
            page = document.get_page(page_number - 1)
            bitmap = None
            try:
                page_width = float(page.get_width())
                page_height = float(page.get_height())
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                target = page_dir / f"page-{page_number:05d}.png"
                image.save(target, format="PNG")
                relative_path = target.relative_to(root).as_posix()
                pixel_width, pixel_height = image.size
                image.close()
                rendered.append(
                    {
                        "page_number": page_number,
                        "image_path": relative_path,
                        "pixel_width": pixel_width,
                        "pixel_height": pixel_height,
                        "page_width": page_width,
                        "page_height": page_height,
                        "coordinate_system": {
                            "origin": "top-left",
                            "units": "pixels",
                        },
                        "rendering": {"library": "pypdfium2", "scale": scale},
                    }
                )
            finally:
                if bitmap is not None:
                    bitmap.close()
                page.close()
        return rendered
    except PageRenderError:
        raise
    except Exception as exc:
        raise PageRenderError(f"Could not render PDF pages from {source}: {exc}") from exc
    finally:
        if document is not None:
            document.close()
