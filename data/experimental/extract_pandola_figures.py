"""Extract figure pages from Pandola et al. (2015) PDF as high-res PNGs.

These images can be loaded directly into WebPlotDigitizer
(https://automeris.io/wpd/) for digitization of Dance et al. data points.

Usage:
    python data/experimental/extract_pandola_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger(__name__)

PDF_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "references"
    / "Pandola_2015_Geant4_brems_validation.pdf"
)
OUT_DIR = Path(__file__).resolve().parent / "dance_1968"

# Page numbers (0-indexed) containing the figures
# Fig 1: Al 2MeV 9 angles (page 5)
# Fig 2: Fe 0.5/1.0/2.8 MeV (page 6)
# Fig 3: Al+Fe single-differential (page 7)
FIGURE_PAGES = {
    "fig1_al_2mev_angles": 5,
    "fig2_fe_multi_energy": 6,
    "fig3_al_fe_integrated": 7,
}

DPI = 300


def main() -> None:
    """Extract figure pages as PNGs."""
    if not PDF_PATH.exists():
        log.error("PDF not found: %s", PDF_PATH)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(PDF_PATH))

    for name, page_num in FIGURE_PAGES.items():
        page = doc[page_num]
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        out_path = OUT_DIR / f"{name}.png"
        pix.save(str(out_path))
        print(f"Saved {out_path.name} ({pix.width}x{pix.height} @ {DPI} DPI)")  # noqa: T201

    doc.close()
    print(f"\nFigures saved to {OUT_DIR}/")  # noqa: T201
    print("Open https://automeris.io/wpd/ and load these images to digitize.")  # noqa: T201


if __name__ == "__main__":
    main()
