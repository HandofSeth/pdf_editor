#!/usr/bin/env python3
"""PDF Editor with PyQt5 and PyMuPDF - preserves original fonts when editing text."""

import io
import re
import sys
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image as PILImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QScrollArea, QToolBar,
    QAction, QFileDialog, QMessageBox, QDialog, QVBoxLayout,
    QHBoxLayout, QTextEdit, QPushButton, QSpinBox, QStatusBar,
    QWidget, QLineEdit, QFormLayout, QDoubleSpinBox,
)
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QPoint


# Mapping of common font family names (lowercase) to Base14 font sets.
# Each value is (regular, bold, italic, bold-italic).
_FONT_FAMILY_MAP = {
    # Sans-serif families
    "arial": ("helv", "hebo", "heit", "hebi"),
    "arialmt": ("helv", "hebo", "heit", "hebi"),
    "helvetica": ("helv", "hebo", "heit", "hebi"),
    "calibri": ("helv", "hebo", "heit", "hebi"),
    "verdana": ("helv", "hebo", "heit", "hebi"),
    "tahoma": ("helv", "hebo", "heit", "hebi"),
    "trebuchet": ("helv", "hebo", "heit", "hebi"),
    "trebuchetms": ("helv", "hebo", "heit", "hebi"),
    "segoeui": ("helv", "hebo", "heit", "hebi"),
    "opensans": ("helv", "hebo", "heit", "hebi"),
    "roboto": ("helv", "hebo", "heit", "hebi"),
    "lato": ("helv", "hebo", "heit", "hebi"),
    # Serif families
    "times": ("tiro", "tibo", "tiit", "tibi"),
    "timesnewroman": ("tiro", "tibo", "tiit", "tibi"),
    "timesnewromanpsmt": ("tiro", "tibo", "tiit", "tibi"),
    "georgia": ("tiro", "tibo", "tiit", "tibi"),
    "garamond": ("tiro", "tibo", "tiit", "tibi"),
    "palatino": ("tiro", "tibo", "tiit", "tibi"),
    "palatinolinotype": ("tiro", "tibo", "tiit", "tibi"),
    "bookman": ("tiro", "tibo", "tiit", "tibi"),
    "cambria": ("tiro", "tibo", "tiit", "tibi"),
    # Monospace families
    "courier": ("cour", "cobo", "coit", "cobi"),
    "couriernew": ("cour", "cobo", "coit", "cobi"),
    "consolas": ("cour", "cobo", "coit", "cobi"),
    "lucidaconsole": ("cour", "cobo", "coit", "cobi"),
    "menlo": ("cour", "cobo", "coit", "cobi"),
    "monaco": ("cour", "cobo", "coit", "cobi"),
    "sourcecodepro": ("cour", "cobo", "coit", "cobi"),
}


def _resolve_font(page, span):
    """Resolve the best font to use for insert_text() given a span.

    Returns (fontname_to_use, display_info) where display_info is a
    human-readable string describing the mapping for the dialog.
    """
    raw_name = span.get("font", "")
    flags = span.get("flags", 0)
    font_size = span.get("size", 12)

    # Decode flags
    is_bold = bool(flags & 0x10)
    is_italic = bool(flags & 0x02)
    is_serif = bool(flags & 0x04)
    is_mono = bool(flags & 0x08)

    # Clean the font name: remove subset prefix like "ABCDEF+"
    clean_name = raw_name.split("+")[-1]

    # --- Step 1: Try to reuse an embedded font from the page ---
    try:
        page_fonts = page.get_fonts()  # list of (xref, ext, type, basefont, name, encoding)
        for _xref, _ext, _type, basefont, ref_name, _enc in page_fonts:
            if clean_name and clean_name in basefont and ref_name:
                # Found the embedded font; use its reference name
                return ref_name, f"{raw_name} -> embedded '{ref_name}'"
    except Exception:
        pass

    # --- Step 2: Map by font family name + flags ---
    # Normalize: strip style suffixes and non-alpha chars for lookup
    lookup = re.sub(r"[-_\s]", "", clean_name).lower()
    # Remove common style suffixes so "ArialMT-BoldItalic" -> "arialmt"
    for suffix in ("bolditalic", "boldit", "bold", "italic", "it",
                    "regular", "roman", "medium", "light", "semibold",
                    "black", "thin", "condensed", "mt", "ps", "psmt"):
        if lookup.endswith(suffix) and len(lookup) > len(suffix):
            lookup = lookup[: -len(suffix)]
            break

    family = _FONT_FAMILY_MAP.get(lookup)

    # If no direct hit, try partial matching
    if family is None:
        for key, val in _FONT_FAMILY_MAP.items():
            if key in lookup or lookup in key:
                family = val
                break

    # If still no match, fall back based on flags (serif/mono heuristics)
    if family is None:
        if is_mono:
            family = ("cour", "cobo", "coit", "cobi")
        elif is_serif:
            family = ("tiro", "tibo", "tiit", "tibi")
        else:
            family = ("helv", "hebo", "heit", "hebi")

    # Pick the right variant: (regular, bold, italic, bold-italic)
    if is_bold and is_italic:
        chosen = family[3]
    elif is_bold:
        chosen = family[1]
    elif is_italic:
        chosen = family[2]
    else:
        chosen = family[0]

    return chosen, f"{raw_name} -> Base14 '{chosen}'"


class TextEditDialog(QDialog):
    """Dialog for editing a text span while showing font info."""

    def __init__(self, text, font_name, font_size, color_rgb, resolved_font_info="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Text")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        info_layout = QFormLayout()
        font_label = QLineEdit(font_name)
        font_label.setReadOnly(True)
        info_layout.addRow("Original font:", font_label)

        if resolved_font_info:
            resolved_label = QLineEdit(resolved_font_info)
            resolved_label.setReadOnly(True)
            info_layout.addRow("Resolved as:", resolved_label)

        size_label = QLineEdit(f"{font_size:.1f}")
        size_label.setReadOnly(True)
        info_layout.addRow("Size:", size_label)

        r, g, b = color_rgb
        color_label = QLineEdit(f"RGB({r}, {g}, {b})")
        color_label.setReadOnly(True)
        color_label.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: {'white' if r+g+b < 384 else 'black'};"
        )
        info_layout.addRow("Color:", color_label)
        layout.addLayout(info_layout)

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setMinimumHeight(100)
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def get_text(self):
        return self.text_edit.toPlainText()


class PDFLabel(QLabel):
    """Label that displays a PDF page and handles click-to-select-text."""

    def __init__(self, parent_editor):
        super().__init__()
        self.editor = parent_editor
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(QCursor(Qt.CrossCursor))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.editor.doc:
            self.editor.handle_click(event.pos())


class PDFEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.doc = None
        self.file_path = None
        self.current_page = 0
        self.zoom = 1.5
        self.page_text_data = None  # cached text dict for current page

        self.setWindowTitle("PDF Editor")
        self.resize(900, 700)
        self._build_ui()

    def _build_ui(self):
        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_act = QAction("Open", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.open_file)
        toolbar.addAction(open_act)

        save_act = QAction("Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self.save_file)
        toolbar.addAction(save_act)

        save_as_act = QAction("Save As", self)
        save_as_act.setShortcut("Ctrl+Shift+S")
        save_as_act.triggered.connect(self.save_file_as)
        toolbar.addAction(save_as_act)

        toolbar.addSeparator()

        prev_act = QAction("< Prev", self)
        prev_act.triggered.connect(self.prev_page)
        toolbar.addAction(prev_act)

        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setPrefix("Page ")
        self.page_spin.valueChanged.connect(self._on_page_spin)
        toolbar.addWidget(self.page_spin)

        self.page_count_label = QLabel(" / 0")
        toolbar.addWidget(self.page_count_label)

        next_act = QAction("Next >", self)
        next_act.triggered.connect(self.next_page)
        toolbar.addAction(next_act)

        toolbar.addSeparator()

        zoom_in_act = QAction("Zoom +", self)
        zoom_in_act.setShortcut("Ctrl+=")
        zoom_in_act.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_act)

        zoom_out_act = QAction("Zoom -", self)
        zoom_out_act.setShortcut("Ctrl+-")
        zoom_out_act.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_act)

        fit_act = QAction("Fit", self)
        fit_act.triggered.connect(self.zoom_fit)
        toolbar.addAction(fit_act)

        self.zoom_label = QLabel(f" {int(self.zoom*100)}%")
        toolbar.addWidget(self.zoom_label)

        # Central widget
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        self.pdf_label = PDFLabel(self)
        self.scroll_area.setWidget(self.pdf_label)
        self.setCentralWidget(self.scroll_area)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Open a PDF to begin.")

    # --- File operations ---

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not path:
            return
        self._load_doc(path)

    def _load_doc(self, path):
        try:
            self.doc = fitz.open(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open file:\n{e}")
            return
        self.file_path = path
        self.current_page = 0
        self.page_spin.setMaximum(len(self.doc))
        self.page_count_label.setText(f" / {len(self.doc)}")
        self.setWindowTitle(f"PDF Editor - {path}")
        self.render_page()

    def save_file(self):
        if not self.doc:
            return
        if not self.file_path:
            self.save_file_as()
            return
        try:
            self.doc.save(self.file_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
            self.status.showMessage(f"Saved to {self.file_path}")
        except Exception:
            # incremental save may fail on new/modified structure; fall back to saveAs-style
            self.save_file_as()

    def save_file_as(self):
        if not self.doc:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF As", "", "PDF Files (*.pdf)")
        if not path:
            return
        try:
            self.doc.save(path)
            self.status.showMessage(f"Saved to {path}")
            # Reopen to allow further incremental saves
            self._load_doc(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot save:\n{e}")

    # --- Navigation ---

    def prev_page(self):
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.render_page()

    def next_page(self):
        if self.doc and self.current_page < len(self.doc) - 1:
            self.current_page += 1
            self.render_page()

    def _on_page_spin(self, val):
        if self.doc:
            self.current_page = val - 1
            self.render_page(update_spin=False)

    # --- Zoom ---

    def zoom_in(self):
        self.zoom = min(self.zoom + 0.25, 5.0)
        self.render_page()

    def zoom_out(self):
        self.zoom = max(self.zoom - 0.25, 0.25)
        self.render_page()

    def zoom_fit(self):
        if not self.doc:
            return
        page = self.doc[self.current_page]
        vp_width = self.scroll_area.viewport().width() - 20
        self.zoom = vp_width / page.rect.width
        self.render_page()

    # --- Rendering ---

    def render_page(self, update_spin=True):
        if not self.doc:
            return
        page = self.doc[self.current_page]
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        self.pdf_label.setPixmap(QPixmap.fromImage(img))
        self.pdf_label.adjustSize()

        if update_spin:
            self.page_spin.blockSignals(True)
            self.page_spin.setValue(self.current_page + 1)
            self.page_spin.blockSignals(False)

        self.zoom_label.setText(f" {int(self.zoom*100)}%")
        self.status.showMessage(f"Page {self.current_page+1} / {len(self.doc)}")

        # Cache text data for click detection
        self.page_text_data = page.get_text("dict")

    # --- Click / edit ---

    def handle_click(self, widget_pos: QPoint):
        """Convert widget click position to PDF coords and find the span."""
        # The label may be centered inside the scroll area; account for offset
        label_rect = self.pdf_label.pixmap().rect()
        # Label content offset (centered)
        ox = (self.pdf_label.width() - label_rect.width()) / 2
        oy = (self.pdf_label.height() - label_rect.height()) / 2
        px = (widget_pos.x() - ox) / self.zoom
        py = (widget_pos.y() - oy) / self.zoom

        if px < 0 or py < 0:
            return

        span = self._find_span(px, py)
        if span is None:
            self.status.showMessage("No text found at click position.")
            return

        self._edit_span(span)

    def _find_span(self, x, y):
        """Find the text span at PDF coordinates (x, y)."""
        if not self.page_text_data:
            return None
        for block in self.page_text_data.get("blocks", []):
            if block.get("type") != 0:  # text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    r = span["bbox"]  # (x0, y0, x1, y1)
                    if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                        return span
        return None

    def _get_page_image_info(self, page):
        """If the page is image-based, return (xref, image_rect) of the main image.

        Returns None if not an image-based page.
        """
        page_area = page.rect.width * page.rect.height
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                img_rects = page.get_image_rects(xref)
                for r in img_rects:
                    if r.width * r.height > page_area * 0.5:
                        return xref, r
            except Exception:
                continue
        return None

    def _erase_on_image(self, page, xref, img_rect, bbox):
        """Erase text from the embedded image using OpenCV inpainting.

        This reconstructs the background naturally from surrounding pixels,
        preserving textures, gradients and patterns (like Photoshop Content-Aware Fill).
        """
        # Extract the image
        img_data = fitz.Pixmap(self.doc, xref)
        if img_data.alpha:
            img_data = fitz.Pixmap(fitz.csRGB, img_data)
        img_bytes = img_data.tobytes("png")
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = pil_img.size

        # Map PDF coordinates to image pixel coordinates
        scale_x = img_w / img_rect.width
        scale_y = img_h / img_rect.height
        px0 = int((bbox.x0 - img_rect.x0) * scale_x)
        py0 = int((bbox.y0 - img_rect.y0) * scale_y)
        px1 = int((bbox.x1 - img_rect.x0) * scale_x)
        py1 = int((bbox.y1 - img_rect.y0) * scale_y)

        # Add a small margin to cover anti-aliasing around text
        margin = 3
        px0 = max(0, px0 - margin)
        py0 = max(0, py0 - margin)
        px1 = min(img_w, px1 + margin)
        py1 = min(img_h, py1 + margin)

        if px1 <= px0 or py1 <= py0:
            return

        # Convert to OpenCV format (BGR)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Create an inpainting mask: white = area to reconstruct
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[py0:py1, px0:px1] = 255

        # Use the TELEA inpainting algorithm to reconstruct the background
        # The radius controls how far from the mask border to sample pixels
        inpaint_radius = max(5, int((py1 - py0) * 0.4))
        result = cv2.inpaint(cv_img, mask, inpaint_radius, cv2.INPAINT_TELEA)

        # Convert back to PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_result = PILImage.fromarray(result_rgb)

        # Replace the image in the PDF
        out_buf = io.BytesIO()
        pil_result.save(out_buf, format="PNG")
        out_buf.seek(0)
        new_pix = fitz.Pixmap(out_buf.read())
        img_data2 = fitz.Pixmap(fitz.csRGB, new_pix)
        page.replace_image(xref, pixmap=img_data2)

    def _edit_span(self, span):
        old_text = span["text"]
        font_name = span["font"]
        font_size = span["size"]
        # color is an int (sRGB packed); convert to (R, G, B)
        c = span["color"]
        color_rgb = ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)

        page = self.doc[self.current_page]
        resolved_fontname, display_info = _resolve_font(page, span)
        img_info = self._get_page_image_info(page)

        dlg = TextEditDialog(old_text, font_name, font_size, color_rgb, display_info, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        new_text = dlg.get_text()
        if new_text == old_text:
            return

        bbox = fitz.Rect(span["bbox"])

        if img_info:
            # Image-based PDF (scanned/OCR): modify the image pixels directly
            xref, img_rect = img_info

            # 1. Erase the old text on the actual image
            try:
                self._erase_on_image(page, xref, img_rect, bbox)
            except Exception as e:
                self.status.showMessage(f"Image edit failed: {e}")
                return

            # 2. Remove the OCR text layer for this area
            page.add_redact_annot(bbox, fill=False)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            # 3. Insert the new text on top of the image
            # For OCR PDFs, text color is often black regardless of span color
            text_color = (0.0, 0.0, 0.0)
            try:
                page.insert_text(
                    fitz.Point(bbox.x0, bbox.y0 + font_size),
                    new_text,
                    fontname=resolved_fontname,
                    fontsize=font_size,
                    color=text_color,
                )
            except Exception:
                page.insert_text(
                    fitz.Point(bbox.x0, bbox.y0 + font_size),
                    new_text,
                    fontname="helv",
                    fontsize=font_size,
                    color=text_color,
                )

            self.render_page()
            self.status.showMessage(f"Text updated (image PDF). Font: {display_info}")
        else:
            # Regular PDF: use redaction to remove text, preserve background
            page.add_redact_annot(bbox, fill=False)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            color_float = tuple(v / 255.0 for v in color_rgb)
            try:
                page.insert_text(
                    fitz.Point(bbox.x0, bbox.y0 + font_size),
                    new_text,
                    fontname=resolved_fontname,
                    fontsize=font_size,
                    color=color_float,
                )
            except Exception:
                page.insert_text(
                    fitz.Point(bbox.x0, bbox.y0 + font_size),
                    new_text,
                    fontname="helv",
                    fontsize=font_size,
                    color=color_float,
                )

            self.render_page()
            self.status.showMessage(f"Text updated. Font: {display_info}")


def main():
    app = QApplication(sys.argv)
    editor = PDFEditor()
    editor.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
