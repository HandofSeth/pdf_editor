# PDF Editor

A desktop PDF editor built with Python, PyQt5 and PyMuPDF. Click on any text in a PDF to edit it in place, with font preservation and support for scanned/OCR documents.

## Features

- **Click-to-edit text** — click any text span to modify it via a dialog
- **Font preservation** — detects the original font and maps it to the closest Base14 equivalent (serif, sans-serif, monospace) with bold/italic support
- **Scanned PDF support** — for image-based pages, erases text from the image using OpenCV inpainting and reinserts the new text
- **Zoom controls** — zoom in/out and fit-to-width
- **Page navigation** — previous/next and direct page number input
- **Save / Save As** — incremental save or export to a new file

## Screenshot

<!-- Add a screenshot here -->

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone https://github.com/<your-username>/pdf-editor.git
cd pdf-editor
pip install -r requirements.txt
```

## Usage

```bash
python pdf_editor.py
```

1. Click **Open** (or `Ctrl+O`) to load a PDF
2. Click on any text in the document
3. Edit the text in the dialog and click **OK**
4. Save with `Ctrl+S` or use **Save As** (`Ctrl+Shift+S`)

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open PDF |
| `Ctrl+S` | Save |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+=` | Zoom in |
| `Ctrl+-` | Zoom out |

## License

MIT
