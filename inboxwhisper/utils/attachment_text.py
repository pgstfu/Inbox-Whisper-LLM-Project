import fitz  # PyMuPDF for PDF
import docx
import pandas as pd
from pathlib import Path

def extract_pdf_text(path):
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text()
    except Exception:
        return ""
    return text

def extract_docx_text(path):
    text = ""
    try:
        d = docx.Document(path)
        for p in d.paragraphs:
            text += p.text + "\n"
    except Exception:
        return ""
    return text

def extract_excel_text(path):
    text = ""
    try:
        # Try modern Excel
        df = pd.read_excel(path, sheet_name=None)
        for sheet_name, sheet_df in df.items():
            text += f"\n[Sheet: {sheet_name}]\n"
            text += sheet_df.to_string()  # convert sheet to readable text
            text += "\n"
    except Exception:
        return ""
    return text

def extract_attachment_text(file_path):
    file_path = str(file_path)
    p = Path(file_path)

    ext = p.suffix.lower()

    if ext == ".pdf":
        return extract_pdf_text(file_path)

    if ext in [".doc", ".docx"]:
        return extract_docx_text(file_path)

    if ext in [".xls", ".xlsx"]:
        return extract_excel_text(file_path)

    # Other unsupported types → return ""
    return ""
