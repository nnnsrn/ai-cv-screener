import os
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langdetect import detect
from deep_translator import GoogleTranslator

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    Falls back to PyTesseract OCR if the extracted text is empty or suspiciously short (< 50 chars).
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted raw text.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"PyMuPDF text extraction notice ({pdf_path}): {e}")

    # OCR Fallback if text is suspiciously short (< 50 chars)
    if len(text.strip()) < 50:
        print(f"Extracted text under threshold (< 50 chars) for {pdf_path}. Falling back to PyTesseract OCR...")
        ocr_text_lines = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_page = pytesseract.image_to_string(img)
                if ocr_page.strip():
                    ocr_text_lines.append(ocr_page.strip())
            if ocr_text_lines:
                text = "\n".join(ocr_text_lines)
        except Exception as e:
            print(f"PyTesseract OCR execution failed ({pdf_path}): {e}")

    return text.strip()


def extract_and_translate(pdf_path_or_text: str) -> str:
    """
    Extracts text from a PDF file (or receives raw text input directly),
    detects language using langdetect, and translates to English if necessary via deep-translator.
    
    Parameters:
        pdf_path_or_text (str): Path to a PDF file or raw text string.
        
    Returns:
        str: English text string.
    """
    if not pdf_path_or_text or not pdf_path_or_text.strip():
        return ""

    if os.path.exists(pdf_path_or_text) and pdf_path_or_text.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(pdf_path_or_text)
    else:
        # Treat input as raw text if not an existing PDF file
        raw_text = pdf_path_or_text.strip()

    if not raw_text:
        return ""

    # Detect language
    try:
        lang = detect(raw_text)
    except Exception:
        lang = "en"

    # Translate non-English text to English
    if lang != "en":
        try:
            translator = GoogleTranslator(source="auto", target="en")
            max_chunk_len = 4000
            chunks = [raw_text[i:i + max_chunk_len] for i in range(0, len(raw_text), max_chunk_len)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks if chunk.strip()]
            return "\n".join(translated_chunks)
        except Exception as e:
            print(f"Translation encountered an issue ({e}); returning raw extracted text.")
            return raw_text

    return raw_text
