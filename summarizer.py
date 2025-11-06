import pdfplumber
import abstractive_summarizer
import extractive_summarizer
from io import BytesIO
from typing import Optional  # <-- Import Optional

def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    """
    Extracts all text from an uploaded PDF file.
    
    Args:
        pdf_file: A file-like object (e.g., from Streamlit's file_uploader).
    
    Returns:
        A string containing all extracted text.
    """
    all_text = ""
    # pdfplumber.open() can accept a file-like object directly
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n\n"
        return all_text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return f"Error: Could not extract text from PDF. The file may be corrupt or an image-based PDF. {e}"


def run_summarization(
    text: str,
    method: str,
    num_sentences: int = None,
    top_k_percent: float = 20.0
) -> tuple[str, dict]:
    """
    Acts as a router to call the correct summarization function
    based on the user's choice.

    Args:
        text: The source text to summarize.
        method: The method to use ('extractive' or 'abstractive').
        num_sentences (int): The target number of sentences.
        top_k_percent (Optional[float]): Target summary size as a percentage
                                          of the original text.

    Returns:
        The summarized text.
    """
    if not text.strip():
        return "Error: Cannot summarize empty text."

    try:
        # Count original sentences
        sentences = extractive_summarizer._sent_tokenize(text)
        original_count = len([s for s in sentences if s and len(s.strip()) > 2])
        
        if method == 'extractive':
            # Always use top_k_percent=20.0 for consistent summarization
            summary = extractive_summarizer.extractive_summarize(
                raw_text=text,
                top_k_percent=20.0
            )
        elif method == 'abstractive':
            summary = abstractive_summarizer.abstractive_summarize(
                raw_text=text,
                top_k_percent=20.0
            )
        else:
            return f"Error: Unknown summarization method '{method}'.", {"original": 0, "summary": 0}
        
        # Count summary sentences
        summary_sentences = extractive_summarizer._sent_tokenize(summary)
        summary_count = len([s for s in summary_sentences if s and len(s.strip()) > 2])
        
        return summary, {
            "original": original_count,
            "summary": summary_count
        }
    
    except Exception as e:
        # Catch errors from the underlying summarizer libraries
        return f"An error occurred during {method} summarization: {e}"

