import streamlit as st
import summarizer  # Your new summarizer.py file
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📚",
    layout="wide"
)

# --- State Management ---
# Use session state to store extracted text
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

# --- UI Layout ---
st.title("📄 PDF Document Summarizer")
st.markdown("Upload a PDF, choose your summarization method, and get a quick summary.")

# --- Sidebar for Upload and Options ---
with st.sidebar:
    st.header("1. Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    st.header("2. Choose Method")
    st.markdown("""
    This is where the "decision" is made. You, the user, choose the trade-off.
    """)
    
    # The "decision" is a user-selected radio button
    summary_method = st.radio(
        "Select summarization type:",
        (
            'Extractive (Fast & Factual)',
            'Abstractive (Fluent & Human-like)'
        ),
        help="Extractive pulls key sentences directly. Abstractive rewrites the text in its own words."
    )
    
    # Map radio button to the method name in summarizer.py
    if summary_method.startswith('Extractive'):
        method_key = 'extractive'
    else:
        method_key = 'abstractive'

    # We use a fixed 20% summarization
    st.header("3. Summary Options")
    st.info("The summary will be approximately 20% of the original text length.")
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        if method_key == 'extractive':
            diversity = st.slider("Diversity", 0.0, 1.0, 0.7,
                                help="Higher values favor relevance over diversity")
            model_name = st.selectbox("Embedding model", 
                                    ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                                    help="Model for computing sentence embeddings")
        else:  # abstractive
            model_name = st.selectbox("Summarization model",
                                    ["facebook/bart-large-cnn", 
                                     "google/pegasus-xsum",
                                     "sshleifer/distilbart-cnn-12-6"],
                                    help="Choose between quality and speed")
            diversity = 0.7  # not used for abstractive

    run_button = st.button("Generate Summary", type="primary", use_container_width=True)

# --- Main Content Area ---
col1, col2 = st.columns(2)

with col1:
    st.header("Original Text")
    if uploaded_file:
        # When a new file is uploaded, extract text and store it
        if st.session_state.get('last_uploaded_name') != uploaded_file.name:
            st.session_state.last_uploaded_name = uploaded_file.name
            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                # Convert to BytesIO to pass to the function
                pdf_bytes = BytesIO(uploaded_file.getvalue())
                st.session_state.extracted_text = summarizer.extract_text_from_pdf(pdf_bytes)
        
    text_container = st.container(height=600, border=True)
    text_container.markdown(st.session_state.extracted_text)

with col2:
    st.header("Generated Summary")
    summary_container = st.container(height=600, border=True)
    
    if run_button:
        if not st.session_state.extracted_text:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner(f"Running {method_key} summarization... (This may take a moment)"):
                final_summary, sentence_counts = summarizer.run_summarization(
                    text=st.session_state.extracted_text,
                    method=method_key
                )
                # Display sentence counts
                st.info(f"Original: {sentence_counts['original']} sentences → Summary: {sentence_counts['summary']} sentences")
                summary_container.markdown(final_summary)
