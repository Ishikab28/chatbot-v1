import os
import re
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_TORCH"] = "1"

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import pandas as pd
import google.generativeai as genai 
from abc import ABC
import asyncio
from collections import Counter
from u1 import InsurancePromptTemplates, InsuranceResponseFormatter  # Import from utility file

# Set up event loop for Streamlit
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set your Gemini API key
GEMINI_API_KEY = "AIzaSyDCmrtox1j9cym8POSlgRkjRcExVjZR7Pg"

# Page config with better visuals
st.set_page_config(
    page_title="Insurance Copilot : Powered by Imarticus",
    layout="wide",
    page_icon="🤖"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f9ff;
        }
        .sidebar .sidebar-content {
            background-color: #e8f1ff;
        }
        .stTextInput input {
            background-color: #ffffff !important;
        }
        .stButton>button {
            background-color: #4a8cff;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #3a7cdf;
            color: white;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #2a5cad;
        }
        .chat-message {
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 12px;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .context-box {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
            background-color: #ffffff;
            border-left: 4px solid #4a8cff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .table-container {
            margin: 12px 0;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            overflow: hidden;
        }
        .fun-fact {
            background: #e8f4ff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------- BACKEND CLASSES ------------------------

class BaseAgent(ABC):
    def __init__(self, use_gemini: bool = True):
        self.use_gemini = use_gemini
        if self.use_gemini:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception as e:
                st.error(f"Failed to initialize Gemini: {str(e)}")
                raise

    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 20000) -> str:
        if self.use_gemini:
            return self._call_gemini(prompt, temperature, max_tokens)
        else:
            raise NotImplementedError("Only Gemini is supported")

    def _call_gemini(self, prompt: str, temperature: float = 0.7, max_tokens: int = 20000) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            return response.text
        except Exception as e:
            st.error(f"Error calling Gemini API: {str(e)}")
            return f"Error calling Gemini API: {str(e)}"

# ----------------------- HELPER FUNCTIONS ------------------------

def extract_headers_from_pdf(pdf_file):
    """Smart header extraction with advanced filtering"""
    headers = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            all_chars = []
            for page in pdf.pages:
                all_chars.extend(page.chars)
            
            if not all_chars:
                return []
            
            sizes = [char['size'] for char in all_chars]
            most_common_size = Counter(sizes).most_common(1)[0][0]
            font_size_threshold = most_common_size
            
            exclude_patterns = [
                r'\b(?:FDA|CI|HR|PFS|CTC|ml|mg|kg|%\d|\(\d|\[\d)\b',
                r'\d+\.\d+', r'\d+–\d+', r'\([^)]*\)', r'\[[^\]]*\]',
                r'\d+\s*(?:ml|mg|kg)', r'[A-Z]{3,}', r'\.\d{2}'
            ]

            for page in pdf.pages:
                chars = page.chars
                if not chars:
                    continue
                
                groups = []
                current_group = [chars[0]]
                y_pos = chars[0]['top']
                
                for char in chars[1:]:
                    if (char['fontname'] == current_group[-1]['fontname'] and 
                        char['size'] == current_group[-1]['size'] and
                        abs(char['top'] - y_pos) < 5):
                        current_group.append(char)
                    else:
                        groups.append(current_group)
                        current_group = [char]
                        y_pos = char['top']
                groups.append(current_group)
                
                for group in groups:
                    first_char = group[0]
                    is_bold = 'Bold' in first_char['fontname']
                    size = first_char['size']
                    
                    if size > font_size_threshold or is_bold:
                        text = ''.join([c['text'] for c in group]).strip()
                        
                        is_valid = (
                            5 <= len(text) <= 50 and
                            not any(re.search(p, text) for p in exclude_patterns) and
                            len(text.split()) >= 2 and
                            not any(c.isdigit() for c in text) and
                            text[0].isupper() and
                            not text.endswith(('.', ',', ';'))
                        )
                        if is_valid:
                            headers.append(text)
    
    except Exception as e:
        st.error(f"Header extraction failed: {str(e)}")
        return []
    
    clean_headers = []
    seen = set()
    for header in headers:
        h = re.sub(r'[^a-zA-Z0-9\s\-:]', '', header).strip()
        if h and h.lower() not in seen:
            seen.add(h.lower())
            clean_headers.append(h)
    
    return clean_headers[:10]

def table_to_string(df):
    if isinstance(df, pd.DataFrame):
        try:
            return df.to_markdown()
        except:
            return df.to_string()
    elif isinstance(df, list):
        return "\n".join(["\t".join(map(str, row)) for row in df])
    return str(df)

def extract_text_and_tables_from_pdf(pdf_file):
    text = ""
    tables = []
    headers = []
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            headers = extract_headers_from_pdf(pdf_file)
            
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                if page_text:
                    text += page_text + "\n"
                
                page_tables = page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 5
                })
                
                for table in page_tables:
                    if not table or len(table) < 2:
                        continue
                    
                    try:
                        # Enhanced header detection
                        header_row = next(
                            (i for i, row in enumerate(table)
                            if sum(len(str(cell).strip()) for cell in row) > 15),
                            0
                        )
                        
                        # Header normalization pipeline
                        raw_headers = [str(cell).strip() for cell in table[header_row]]
                        cleaned_headers = []
                        seen = {}
                        
                        for idx, header in enumerate(raw_headers):
                            # Handle empty headers
                            if not header:
                                header = f"Column_{idx+1}"
                            else:
                                # Normalize whitespace and special chars
                                header = re.sub(r'\s+', ' ', header)
                                header = re.sub(r'[^\w\s-]', '', header)
                            
                            # Case-insensitive duplicate handling
                            norm_header = header.lower().replace(' ', '_')
                            if norm_header in seen:
                                seen[norm_header] += 1
                                header = f"{header}_{seen[norm_header]}"
                            else:
                                seen[norm_header] = 0
                            
                            # Final uniqueness guarantee
                            base_header = header
                            counter = 0
                            while header in cleaned_headers:
                                counter += 1
                                header = f"{base_header}_{counter}"
                            
                            cleaned_headers.append(header)

                        # Create dataframe with safe headers
                        data_rows = [
                            [str(cell).strip() for cell in row]
                            for row in table[header_row+1:]
                            if any(cell for cell in row)
                        ]
                        
                        if data_rows:
                            df = pd.DataFrame(data_rows, columns=cleaned_headers)
                            tables.append(df)
                            
                    except Exception as e:
                        st.warning(f"Automatically processed table with minor formatting issues")
                        continue
    
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        raise
    
    return text, tables, headers

def chunk_text(text, chunk_size=10000, overlap=2000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks, model):
    return model.encode(chunks) if chunks else np.array([])

# ----------------------- SIDEBAR LAYOUT ------------------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("📄 Document Settings")
    
    with st.expander("Upload Document", expanded=True):
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    
    with st.expander("Processing Settings", expanded=True):
        chunk_size = st.number_input(
            "Chunk Size (words)", 
            min_value=100, 
            max_value=20000, 
            value=10000,
            step=50
        )
        overlap = st.number_input(
            "Chunk Overlap (words)", 
            min_value=0, 
            max_value=chunk_size - 1, 
            value=2000,
            step=50
        )
        top_k = st.slider("Chunks to Retrieve", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("**About Insurance Copilot**")
    st.markdown("This AI assistant helps you analyze insurance documents and answer your questions.")

# ----------------------- MAIN INTERFACE ------------------------

st.title("🤖 Insurance Copilot")
st.caption("Powered by Imarticus Learning")

if uploaded_file:
    try:
        with st.spinner("Extracting and processing document..."):
            text, tables, headers = extract_text_and_tables_from_pdf(uploaded_file)
            if not text and not tables:
                st.error("No readable content found in the document.")
                st.stop()
            
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) if text else []
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            if chunks:
                embeddings = model.encode(chunks)
                has_embeddings = embeddings.size > 0 and embeddings.shape[0] > 0
            else:
                embeddings = np.array([])
                has_embeddings = False
            
            st.toast(f"✅ Document processed: {len(chunks)} text chunks, {len(tables)} tables, {len(headers)} headers", icon="✅")

        st.subheader("Ask Your Question")
        question = st.text_input(
            "Type your question about the document here...",
            key="question_input",
            label_visibility="collapsed"
        )

        if question:
            with st.spinner("Analyzing document..."):
                text_context = ""
                top_indices = []
                tables_context = ""
                
                try:
                    if has_embeddings:
                        q_embedding = model.encode([question])
                        sims = cosine_similarity(q_embedding, embeddings)[0]
                        top_indices = np.argsort(sims)[-top_k:][::-1]
                        text_context = "\n".join([chunks[i] for i in top_indices])
                except Exception as e:
                    st.error(f"Similarity error: {str(e)}")
                
                if tables:
                    tables_context = "\n\n".join([
                        f"Table {i+1}:\n{table_to_string(df)}" 
                        for i, df in enumerate(tables[:top_k]) if not df.empty
                    ])
                
                full_context = f"Document Headers:\n{', '.join(headers)}\n\nText Context:\n{text_context}\n\nTable Context:\n{tables_context}"
                
                agent = BaseAgent()
                
                if "cpt code" in question.lower():
                    cpt_code = ''.join(filter(str.isdigit, question))
                    # Use the prompt template from u1.py
                    prompt = InsurancePromptTemplates.get_cpt_coverage_prompt(
                        cpt_code=cpt_code,
                        context=full_context
                    )
                else:
                    # Use the general question template from u1.py
                    prompt = InsurancePromptTemplates.get_general_question_prompt(
                        question=question,
                        context=full_context
                    )
                
                answer = agent._call_llm(prompt)
                
                # Format the response using the utility formatter
                if "cpt code" in question.lower():
                    formatted_response = InsuranceResponseFormatter.format_cpt_response(answer)
                    display_answer = "\n".join([f"🔹 **{k}**: {v}" for k, v in formatted_response.items()])
                else:
                    formatted_response = InsuranceResponseFormatter.format_general_response(answer)
                    display_answer = "\n".join([f"🔹 **{k}**: {v}" for k, v in formatted_response.items()])

            st.subheader("Assistant Response")
            with st.chat_message("assistant"):
                st.markdown(display_answer)

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.stop()

else:
    with st.container():
        st.info("📤 Upload a PDF document to begin")
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3767/3767084.png", width=200)
        st.markdown("""
        <div class="fun-fact">
            <h3>🌟 Did You Know?</h3>
            <p>Modern insurance uses AI for:<br>
            - Instant claim processing<br>
            - Fraud detection<br>
            - Personalized premiums</p>
        </div>
        """, unsafe_allow_html=True)