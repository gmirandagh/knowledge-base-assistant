"""
ETL for RAG + Minsearch: robust PDF -> page text -> semantic chunks -> embeddings -> JSONL

Key features:
- PyMuPDF text extraction with layout-aware blocks per page
- LLM-driven chunking with strict JSON-array enforcement + schema validation
- Rule-based fallback chunker to guarantee output even when API fails
- Deterministic chunk_id + full document/page metadata attached to every chunk
- Optional OpenAI embeddings (skipped if no API key is present)
- CLI to process a single file or an entire folder (recursive), with progress
- Output: one JSONL per source document + a combined index JSONL

Requirements (install in your environment):
    pip install openai pymupdf python-dotenv tqdm rapidfuzz

Set environment variables:
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=... (optional)
"""
from pathlib import Path

import os
import re
import io
import json
import time
import math
import uuid
import glob
import shutil
import random
import signal
import string
import logging
import functools
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable, Union

from tqdm import tqdm
import pandas as pd

try:
    import minsearch
except ImportError:
    minsearch = None
import pickle

try:
    import fitz
except Exception as _e:
    fitz = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


DEFAULT_TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

CLIENT = None
if _OPENAI_AVAILABLE and OPENAI_API_KEY:
    if OPENAI_BASE_URL:
        CLIENT = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        CLIENT = OpenAI(api_key=OPENAI_API_KEY)

# Chunking + size controls
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1800"))  # split oversized chunks by sentence
MIN_CHARS_TO_KEEP = int(os.getenv("MIN_CHARS_TO_KEEP", "40"))  # drop very small fragments


# Data paths - ETL outputs only, manual promotion to production
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pdf_path = os.path.join(PROJECT_ROOT, 'etl', 'input_pdfs')
ETL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'etl', 'output')

# ETL output paths (safe staging area)
etl_output_data_path = os.path.join(ETL_OUTPUT_DIR, 'data.jsonl')
etl_output_data_csv_path = os.path.join(ETL_OUTPUT_DIR, 'data.csv')
etl_output_data_index_path = os.path.join(ETL_OUTPUT_DIR, 'data_index.bin')
etl_output_ground_truth_retrieval_data_path = os.path.join(ETL_OUTPUT_DIR, 'ground-truth-retrieval.csv')

# Production data paths (for reference only - no direct writes)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
data_path = os.path.join(DATA_DIR, 'data.jsonl')
data_csv_path = os.path.join(DATA_DIR, 'data.csv')
data_index_path = os.path.join(DATA_DIR, 'data_index.bin')
ground_truth_retrieval_data_path = os.path.join(DATA_DIR, 'ground-truth-retrieval.csv')


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("etl_rag_minsearch")


# Helpers
def retry_on(exceptions, tries=5, base=1.5, ceiling=12.0, jitter=True):
    """Exponential backoff retry decorator."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base
            for attempt in range(tries):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == tries - 1:
                        raise
                    sleep_for = delay + (random.random() * 0.5 if jitter else 0.0)
                    sleep_for = min(sleep_for, ceiling)
                    logger.warning(f"{fn.__name__} failed ({e}); retrying in {sleep_for:.1f}s...")
                    time.sleep(sleep_for)
                    delay *= 2
        return wrapper
    return deco


def split_sentences(text: str) -> List[str]:
    # Sentence splitter
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(])', text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) >= 1]


def normalize_ws(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text.replace('\u00a0', ' ')).strip()


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


# Extraction
def extract_pages_with_pymupdf(pdf_path: str) -> List[Dict]:
    if not fitz:
        raise RuntimeError("PyMuPDF is not installed. Please `pip install pymupdf`.")
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        pages.append({
            "page_number": i + 1,
            "text": text or ""
        })
    doc.close()
    return pages


# LLM (OpenAI) Calls
@retry_on((Exception,), tries=4, base=1.2, ceiling=10.0, jitter=True)
def llm_json_array(prompt: str, model: Optional[str] = None) -> List[Dict]:
    """
    Calls Chat Completions and returns a JSON array. It relies on strong
    prompting and robust parsing rather than the response_format parameter.
    """
    if CLIENT is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL).")
    model = model or DEFAULT_TEXT_MODEL

    resp = CLIENT.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise ETL assistant. "
                    "You will be given a prompt asking for a JSON array. "
                    "Your entire response MUST be a single, valid JSON array `[...]` and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content.strip()

    # Strip unwanted text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("Could not find a JSON array in the model's response.", text, 0)
    
    clean_text = match.group(0)
    
    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error(f"Final JSON parsing failed for text: {clean_text}")
        raise e

    if isinstance(parsed, list):
        return parsed
    else:
        raise ValueError(f"Unexpected JSON type from model after cleaning: {type(parsed)}")


@retry_on((Exception,), tries=4, base=1.2, ceiling=10.0, jitter=True)
def llm_json_object(prompt: str, model: Optional[str] = None) -> Dict:
    """
    Calls Chat Completions with JSON-mode and returns a single JSON object.
    Raises on non-object or parse errors.
    """
    if CLIENT is None:
        raise RuntimeError("OpenAI client not configured.")
    model = model or DEFAULT_TEXT_MODEL

    resp = CLIENT.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a precise ETL assistant. Always return strictly valid, minimal JSON with no markdown."},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content.strip()
    parsed = json.loads(text)

    if isinstance(parsed, dict):
        return parsed
    else:
        raise ValueError(f"Model was expected to return a JSON object, but returned type: {type(parsed)}")


def embed_texts(texts: List[str], model: Optional[str] = None) -> List[Optional[List[float]]]:
    """
    Returns list of embeddings or list of None (if no client/key).
    """
    if CLIENT is None or not texts:
        return [None] * len(texts)

    model = model or DEFAULT_EMBED_MODEL
    # Batching
    out = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = CLIENT.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
        time.sleep(0.1)
    return out


# Example-based prompts
def build_metadata_prompt(full_text: str) -> str:
    prompt_template = """
You are a precise ETL assistant. Your task is to extract key metadata from the provided text of a scientific paper.
Analyze the text and extract the required fields. If a field is not present, use a sensible default (e.g., empty string, empty list, null).

The text to analyze:

---
{full_text}
---

Provide the output in a parsable JSON object without using code blocks. The JSON object must conform to the following structure:

{{
  "title": "The full title of the paper",
  "authors": ["Author One", "Author Two"],
  "year": 2024,
  "abstract": "A one-paragraph summary of the abstract section."
}}
""".strip()

    return prompt_template.format(full_text=full_text[:8000])


def build_chunking_prompt(page_text: str, page_number: int, document_id: str, metadata_obj: Dict) -> str:
    prompt_template = """
You are a precise ETL assistant. Your task is to chunk the text from a single document page into a list of JSON objects.

Analyze the page text provided and create a chunk for each distinct paragraph, list, or caption.

The page text to analyze:
---
{page_text}
---

Provide the output in a parsable JSON array `[...]` without using code blocks. Each object in the array must conform to the following structure and rules:

CRITICAL OUTPUT RULES:
- Your entire output MUST be a valid JSON array `[...]`.
- If you only find one chunk on the page, you MUST still return it as an array with a single object: `[ {{...}} ]`.
- Do not include any text before `[` or after `]`.

RULES FOR EACH CHUNK OBJECT:
- `document_metadata`: Copy the provided metadata object verbatim.
- `chunk_id`: Use the literal placeholder string "{document_id}_PAGE{page_number}___INDEX".
- `section_title`: The most recent section heading seen on the page. If none is visible, use "Unknown".
- `chunk_type`: Must be one of ["paragraph", "list", "figure_caption", "table_caption", "conclusion", "abstract"].
- `page_number`: Must be the integer {page_number}.
- `content`: The verbatim text of the chunk.
- Exclude any content from sections titled "References", "Bibliography", or "Appendix".

EXAMPLE OUTPUT STRUCTURE:
[
  {{
    "document_metadata": {metadata_obj_str},
    "chunk_id": "{document_id}_PAGE{page_number}___INDEX",
    "section_title": "Example Section Title",
    "chunk_type": "paragraph",
    "page_number": {page_number},
    "content": "This is the text content of the first chunk from the page."
  }}
]
""".strip()

    return prompt_template.format(
        page_text=page_text,
        page_number=page_number,
        document_id=document_id,
        metadata_obj_str=json.dumps(metadata_obj, ensure_ascii=False)
    )


# Fallback chunker
LIST_LINE = re.compile(r'^\s*([-*•·]|\d+[\.)])\s+')

def fallback_chunk_page(page_text: str, page_number: int, metadata_obj: Dict, document_id: str) -> List[Dict]:
    # 1) Early exit if references/bibliography page
    if re.match(r'^\s*(References|Bibliography|Appendix)\b', page_text.strip(), re.I):
        return []

    # 2) Split paragraphs by blank lines
    blocks = [b.strip() for b in re.split(r'\n\s*\n', page_text) if b.strip()]
    chunks = []
    for bi, block in enumerate(blocks):
        lines = [normalize_ws(l) for l in block.splitlines() if normalize_ws(l)]
        if not lines:
            continue
        if sum(1 for l in lines if LIST_LINE.match(l)) >= max(2, int(0.5 * len(lines))):
            content = "\n".join(lines)
            chunk_type = "list"
        else:
            content = " ".join(lines)
            chunk_type = "paragraph"

        chunks.append({
            "document_metadata": metadata_obj,
            "chunk_id": f"{document_id}_PAGE{page_number}___INDEX",
            "section_title": "Unknown",
            "chunk_type": chunk_type,
            "page_number": page_number,
            "content": content
        })

    # Split oversized chunks by sentences
    final = []
    for c in chunks:
        if len(c["content"]) <= MAX_CHUNK_CHARS:
            final.append(c)
        else:
            sents = split_sentences(c["content"])
            acc = []
            cur = ""
            for s in sents:
                if len(cur) + 1 + len(s) > MAX_CHUNK_CHARS and cur:
                    acc.append(cur.strip())
                    cur = s
                else:
                    cur = (cur + " " + s).strip() if cur else s
            if cur:
                acc.append(cur.strip())
            for a in acc:
                if len(a) >= MIN_CHARS_TO_KEEP:
                    d = dict(c)
                    d["content"] = a
                    final.append(d)
    return final


# Validation and normalization
REQUIRED_KEYS = {"document_metadata","chunk_id","section_title","chunk_type","page_number","content"}

def validate_chunks(chunks: List[Dict], page_number: int) -> List[Dict]:
    valid = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        if not REQUIRED_KEYS.issubset(ch.keys()):
            continue
        ch["section_title"] = ch.get("section_title") or "Unknown"
        try:
            ch["page_number"] = int(ch.get("page_number"))
        except Exception:
            ch["page_number"] = page_number
        ch["content"] = normalize_ws(str(ch.get("content","")))
        if len(ch["content"]) < MIN_CHARS_TO_KEEP:
            continue
        valid.append(ch)
    return valid


def assign_indices(chunks: List[Dict]) -> List[Dict]:
    per_page_counter = {}
    out = []
    for ch in chunks:
        page = ch["page_number"]
        per_page_counter[page] = per_page_counter.get(page, 0) + 1
        idx = per_page_counter[page]
        ch["chunk_id"] = ch["chunk_id"].replace("___INDEX", f"{idx:03d}")
        out.append(ch)
    return out


# Orchestration
@dataclass
class DocumentRecord:
    source_path: str
    document_id: str
    title: str = ""
    authors: List[str] = None
    publication: str = ""
    year: Optional[int] = None
    doi: str = ""
    abstract: str = ""
    top_level_sections: List[str] = None

    def to_dict(self):
        d = asdict(self)
        if d["authors"] is None: d["authors"] = []
        if d["top_level_sections"] is None: d["top_level_sections"] = []
        return d


def slugify_title(title: str) -> str:
    if not title:
        return uuid.uuid4().hex[:12]
    s = re.sub(r'[^a-z0-9]+', '-', title.lower())
    s = s.strip('-')
    return s[:64] if s else uuid.uuid4().hex[:12]


def extract_metadata(full_text: str) -> DocumentRecord:
    try:
        prompt = build_metadata_prompt(full_text)
        obj = llm_json_object(prompt)
    except Exception as e:
        logger.warning(f"Metadata LLM failed, using fallback: {e}")
        first_lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        title = first_lines[0] if first_lines else "Untitled"
        obj = { "title": title }
    
    doc_id = slugify_title(obj.get("title",""))
    return DocumentRecord(
        source_path="",
        document_id=doc_id,
        title=obj.get("title",""),
        authors=obj.get("authors") or [],
        year=obj.get("year"),
        abstract=obj.get("abstract",""),
    )


def chunk_one_page(page_text: str, page_number: int, doc_meta: DocumentRecord) -> List[Dict]:
    # LLM-first
    try:
        prompt = build_chunking_prompt(
            page_text=page_text,
            page_number=page_number,
            document_id=doc_meta.document_id,
            metadata_obj=doc_meta.to_dict()
        )
        arr = llm_json_array(prompt)
        chunks = validate_chunks(arr, page_number)
        if chunks:
            return chunks
    except Exception as e:
        logger.warning(f"LLM chunking failed on page {page_number}: {e}")

    # Fallback
    chunks = fallback_chunk_page(page_text, page_number, doc_meta.to_dict(), doc_meta.document_id)
    return validate_chunks(chunks, page_number)


def process_pdf(pdf_path: str) -> List[Dict]:
    logger.info(f"Processing: {pdf_path}")
    pages = extract_pages_with_pymupdf(pdf_path)
    full_text = "\n\n".join([p["text"] for p in pages])

    # Metadata (LLM + fallback)
    meta = extract_metadata(full_text)
    meta.source_path = str(Path(pdf_path).resolve())

    all_chunks = []
    for p in pages:
        ch = chunk_one_page(p["text"], p["page_number"], meta)
        all_chunks.extend(ch)

    all_chunks = assign_indices(all_chunks)

    # Split oversized chunks (2nd pass)
    final_chunks = []
    for ch in all_chunks:
        content = ch["content"]
        if len(content) <= MAX_CHUNK_CHARS:
            final_chunks.append(ch)
        else:
            sents = split_sentences(content)
            cur = ""
            buf = []
            for s in sents:
                if len(cur) + 1 + len(s) > MAX_CHUNK_CHARS and cur:
                    buf.append(cur.strip())
                    cur = s
                else:
                    cur = (cur + " " + s).strip() if cur else s
            if cur:
                buf.append(cur.strip())
            for i, b in enumerate(buf, start=1):
                if len(b) < MIN_CHARS_TO_KEEP:
                    continue
                dup = dict(ch)
                dup["chunk_id"] = f"{meta.document_id}_PAGE{ch['page_number']}_{uuid.uuid4().hex[:6]}_{i:02d}"
                dup["content"] = b
                final_chunks.append(dup)

    # Embeddings for future vector RAG (Minsearch will not use them)
    embs = embed_texts([c["content"] for c in final_chunks])
    for c, e in zip(final_chunks, embs):
        c["embedding"] = e

        c["id"] = c["chunk_id"]
        c["title"] = meta.title or meta.document_id
        c["url"] = meta.source_path
        c["text"] = c["content"]

    return final_chunks


def write_jsonl(records: Iterable[Dict], path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_csv_from_jsonl(jsonl_path: str, csv_path: str):
    """Generate a CSV file from JSONL, removing embeddings and flattening document_metadata."""
    logger.info(f"Generating CSV from: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL file not found: {jsonl_path}")
        return
    
    # Load documents
    documents = []
    with open(jsonl_path, 'rt', encoding='utf-8') as f_in:
        for line in f_in:
            doc = json.loads(line)
            doc.pop('embedding', None)  # Remove embedding field
            documents.append(doc)
    
    if not documents:
        logger.warning("No documents found in JSONL file")
        return
    
    # Convert to DataFrame and flatten metadata
    df = pd.DataFrame(documents)
    
    # Flatten document_metadata if it exists
    if 'document_metadata' in df.columns:
        metadata_df = pd.json_normalize(df['document_metadata'])
        # Rename conflicting columns
        metadata_df.rename(columns={
            'title': 'document_title', 
            'url': 'document_url'
        }, inplace=True)
        df = df.drop(columns=['document_metadata']).join(metadata_df)
    
    # Final check to remove any remaining embedding columns
    if 'embedding' in df.columns:
        df = df.drop(columns=['embedding'])
    
    # Save to CSV
    ensure_dir(str(Path(csv_path).parent))
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"✅ CSV saved to: {csv_path} (records={len(df)})")


def cli(input_path: str):
    ensure_dir(ETL_OUTPUT_DIR)
    input_path = Path(input_path)
    pdfs = []
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdfs = [str(input_path)]
    elif input_path.is_dir():
        pdfs = [str(p) for p in input_path.rglob("*.pdf")]
    else:
        raise SystemExit("Pass a PDF file or a folder containing PDFs.")

    combined = []
    for pdf in tqdm(pdfs, desc="Documents"):
        try:
            chunks = process_pdf(pdf)
            out_path = Path(ETL_OUTPUT_DIR) / (Path(pdf).stem + ".jsonl")
            write_jsonl(chunks, str(out_path))
            combined.extend(chunks)
        except Exception as e:
            logger.error(f"Failed: {pdf} ({e})")

    # Save combined JSONL to ETL output only
    write_jsonl(combined, etl_output_data_path)
    logger.info(f"✅ Combined JSONL saved to: {etl_output_data_path} (records={len(combined)})")
    
    # Generate CSV version without embeddings
    generate_csv_from_jsonl(etl_output_data_path, etl_output_data_csv_path)
    
    logger.info(f"""
📁 ETL Processing Complete! Files saved to: {ETL_OUTPUT_DIR}
   - data.jsonl: {len(combined)} records
   - data.csv: CSV version without embeddings
   - Individual PDFs: {len(pdfs)} files processed

🔄 Next steps:
   1. Review the generated files in etl/output/
   2. Build search index: python etl_rag.py --index etl/output/data.jsonl
   3. Generate Q&A pairs: python etl_rag.py --generate-qa etl/output/data.jsonl  
   4. Manually copy to data/ directory when satisfied with quality
    """)


def build_minsearch_index(jsonl_path: str, output_path: str):
    """Builds and saves a Minsearch index from a JSONL file to ETL output directory only."""
    if not minsearch:
        raise RuntimeError("Minsearch is not installed. Please `pip install minsearch`.")
    
    logger.info(f"Building Minsearch index from: {jsonl_path}")
    
    with open(jsonl_path, 'rt', encoding='utf-8') as f_in:
        docs = [json.loads(line) for line in f_in]

    documents = [doc for doc in docs if doc.get('text')]  # Use 'text' field instead of 'embedding'
    
    index = minsearch.Index(
        text_fields=["text"],
        keyword_fields=["title", "document_id", "chunk_type", "section_title"]
    )
    index.fit(documents)

    # Save only to ETL output directory
    ensure_dir(str(Path(etl_output_data_index_path).parent))
    with open(etl_output_data_index_path, 'wb') as f_out:
        pickle.dump(index, f_out)

    logger.info(f"✅ Minsearch index with {len(documents)} records saved to: {etl_output_data_index_path}")
    logger.info(f"""
🔍 Search Index Built Successfully!
   - Records indexed: {len(documents)}
   - Output: {etl_output_data_index_path}
   
🔄 Next step: Copy to data/ directory when ready:
   cp {etl_output_data_index_path} {data_index_path}
    """)


# Ground Truth Generation Functions
PROMPT_TEMPLATE = """
You are an AI assistant tasked with creating a high-quality ground-truth dataset for evaluating a Retrieval-Augmented Generation (RAG) system.
Your role is to act as an inquisitive researcher or engineer and generate **5 distinct, realistic, and high-quality questions** that can be **completely answered** using only the information in the provided CONTEXT.

### RULES:
1. **Exactly Five Questions**: Produce precisely 5 questions, no more, no less.
2. **Grounded in Context**: Each question must be answerable *exclusively* from the provided CONTEXT. Do not introduce outside knowledge.
3. **Realistic User Queries**: Formulate questions as if they were asked by a real person seeking information. Avoid exam-style or trivial "fill-in-the-blank" questions.
4. **Clarity and Precision**: Questions should be well-structured, specific, and self-contained. Avoid ambiguity and overly short phrasing.
5. **No Copy-Paste**: Do not copy sentences verbatim from the CONTEXT. Always rephrase naturally, ensuring the question sounds conversational.
6. **Variety of Question Types**: Mix styles and intents (e.g., definitions, comparisons, processes, causes/effects, implications, factual lookups). At least one "how/why" style question is required.
7. **Balanced Scope**: Avoid questions that are either too broad ("Explain everything about…") or too narrow ("What is the third word in…"). Each question should target a meaningful, self-contained piece of information.

### CONTEXT:
---
Source Document: {title}
Section: {section_title} (Page {page_number})
Content: {content}
---

### OUTPUT FORMAT:
Return the result as a single valid parsable JSON (without markdown/code block formatting), exactly in this structure:
{{"questions": ["question1", "question2", "question3", "question4", "question5"]}}
""".strip()

MAX_RETRIES = 3

def generate_questions(doc: Dict, model: Optional[str] = None) -> str:
    """Generate questions for a document chunk using LLM."""
    if CLIENT is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL).")
    
    model = model or DEFAULT_TEXT_MODEL
    
    prompt = PROMPT_TEMPLATE.format(
        title=doc.get('title', 'Unknown Document'),
        section_title=doc.get('section_title', 'Unknown Section'),
        page_number=doc.get('page_number', 'Unknown'),
        content=doc.get('content', '')
    )
    
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def safe_generate_questions(doc: Dict, max_retries: int = MAX_RETRIES) -> Dict:
    """
    Generate questions and ensure valid JSON output.
    Retries up to `max_retries` times if JSON decoding fails.
    """
    if CLIENT is None:
        logger.warning("OpenAI client not configured. Skipping question generation.")
        return {"questions": []}
    
    for attempt in range(1, max_retries + 1):
        try:
            questions_raw = generate_questions(doc)
            questions = json.loads(questions_raw)
            
            # Ensure expected structure
            if isinstance(questions, dict) and "questions" in questions:
                return questions
            else:
                raise ValueError("Invalid JSON structure, missing 'questions' key")
                
        except Exception as e:
            logger.warning(f"[Attempt {attempt}/{max_retries}] JSON parsing failed for doc {doc.get('id', 'unknown')}: {e}")
            
            if attempt < max_retries:
                # Try to fix malformed JSON
                try:
                    fix_prompt = f"""
                    The following text was supposed to be valid JSON but is malformed:
                    
                    {questions_raw}
                    
                    Please return ONLY a valid JSON object in the form:
                    {{"questions": ["question1", "question2", "question3", "question4", "question5"]}}
                    """
                    
                    response = CLIENT.chat.completions.create(
                        model=DEFAULT_TEXT_MODEL,
                        messages=[{"role": "user", "content": fix_prompt}]
                    )
                    questions_raw = response.choices[0].message.content
                    time.sleep(1)  # Rate limiting
                    
                except Exception as fix_error:
                    logger.warning(f"Failed to fix JSON: {fix_error}")
                    time.sleep(1)
    
    # Log and skip when failed
    logger.error(f"❌ Failed to get valid JSON after {max_retries} attempts for doc {doc.get('id', 'unknown')}")
    return {"questions": []}

def generate_ground_truth_qa(jsonl_path: str):
    """Generate ground truth Q&A pairs from a JSONL file and save to ETL output directory only."""
    logger.info(f"Loading document chunks from: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    # Load documents
    documents = []
    with open(jsonl_path, 'rt', encoding='utf-8') as f_in:
        for line in f_in:
            doc = json.loads(line)
            doc.pop('embedding', None)  # Remove embedding field if present
            documents.append(doc)
    
    logger.info(f"Successfully loaded {len(documents)} document chunks")
    
    if not documents:
        logger.warning("No documents found in JSONL file")
        return
    
    # Check document structure
    logger.info(f"Document keys: {list(documents[0].keys())}")
    
    # Generate questions for each document
    results = {}
    
    logger.info("Generating questions for documents...")
    for doc in tqdm(documents, desc="Generating Q&A pairs"):
        doc_id = doc.get('id') or doc.get('chunk_id', str(uuid.uuid4()))
        
        if doc_id in results:
            continue
            
        questions_result = safe_generate_questions(doc)
        results[doc_id] = questions_result.get('questions', [])
        
        # Add small delay to avoid hitting rate limits
        time.sleep(0.1)
    
    # Flatten results into list of (doc_id, question) pairs
    final_results = []
    for doc_id, questions in results.items():
        for q in questions:
            if q.strip():  # Only add non-empty questions
                final_results.append((doc_id, q.strip()))
    
    logger.info(f"Generated {len(final_results)} Q&A pairs")
    
    if not final_results:
        logger.warning("No Q&A pairs were generated")
        return
    
    # Create DataFrame and save to ETL output directory only
    df_results = pd.DataFrame(final_results, columns=['id', 'question'])
    
    ensure_dir(str(Path(etl_output_ground_truth_retrieval_data_path).parent))
    df_results.to_csv(etl_output_ground_truth_retrieval_data_path, index=False, encoding='utf-8')
    
    logger.info(f"✅ Ground truth Q&A pairs saved to: {etl_output_ground_truth_retrieval_data_path}")
    logger.info(f"""
❓ Q&A Generation Complete!
   - Generated: {len(final_results)} question-answer pairs
   - Output: {etl_output_ground_truth_retrieval_data_path}
   
🔄 Next step: Copy to data/ directory when ready:
   cp {etl_output_ground_truth_retrieval_data_path} {ground_truth_retrieval_data_path}
    """)
    
    return len(final_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ETL and data processing pipeline for the Knowledge Base Assistant.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # .jsonl
    parser.add_argument(
        "path",
        nargs='?',
        default=None,
        help="Path to a PDF file or a folder containing PDFs for processing."
    )
    # /bin
    parser.add_argument(
        "--index",
        metavar="JSONL_FILE",
        help="Build a Minsearch index from the specified combined JSONL file."
    )

    # ground-truth-retrieval.csv
    parser.add_argument(
        "--generate-qa", 
        metavar="JSONL_FILE", 
        help="Generate Q&A pairs for evaluation from the specified JSONL file."
    )


    args = parser.parse_args()

    if args.path:
        cli(args.path)
    elif args.index:
        output_index_path = "data/data_index.bin"
        build_minsearch_index(args.index, output_index_path)
    elif args.generate_qa:
        generate_ground_truth_qa(args.generate_qa)
    else:
        parser.print_help()
        raise SystemExit("\nError: You must specify a PDF path for processing, use --index to build search index, or use --generate-qa to generate Q&A pairs.")