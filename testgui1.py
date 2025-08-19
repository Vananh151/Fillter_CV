# testgui1.py
# Streamlit CV Filter — full version
# - Giữ tính năng semantic chunking, cross-encoder (tùy chọn)
# - Phân loại must-have / nice-to-have, phạt khi thiếu must-have
# - Ước lượng kinh nghiệm (year ranges + simple years)
# - OCR (pdf scan / ảnh) bằng pytesseract fallback
# - So khớp nội dung JD theo dòng + highlight các skill trùng

import streamlit as st
import io, os, re, json, uuid, unicodedata, math
from typing import List, Dict, Any, Tuple
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pdfplumber
from docx import Document
from dotenv import load_dotenv
import pytesseract
import numpy as np
from datetime import datetime
import re
import spacy
from typing import List, Dict, Any
import base64
# embeddings
try:
    from sentence_transformers import SentenceTransformer, util
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception:
    # Nếu thiếu package, sẽ báo lỗi khi chạy phần embed/cross-encoder
    SentenceTransformer = None
    util = None
    CrossEncoder = None

# --- Cấu hình Tesseract (điều chỉnh theo máy của bạn) ---
# Nếu path khác, bạn có thể comment / set biến môi trường
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

st.set_page_config(page_title="CV Filter Chatbot (Streamlit)", layout="wide")

# ---------------------
# Config / Helpers
# ---------------------
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_COMMON_SKILLS = [
    "python","java","aws","docker","kubernetes","django","flask","pytorch","tensorflow",
    "react","node","sql","excel","spark","etl","airflow","gcp","azure","linux","git","nlp",
    "opencv","rest","fastapi","django rest","pandas","numpy","scikit","mlflow"
]

# Try load embedder (if sentence-transformers present)
@st.cache_resource(show_spinner=False)
def load_embedder():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(EMBED_MODEL_NAME)

embedder = load_embedder()

@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception:
        return None

cross_encoder = load_cross_encoder()

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")
YEARS_RE_SIMPLE =  re.compile(
    r"(?:hơn\s*|more than\s*|over\s*)?(\d+)\+?\s*"
    r"(?:year|years|năm|yrs|yr)\s*"
    r"(?:kinh nghiệm|of experience|work experience)?",
    re.IGNORECASE
)
# Khoảng năm: 2018-2023, 2019–Present, 2020 đến nay, 2021-hiện tại...
YEAR_RANGE_RE = re.compile(
    r"(?P<start>(?:19|20)\d{2})\s*(?:[-–—]|to|đến|->|—)\s*(?P<end>(?:19|20)\d{2}|present|hiện tại|nay)",
    re.IGNORECASE
)

# --- Chuẩn hoá text để match cơ bản (bỏ dấu, hạ chữ) ---
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))  # bỏ dấu
    text = text.lower()
    return text

# ================== Chuẩn hoá đa ngôn ngữ + so khớp nội dung JD ==================
import nltk
from nltk.stem import WordNetLemmatizer
# Vi tokenizer (pyvi) - dùng để tách Tiếng Việt
try:
    from pyvi.ViTokenizer import ViTokenizer
except Exception:
    # Nếu không có pyvi, fallback: simple whitespace (ít chuẩn hơn)
    ViTokenizer = None

# Tải dữ liệu cho NLTK 1 lần (Streamlit cache sẽ giữ lại)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

def preprocess_multilang(text: str) -> List[str]:
    """Tokenize tiếng Việt + lemmatize tiếng Anh, giữ Unicode tiếng Việt."""
    if not text:
        return []
    if ViTokenizer:
        text_tok = ViTokenizer.tokenize(text)
    else:
        text_tok = text
    text_tok = text_tok.lower()
    text_tok = re.sub(
        r"[^a-z0-9áàạảãăắằặẳẵâấầậẩẫéèẹẻẽêếềệểễ"
        r"íìịỉĩóòọỏõôốồộổỗơớờợởỡúùụủũưứừựửữ"
        r"ýỳỵỷỹ_ ]",
        " ",
        text_tok,
    )
    tokens = text_tok.split()
    processed = []
    for tok in tokens:
        if re.match(r"^[a-z]+$", tok):  # chỉ lemmatize từ thuần chữ cái latin
            processed.append(lemmatizer.lemmatize(tok))
        else:
            processed.append(tok)
    return processed

def compare_cv_to_jd_content(cv_text: str, jd_text: str) -> List[Dict[str, Any]]:
    """So khớp từng dòng CV với các từ khóa quan trọng trong JD."""
    jd_tokens = set(extract_keywords(jd_text))
    results = []
    if not jd_tokens:
        return results

    for line in (cv_text or "").splitlines():
        line_tokens = set(extract_keywords(line))
        matched = jd_tokens & line_tokens
        match_ratio = len(matched) / max(1, len(jd_tokens))
        results.append({
            "line": line.strip(),
            "matched_words": sorted(matched),
            "match_ratio": round(match_ratio, 3)
        })
    return results
# ======================================================================

# --- Chunking cho semantic ---
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150, max_chunks: int = 12) -> List[str]:
    text = text or ""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(text) and len(chunks) < max_chunks:
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# --- OCR helpers ---
def extract_text_from_image_bytes(b: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(b))
        return pytesseract.image_to_string(img, lang="eng+vie")
    except Exception:
        return ""

def extract_text_from_pdf_bytes(b: bytes) -> str:
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                # Nếu text quá ít, thử OCR trang đó
                if len(text.strip()) < 30:
                    try:
                        pil_img = p.to_image(resolution=250).original
                        text_ocr = pytesseract.image_to_string(pil_img, lang="eng+vie")
                        text += "\n" + (text_ocr or "")
                    except Exception:
                        pass
                if text:
                    parts.append(text)
    except Exception:
        return ""
    return "\n".join(parts)

def extract_text_from_docx_bytes(b: bytes) -> str:
    try:
        doc = Document(io.BytesIO(b))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

# --- Kỹ năng & từ đồng nghĩa cơ bản ---
SYNONYMS = {
    "tf": "tensorflow",
    "sklearn": "scikit",
    "np": "numpy",
    "pd": "pandas",
    "js": "javascript",
    "ts": "typescript",
    "sql server": "sql",
    "postgres": "postgresql",
    "more": ">"
}

def normalize_skill(s: str) -> str:
    s2 = normalize_text(s).strip()
    return SYNONYMS.get(s2, s2)

# --- Gộp khoảng năm để tránh cộng trùng ---
def merge_year_ranges(ranges: List[Tuple[int, int]]) -> int:
    if not ranges:
        return 0
    ranges.sort(key=lambda x: x[0])
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # trùng hoặc nối tiếp
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)

# --- Parse thông tin cấu trúc từ CV (mở rộng: nhận JD + skill lists) ---
JD_MATCH_THRESHOLD = 0.5
def parse_structured_info(text: str, jd_text: str, must_have_skills: list, nice_to_have_skills: list) -> dict:
    parsed = {}
    emails = EMAIL_RE.findall(text or "")
    phones = PHONE_RE.findall(text or "")

    # --- tìm năm kinh nghiệm ---
    years_simple = YEARS_RE_SIMPLE.findall(text or "") or []
    years_simple = [int(y) for y in years_simple if str(y).isdigit()]

    year_ranges = []
    for m in YEAR_RANGE_RE.finditer(text or ""):
        start = m.group("start")
        end = m.group("end")
        try:
            start = int(start)
            if re.match(r"^(present|hiện tại|nay)$", end, flags=re.I):
                end_year = datetime.now().year
            else:
                end_year = int(end)
            if 1900 <= start <= 2100 and 1900 <= end_year <= 2100 and end_year >= start:
                year_ranges.append((start, end_year))
        except Exception:
            pass

    total_years_from_ranges = merge_year_ranges(year_ranges)
    est_years = max(total_years_from_ranges if total_years_from_ranges > 0 else 0,
                    max(years_simple) if years_simple else 0)

    parsed["emails"] = list(dict.fromkeys(emails))
    parsed["phones"] = list(dict.fromkeys(phones))
    parsed["years_mentioned"] = years_simple
    parsed["estimated_experience_years"] = est_years

    # --- Tokens và skill matching ---
    cv_tokens = set(preprocess_multilang(text or ""))
    must_tokens = set(preprocess_multilang(" ".join(must_have_skills)))
    nice_tokens = set(preprocess_multilang(" ".join(nice_to_have_skills)))

    parsed["must_hit_tokens"] = sorted(must_tokens & cv_tokens)
    parsed["nice_hit_tokens"] = sorted(nice_tokens & cv_tokens)
    parsed["missing_must_tokens"] = sorted(must_tokens - cv_tokens)

    # --- JD content match (chỉ tập trung vào skills trong JD) ---
    jd_match_tokens = must_tokens | nice_tokens
    overlap = jd_match_tokens & cv_tokens
    parsed["jd_match_ratio"] = round(len(overlap) / max(1, len(jd_match_tokens)), 3)

    # JD match = phải có đủ must-have + đạt tỷ lệ threshold
    parsed["is_skill_match"] = len(parsed["missing_must_tokens"]) == 0
    parsed["is_jd_match"] = parsed["is_skill_match"] and parsed["jd_match_ratio"] >= JD_MATCH_THRESHOLD

    return parsed
# >>> NEW: JD years extraction (regex + optional spaCy NER)
@st.cache_resource(show_spinner=False)
def _load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

_nlp = _load_spacy()

_JD_YEARS_PATTERNS = [
    # at least X years
    r"(?:at\s+least|minimum|min\.?|tối\s+thiểu|ít\s+nhất)\s*(\d+)\s*\+?\s*(?:year|years|năm)",
    
    # X years of experience
    r"(\d+)\s*\+?\s*(?:year|years|năm)\s*(?:of)?\s*(?:experience|kinh\s+nghiệm)",
    
    # X years exp
    r"(\d+)\s*(?:year|years|năm)\s*(?:exp\.?|experience)",
    
    # X+ years
    r"(\d+)\s*\+\s*(?:years|năm)",
    
    # more than X years
    r"(?:hơn\s*|more than\s*|over\s*)?(\d+)\+?\s*(?:year|years|năm)"
]

def extract_keywords(text: str) -> List[str]:
    """Tách từ khóa quan trọng (skill, title, requirement) từ JD."""
    tokens = preprocess_multilang(text)
    stopwords = {"and", "or", "with", "the", "a", "an", "of", "to", "in", "for", "on", 
                 "at", "với", "có", "là", "của", "trong", "và", "hoặc"}  # thêm stopwords đa ngôn ngữ
    keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
    return list(set(keywords))
def extract_min_years_from_jd(jd_text: str) -> int:
    """Ưu tiên regex, fallback NER (tìm số gần 'year/years'). Trả về 0 nếu không thấy."""
    text = jd_text or ""
    for pat in _JD_YEARS_PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return int(m.group(1))
            except:
                pass

    # Fallback NER: tìm token 'year/years' và số ở gần
    if _nlp is not None:
        try:
            doc = _nlp(text)
            toks = list(doc)
            for i, t in enumerate(toks):
                if t.lower_ in ("year", "years", "năm"):
                    for j in range(max(0, i-3), i+1):
                        if toks[j].like_num:
                            try:
                                return int(toks[j].text)
                            except:
                                continue
        except Exception:
            pass
    return 0

# --- Tính semantic similarity với chunk ---
def semantic_similarity(cv_text: str, jd_text: str, use_chunks: bool = True, max_chunks: int = 12, top_k: int = 3) -> float:
    if embedder is None or util is None:
        return 0.0

    if not use_chunks:
        emb_cv = embedder.encode(cv_text or "", convert_to_tensor=True)
        emb_jd = embedder.encode(jd_text or "", convert_to_tensor=True)
        sem_sim = float(util.cos_sim(emb_cv, emb_jd).item())
        return max(0.0, min(1.0, sem_sim))  # cosine đã ở [-1,1], clamp về [0,1]

    chunks = chunk_text(cv_text or "", max_chunks=max_chunks)
    jd_emb = embedder.encode(jd_text or "", convert_to_tensor=True)

    sims = []
    for ch in chunks:
        emb = embedder.encode(ch, convert_to_tensor=True)
        sims.append(float(util.cos_sim(emb, jd_emb).item()))

    if not sims:
        return 0.0

    # Lấy trung bình top-k chunk thay vì chỉ max
    sims = sorted(sims, reverse=True)[:top_k]
    avg_sim = sum(sims) / len(sims)

    # Clamp kết quả về [0,1]
    return max(0.0, min(1.0, avg_sim))


# --- (Tùy chọn) Rerank Cross-Encoder ---
def cross_encoder_score(cv_text: str, jd_text: str, enabled: bool, max_chunks: int = 6) -> float:
    if not enabled or cross_encoder is None:
        return -1.0
    chunks = chunk_text(cv_text or "", chunk_size=800, overlap=100, max_chunks=max_chunks)
    pairs = [(jd_text or "", ch) for ch in chunks]
    try:
        scores = cross_encoder.predict(pairs)
        return float(np.max(scores)) if len(scores) else -1.0
    except Exception:
        return -1.0

# --- Highlight kỹ năng trong text ---
def highlight_terms(text: str, terms: List[str]) -> str:
    """Highlight terms (basic). Uses normalized skill tokens for matching."""
    if not text:
        return ""
    html = text
    # sort by length to replace longer tokens first
    for t in sorted(set([normalize_skill(tt) for tt in terms if tt]), key=len, reverse=True):
        if not t:
            continue
        try:
            pattern = re.compile(re.escape(t), re.IGNORECASE)
            html = pattern.sub(rf"<span style='background:#fff3cd;padding:0 2px;border-radius:3px'>{t}</span>", html)
        except re.error:
            continue
    return html

# --- Tính điểm tổng + skill token & content match ---
def score_candidate(cv_text: str, jd: dict, cfg: dict) -> dict:
    # Parse structured info (uses JD + skills to compute flags)
    parsed = parse_structured_info(cv_text, jd.get("content", ""), jd.get("must_have_skills", []), jd.get("nice_to_have_skills", []))

    # Semantic
    sem_score = semantic_similarity(cv_text, jd.get("content", ""), use_chunks=cfg.get("use_chunks", True), max_chunks=cfg.get("max_chunks", 12))

    # Cross-encoder
    ce_score = cross_encoder_score(cv_text, jd.get("content", ""), enabled=cfg.get("use_cross_encoder", False), max_chunks=min(6, cfg.get("max_chunks", 12)))
    if ce_score >= 0:
        if ce_score > 1:
            ce_score = 1 - math.exp(-ce_score)
        sem_score = max(sem_score, ce_score)

    # Keywords scoring
    must_tokens = set(preprocess_multilang(" ".join(jd.get("must_have_skills", []))))
    nice_tokens = set(preprocess_multilang(" ".join(jd.get("nice_to_have_skills", []))))
    cv_tokens = set(preprocess_multilang(cv_text or ""))
    must_hit_tokens = must_tokens & cv_tokens
    nice_hit_tokens = nice_tokens & cv_tokens

    denom = max(1, len(must_tokens) + 0.5 * len(nice_tokens))
    kw_score = (len(must_hit_tokens) + 0.5 * len(nice_hit_tokens)) / denom

    missing_must_tokens = sorted(must_tokens - cv_tokens)

    # Experience
    exp_years = parsed.get("estimated_experience_years", 0)
    min_exp = float(jd.get("min_experience_years", 0) or 0)

    # So sánh chính xác
    if min_exp <= 0:
        exp_score = 1.0
        exp_match = True
    else:
        exp_score = min(exp_years / min_exp, 1.0)  # tỉ lệ không vượt quá 1
        exp_match = exp_years >= min_exp
    # Combine weights
    w_kw, w_sem, w_exp = cfg.get("w_kw", 0.35), cfg.get("w_sem", 0.45), cfg.get("w_exp", 0.2)
    w_sum = max(1e-6, w_kw + w_sem + w_exp)
    w_kw, w_sem, w_exp = w_kw/w_sum, w_sem/w_sum, w_exp/w_sum
    total = w_kw*kw_score + w_sem*sem_score + w_exp*exp_score

    # penalty when missing must-have
    if missing_must_tokens:
        total *= cfg.get("missing_must_penalty", 0.8)

    # top content matches (per-line)
    content_matches = compare_cv_to_jd_content(cv_text or "", jd.get("content", "") or "")
    top_content_matches = sorted(content_matches, key=lambda x: x["match_ratio"], reverse=True)[:5]

    return {
        "kw_score": round(kw_score, 3),
        "semantic_score": round(sem_score, 3),
        "exp_score": round(exp_score, 3),
        "total_score": round(total, 3),
        "exp_years_estimated": exp_years,
        "parsed": parsed,
        "must_hit_tokens": sorted(list(must_hit_tokens)),
        "nice_hit_tokens": sorted(list(nice_hit_tokens)),
        "missing_must_tokens": missing_must_tokens,
        "hits": sorted(list(must_hit_tokens | nice_hit_tokens)),
        "missing_must": missing_must_tokens,
        "content_top_matches": top_content_matches,
        "exp_required": min_exp,           
        "exp_match": exp_match, 
    }

# ---------------------
# Streamlit UI
# ---------------------
st.title("🔎 CV Filter — Streamlit (Accurate JD Matching)")
st.markdown("Upload JD và nhiều CV, hệ thống sẽ parse & **xếp hạng** theo mức độ phù hợp JD. Hỗ trợ so khớp **kỹ năng** và **nội dung JD** (đa ngôn ngữ).")

with st.sidebar:
    st.header("Job Description (JD)")
    jd_content = st.text_area("JD content", "")
    if jd_content:
        # >>> NEW: dùng extractor chuẩn
        min_ex = extract_min_years_from_jd(jd_content)
        # 2. Tách skills
        must_have_skills = []
        nice_to_have_skills = []
        jd_lower = jd_content.lower()

        for skill in DEFAULT_COMMON_SKILLS:
            if skill in jd_lower:
                # nếu xuất hiện gần từ "must have", "yêu cầu", "required"
                if re.search(r"(must have|required|bắt buộc|yêu cầu).*" + skill, jd_lower):
                    must_have_skills.append(skill)
                # nếu xuất hiện gần từ "ưu tiên", "nice to have", "plus"
                elif re.search(r"(nice to have|ưu tiên|plus).*" + skill, jd_lower):
                    nice_to_have_skills.append(skill)
                else:
                    must_have_skills.append(skill)  # mặc định là must

    else:
        min_ex = 0
        must_have_skills = []
        nice_to_have_skills = []

    st.markdown("---")
    w_kw = st.slider("Weight: Keyword skills", 0.0, 1.0, 0.35, 0.05)
    w_sem = st.slider("Weight: Semantic", 0.0, 1.0, 0.45, 0.05)
    w_exp = st.slider("Weight: Experience", 0.0, 1.0, 0.20, 0.05)

    st.markdown("---")
    use_chunks = st.checkbox("Use chunked semantic (recommended)", value=True)
    max_chunks = st.slider("Max chunks per CV", 1, 24, 12)
    use_ce = st.checkbox("Rerank with Cross-Encoder (if available)", value=False)

    st.markdown("---")
    THRESHOLD = st.slider("Pass threshold", 0.0, 1.0, 0.55, 0.01)
    penalty = st.slider("Penalty when missing must-have", 0.5, 1.0, 0.8, 0.05)

    run_button = st.button("▶️ Run matching")
uploaded_files = st.file_uploader(
    "Upload CV (PDF, DOCX, JPG, PNG) — chọn nhiều file",
    type=["pdf","docx","doc","png","jpg","jpeg"],
    accept_multiple_files=True
)

# store uploaded in-memory
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if uploaded_files:
    st.session_state.candidates = []
    for f in uploaded_files:
        raw = f.read()
        ext = f.name.split(".")[-1].lower()
        text = ""
        try:
            if ext in("pdf",):
                text = extract_text_from_pdf_bytes(raw)
            elif ext in ("docx",):
                text = extract_text_from_docx_bytes(raw)
            elif ext in ("png","jpg","jpeg"):
                text = extract_text_from_image_bytes(raw)
            elif ext in ("doc",):
                # .doc cũ => cố gắng decode thô
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
        except Exception:
            text = ""
        if not text:
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        st.session_state.candidates.append({"filename": f.name, "bytes": raw, "text": text})

# Action: run matching
if "results" not in st.session_state:
    st.session_state.results = []

if run_button:
    st.session_state.results = []
    if not st.session_state.candidates:
        st.warning("Bạn chưa upload CV nào.")
    else:
        jd = {
            "content": jd_content,
            "min_experience_years": int(min_ex),
            "must_have_skills": must_have_skills,
            "nice_to_have_skills": nice_to_have_skills,
        }
        cfg = {
            "w_kw": w_kw,
            "w_sem": w_sem,
            "w_exp": w_exp,
            "use_chunks": use_chunks,
            "max_chunks": max_chunks,
            "use_cross_encoder": use_ce,
            "missing_must_penalty": penalty,
        }
        with st.spinner("Processing CVs and computing scores..."):
            for c in st.session_state.candidates:
                # Giới hạn độ dài để embed nhanh hơn; chunker sẽ xử lý semantic
                txt = c["text"][:80000]
                score = score_candidate(txt, jd, cfg)
                st.session_state.results.append({
                    "id": str(uuid.uuid4()),
                    "filename": c["filename"],
                    "text": txt,
                    "score": score
                })
        st.session_state.results = sorted(st.session_state.results, key=lambda x: x["score"]["total_score"], reverse=True)
        st.success(f"Done — {len(st.session_state.results)} CVs scored.")

# Show results table
# import pandas as pd
import json
import csv
from io import StringIO
import pandas as pd
results = st.session_state.get("results", [])
if results:
    st.subheader("Top candidates")

    # Chuẩn bị dữ liệu cho bảng
    data = []
    for r in results:
        s = r["score"]
        missing_must = s.get("missing_must", [])
        exp_short = ""
        if not s.get("exp_match", True):
            exp_short = f"req {s.get('exp_required',0)}y, has {s.get('exp_years_estimated',0)}y"

        data.append({
            "Candidate": r["filename"],
            "Total": s.get("total_score", 0),
            "Semantic": s.get("semantic_score", 0),
            "JD Match %": f"{s['parsed'].get('jd_match_ratio',0.0)*100:.1f}%",
            "Exp (years)": s.get("exp_years_estimated", 0),
            "Missing Must": ", ".join(missing_must[:3]) + ("…" if len(missing_must) > 3 else ""),
            "Exp Short": exp_short,
            "Email": ";".join(s["parsed"].get("emails", [])[:1])
        })

    df = pd.DataFrame(data)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_columns(["Candidate"], 
                        cellRenderer='''function(params){return `<b style="color:blue;cursor:pointer">${params.value}</b>`}''')
    gridOptions = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
    )

    # Lấy file CV từ session_state.candidates khi click tên
    selected_rows = grid_response.get("selected_rows", [])
    if hasattr(selected_rows, "to_dict"):
        selected_rows = selected_rows.to_dict("records")

    if selected_rows:
        selected = selected_rows[0]
        filename = selected["Candidate"]
        st.write(f"**Selected Candidate:** {filename}")

        cv_entry = next((c for c in st.session_state.candidates if c["filename"] == filename), None)
        if cv_entry:
            ext = filename.split(".")[-1].lower()
            content = cv_entry["bytes"]

            # if ext == "pdf":
            #     # Hiển thị PDF trực tiếp
            #     b64 = base64.b64encode(content).decode()
            #     pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="900" type="application/pdf"></iframe>'
            #     st.markdown(pdf_display, unsafe_allow_html=True)
            # elif ext in ("docx", "doc"):
            #     # Hiển thị text trích xuất từ DOC/DOCX
            #     st.text_area(f"Content of {filename}", cv_entry["text"], height=500)
            # elif ext in ("png", "jpg", "jpeg"):
            #     st.image(content, caption=filename)
            # else:
            #     st.warning("Không hỗ trợ loại file này để mở trực tiếp!")
            if ext == "pdf":
    # Cho phép tải xuống PDF
                st.download_button(
                    label=f"📄 Download {filename}",
                    data=content,
                    file_name=filename,
                    mime="application/pdf"
                )

                # Thử hiển thị inline (có thể bị chặn trên Streamlit Cloud nhưng ok ở local)
                try:
                    b64 = base64.b64encode(content).decode()
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="900" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception:
                    st.warning("Không thể hiển thị PDF trực tiếp, hãy tải về để xem.")
            elif ext in ("docx", "doc"):
                # Hiển thị text trích xuất từ DOC/DOCX
                st.text_area(f"Content of {filename}", cv_entry["text"], height=500)
            elif ext in ("png", "jpg", "jpeg"):
                st.image(content, caption=filename, use_column_width=True)
            else:
                st.warning("Không hỗ trợ loại file này để mở trực tiếp!")
        else:
            st.warning("File CV này chưa được upload hoặc không tồn tại trong session!")
    # Download results CSV
    csv_buf = StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["filename","total","semantic","kw","exp","exp_years","emails","missing_must","jd_match"])
    for r in results:
        s = r["score"]
        writer.writerow([
            r["filename"], s.get("total_score", 0), s.get("semantic_score", 0),
            s.get("kw_score", ""), s.get("exp_score", ""),
            s.get("exp_years_estimated", 0),
            ";".join(s["parsed"].get("emails", [])),
            ";".join(s.get("missing_must", [])),
            f"{s['parsed'].get('jd_match_ratio',0.0)*100:.1f}%"
        ])
    st.download_button("📥 Download results CSV",
                       data=csv_buf.getvalue().encode("utf-8"),
                       file_name="cv_filter_results.csv", mime="text/csv")

    # --- Top CV hợp lệ ---
    valid_results = [
        r for r in results
        if r["score"]["total_score"] >= THRESHOLD
        and (r["score"]["exp_years_estimated"] or 0) >= min_ex
        and r["score"]["parsed"].get("is_skill_match", False)
        and r["score"]["parsed"].get("is_jd_match", False)
    ]

    if valid_results:
        st.markdown(f"### ✅ Top CV hợp lệ — {len(valid_results)} ứng viên")

    # Chuẩn bị dữ liệu cho bảng
    valid_data = []
    for r in valid_results:
        s = r["score"]
        valid_data.append({
            "Candidate": r["filename"],
            "Total": s.get("total_score", 0),
            "Semantic": s.get("semantic_score", 0),
            "JD Match %": f"{s['parsed'].get('jd_match_ratio',0.0)*100:.1f}%",
            "Exp (years)": s.get("exp_years_estimated", 0),
            "Email": ";".join(s["parsed"].get("emails", [])[:1]),
        })

    df_valid = pd.DataFrame(valid_data)

    # Cấu hình bảng với AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_valid)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_columns(["Candidate"],
                         cellRenderer='''function(params){return `<b style="color:blue;cursor:pointer">${params.value}</b>`}''')
    gridOptions = gb.build()

    grid_response = AgGrid(
        df_valid,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
    )

    # Lấy row được chọn
    selected_rows = grid_response.get("selected_rows", [])
    if hasattr(selected_rows, "to_dict"):
        selected_rows = selected_rows.to_dict("records")

    if selected_rows:
        selected = selected_rows[0]
        filename = selected["Candidate"]

        # Lấy CV từ session_state
        cv_entry = next((c for c in st.session_state.candidates if c["filename"] == filename), None)
        if cv_entry:
            ext = filename.split(".")[-1].lower()
            content = cv_entry["bytes"]

            st.markdown(f"**Hiển thị CV: {filename}**")

            # if ext == "pdf":
            #     b64 = base64.b64encode(content).decode()
            #     pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="900" type="application/pdf"></iframe>'
            #     st.markdown(pdf_display, unsafe_allow_html=True)

            # elif ext in ("png", "jpg", "jpeg"):
            #     st.image(content, caption=filename, use_column_width=True)

            # elif ext in ("docx", "doc"):
            #     # Hiển thị text trích xuất từ DOC/DOCX
            #     st.text_area(f"Content of {filename}", cv_entry["text"], height=500)

            # else:
            #     st.warning("Không hỗ trợ loại file này để mở trực tiếp!")
            if ext == "pdf":
    # Cho phép tải xuống PDF
                st.download_button(
                    label=f"📄 Download {filename}",
                    data=content,
                    file_name=filename,
                    mime="application/pdf"
                )

                # Thử hiển thị inline (có thể bị chặn trên Streamlit Cloud nhưng ok ở local)
                try:
                    b64 = base64.b64encode(content).decode()
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="900" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception:
                    st.warning("Không thể hiển thị PDF trực tiếp, hãy tải về để xem.")
            elif ext in ("docx", "doc"):
                # Hiển thị text trích xuất từ DOC/DOCX
                st.text_area(f"Content of {filename}", cv_entry["text"], height=500)
            elif ext in ("png", "jpg", "jpeg"):
                st.image(content, caption=filename, use_column_width=True)
            else:
                st.warning("Không hỗ trợ loại file này để mở trực tiếp!")
        else:
            st.warning("File CV này chưa được upload hoặc không tồn tại trong session!")

    # Export CSV danh sách hợp lệ
    csv_buf = io.StringIO()
    df_valid.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 Download danh sách hợp lệ (.csv)",
        data=csv_buf.getvalue(),
        file_name="top_cv_hople.csv",
        mime="text/csv"
    )

else:
    st.warning("Không có CV nào hợp lệ")
