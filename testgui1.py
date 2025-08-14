# testgui1.py
# Streamlit CV Filter — full version
# - Giữ tính năng semantic chunking, cross-encoder (tùy chọn)
# - Phân loại must-have / nice-to-have, phạt khi thiếu must-have
# - Ước lượng kinh nghiệm (year ranges + simple years)
# - OCR (pdf scan / ảnh) bằng pytesseract fallback
# - So khớp nội dung JD theo dòng + highlight các skill trùng
# Mở thư mục chứa file testgui1.py
cd /path/to/your/project

# Khởi tạo Git repo
git init

# Thêm toàn bộ file vào repo
git add testgui1.py requirements.txt

# Commit lần đầu
git commit -m "Initial commit for CV Filter Streamlit app"

# Kết nối tới GitHub repo vừa tạo
git remote add origin https://github.com/<username>/cv-filter-streamlit.git

# Push code lên GitHub
git push -u origin main
import streamlit as st
import io, os, re, json, uuid, unicodedata, math
from typing import List, Dict, Any, Tuple
from PIL import Image
import pdfplumber
from docx import Document
from dotenv import load_dotenv
import pytesseract
import numpy as np
from datetime import datetime

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
    """So khớp từng dòng CV với toàn bộ nội dung JD (đa ngôn ngữ)."""
    jd_tokens = set(preprocess_multilang(jd_text))
    results = []
    if not jd_tokens:
        return results
    for line in (cv_text or "").splitlines():
        line_tokens = set(preprocess_multilang(line))
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
JD_MATCH_THRESHOLD = 0.8
def parse_structured_info(text: str, jd_text: str, must_have_skills: list, nice_to_have_skills: list) -> dict:
    parsed = {}
    emails = EMAIL_RE.findall(text or "")
    phones = PHONE_RE.findall(text or "")

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

    # Tokens and skill matching
    cv_tokens = set(preprocess_multilang(text or ""))
    must_tokens = set(preprocess_multilang(" ".join(must_have_skills)))
    nice_tokens = set(preprocess_multilang(" ".join(nice_to_have_skills)))
    jd_tokens = set(preprocess_multilang(jd_text or ""))

    parsed["must_hit_tokens"] = sorted(must_tokens & cv_tokens)
    parsed["nice_hit_tokens"] = sorted(nice_tokens & cv_tokens)
    parsed["missing_must_tokens"] = sorted(must_tokens - cv_tokens)

    # JD content match ratio (overall tokens)
    parsed["jd_match_ratio"] = round(len(jd_tokens & cv_tokens) / max(1, len(jd_tokens)), 3) if jd_tokens else 0.0
    parsed["is_jd_match"] = parsed["jd_match_ratio"] >= JD_MATCH_THRESHOLD
    parsed["is_skill_match"] = len(parsed["missing_must_tokens"]) == 0

    return parsed

# --- Tính semantic similarity với chunk ---
def semantic_similarity(cv_text: str, jd_text: str, use_chunks: bool = True, max_chunks: int = 12) -> float:
    if embedder is None or util is None:
        return 0.0
    if not use_chunks:
        emb_cv = embedder.encode(cv_text or "", convert_to_tensor=True)
        emb_jd = embedder.encode(jd_text or "", convert_to_tensor=True)
        sem_sim = float(util.cos_sim(emb_cv, emb_jd).item())
        return (sem_sim + 1) / 2

    chunks = chunk_text(cv_text or "", max_chunks=max_chunks)
    jd_emb = embedder.encode(jd_text or "", convert_to_tensor=True)
    sims = []
    for ch in chunks:
        emb = embedder.encode(ch, convert_to_tensor=True)
        sims.append(float(util.cos_sim(emb, jd_emb).item()))
    if not sims:
        return 0.0
    max_sim = max(sims)  # max-over-chunks
    return (max_sim + 1) / 2

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
    min_exp = int(jd.get("min_experience_years", 0) or 0)
    exp_score = 1.0 if exp_years >= min_exp else (exp_years / max(1, min_exp)) if min_exp > 0 else 1.0

    # Combine weights
    w_kw, w_sem, w_exp = cfg.get("w_kw", 0.35), cfg.get("w_sem", 0.45), cfg.get("w_exp", 0.2)
    w_sum = max(1e-6, w_kw + w_sem + w_exp)
    w_kw, w_sem, w_exp = w_kw/w_sum, w_sem/w_sum, w_exp/w_sum
    total = w_kw*kw_score + w_sem*sem_score + w_exp*exp_score

    # penalty when missing must-have
    if missing_must_tokens:
        total *= cfg.get("missing_must_penalty", 0.6)

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
    }

# ---------------------
# Streamlit UI
# ---------------------
st.title("🔎 CV Filter — Streamlit (Accurate JD Matching)")
st.markdown("Upload JD và nhiều CV, hệ thống sẽ parse & **xếp hạng** theo mức độ phù hợp JD. Hỗ trợ so khớp **kỹ năng** và **nội dung JD** (đa ngôn ngữ).")

with st.sidebar:
    st.header("Job Description (JD)")
    jd_content = st.text_area("JD content", "")
    min_ex = st.number_input("Min experience (years)", min_value=0, max_value=50, value=2)
    must_raw = st.text_input("Must-have skills (comma)", "")
    nice_raw = st.text_input("Nice-to-have skills (comma)", "")

    st.markdown("---")
    st.subheader("Scoring Weights")
    w_kw = st.slider("Weight: Keyword skills", 0.0, 1.0, 0.35, 0.05)
    w_sem = st.slider("Weight: Semantic", 0.0, 1.0, 0.45, 0.05)
    w_exp = st.slider("Weight: Experience", 0.0, 1.0, 0.20, 0.05)

    st.markdown("---")
    st.subheader("Semantic Options")
    use_chunks = st.checkbox("Use chunked semantic (recommended)", value=True)
    max_chunks = st.slider("Max chunks per CV", 1, 24, 12)
    use_ce = st.checkbox("Rerank with Cross-Encoder (if available)", value=False, help="Cần tải model lần đầu")

    st.markdown("---")
    THRESHOLD = st.slider("Pass threshold (total score)", 0.0, 1.0, 0.55, 0.01)
    penalty = st.slider("Penalty when missing must-have (multiplier)", 0.2, 1.0, 0.6, 0.05)

    run_button = st.button("▶️ Run matching")

must_have_skills = [s.strip() for s in must_raw.split(",") if s.strip()]
nice_to_have_skills = [s.strip() for s in nice_raw.split(",") if s.strip()]

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
            if ext in ("pdf",):
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
results = st.session_state.get("results", [])
if results:
    st.subheader("Top candidates")
    # Header
    cols = st.columns((4,1,1,1))
    cols[0].markdown("**Candidate**")
    cols[1].markdown("**Total**")
    cols[2].markdown("**Semantic**")
    cols[3].markdown("**JD Match %**")

    for r in results:
        c0, c1, c2, c3 = st.columns((4,1,1,1))
        missing_must = r["score"].get("missing_must_tokens", [])
        badge = ""
        if missing_must:
            # hiện tối đa 3 token còn thiếu
            badge = f"<span style='background:#fde2e1;color:#b42318;padding:2px 6px;border-radius:999px;font-size:11px'>Missing must: {', '.join(missing_must[:3])}{'…' if len(missing_must)>3 else ''}</span>"
        c0.markdown(f"**{r['filename']}**  {badge}", unsafe_allow_html=True)
        # show email small
        c0.caption(", ".join(r["score"]["parsed"].get("emails", [])[:1]) or "")
        c1.markdown(r["score"]["total_score"])
        c2.markdown(r["score"]["semantic_score"])
        c3.markdown(f"{r['score']['parsed'].get('jd_match_ratio', 0.0)*100:.1f}%")

        with c0.expander("Details / Preview"):
            st.write("Estimated years:", r["score"]["exp_years_estimated"])

            # skill hit tokens (multi-lang)
            st.write("Must-hit skills (tokens):", r["score"].get("must_hit_tokens", []))
            st.write("Nice-hit skills (tokens):", r["score"].get("nice_hit_tokens", []))

            st.write("Emails:", r["score"]["parsed"].get("emails", []))
            st.write("Phones:", r["score"]["parsed"].get("phones", []))

            # Top dòng CV khớp nội dung JD
            top_lines = r["score"].get("content_top_matches", [])
            if top_lines:
                st.markdown("**Top CV lines matching JD content:**")
                for m in top_lines:
                    st.markdown(f"- _{m['line']}_  → **{m['match_ratio']*100:.1f}%**  *(match: {', '.join(m['matched_words'])})*")

            st.markdown("---")
            # highlight các kỹ năng JD trong preview (dựa trên danh sách raw để dễ nhìn)
            preview_terms = list(set(must_have_skills + nice_to_have_skills))
            highlighted = highlight_terms(r["text"][:2000], preview_terms)
            st.markdown("<div style='white-space:pre-wrap;'>" + highlighted + "</div>", unsafe_allow_html=True)

    # Download results JSON/CSV
    out_json = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button("📥 Download results JSON", data=out_json, file_name="cv_filter_results.json", mime="application/json")

    import csv
    from io import StringIO
    csv_buf = StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["filename","total","semantic","kw","exp","exp_years","emails","missing_must","jd_match"])
    for r in results:
        s = r["score"]
        writer.writerow([
            r["filename"], s["total_score"], s["semantic_score"], s["kw_score"] if "kw_score" in s else "",
            s["exp_score"] if "exp_score" in s else "", s["exp_years_estimated"],
            ";".join(s["parsed"].get("emails", [])), ";".join(s.get("missing_must", [])),
            f"{s['parsed'].get('jd_match_ratio',0.0)*100:.1f}%"
        ])
    st.download_button("📥 Download results CSV", data=csv_buf.getvalue().encode("utf-8"), file_name="cv_filter_results.csv", mime="text/csv")

    # Danh sách hợp lệ — yêu cầu: score >= THRESHOLD, exp >= min_ex, đủ must-have, JD match >= JD_MATCH_THRESHOLD
    valid_results = [
        r for r in results
        if r["score"]["total_score"] >= THRESHOLD
        and (r["score"]["exp_years_estimated"] or 0) >= min_ex
        and r["score"]["parsed"].get("is_skill_match", False)
        and r["score"]["parsed"].get("is_jd_match", False)
    ]

    if valid_results:
        st.markdown(f"### ✅ Top CV hợp lệ — {len(valid_results)} ứng viên")
        valid_list_str = "\n".join(
            [f"- {r['filename']} (score={r['score']['total_score']})" for r in valid_results]
        )
        st.text(valid_list_str)
        st.download_button(
            "📥 Download danh sách hợp lệ (.txt)",
            data=valid_list_str,
            file_name="top_cv_hople.txt",
            mime="text/plain"
        )
    else:
        st.warning("Không có CV nào hợp lệ ")

# Notes & next steps
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Gợi ý:** Nếu dữ liệu cực lớn (hàng chục nghìn CV), hãy build chỉ mục FAISS cho embeddings CV,\n"
    "tìm top-k gần JD trước rồi mới chấm điểm chi tiết để tăng tốc. FAISS tăng tốc — không thay đổi độ chính xác."
)

