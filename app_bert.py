import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import random
import io
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM THREAT CLASSIFIER / BERT", layout="wide")

# ── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

  .stApp {
      background-color: #ffffff;
      color: #0000ff;
      font-family: 'Share Tech Mono', 'Courier New', monospace;
  }
  .block-container {
      max-width: 98vw !important;
      padding: 1.5rem 2rem 1.5rem 2rem !important;
  }
  #MainMenu, footer, header { visibility: hidden; }

  /* Quiet labels */
  .stTextArea label, .stFileUploader label, .stNumberInput label {
      color: #8888aa !important;
      font-family: 'Share Tech Mono', monospace !important;
      font-size: 14px !important;
      letter-spacing: 0.1em !important;
      text-transform: uppercase !important;
  }
  /* Textarea */
  .stTextArea textarea {
      background-color: #f4f4ff !important;
      color: #0000ff !important;
      font-family: 'Share Tech Mono', 'Courier New', monospace !important;
      font-size: 16px !important;
      line-height: 1.5 !important;
      border: 2px solid #0000ff !important;
      border-radius: 0 !important;
      padding: 14px !important;
  }
  /* Number input */
  .stNumberInput input {
      border: 2px solid #0000ff !important;
      border-radius: 0 !important;
      color: #0000ff !important;
      font-size: 16px !important;
      font-family: 'Share Tech Mono', monospace !important;
      background-color: #f4f4ff !important;
  }
  /* Buttons */
  .stButton button {
      background-color: #0000ff !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 0 !important;
      font-family: 'Share Tech Mono', monospace !important;
      font-size: 14px !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase !important;
      padding: 10px 18px !important;
  }
  .stButton button:hover { background-color: #0000cc !important; }

  /* Custom tab switcher buttons */
  .tab-btn-active button {
      background-color: #0000ff !important;
      color: #ffffff !important;
      border: 2px solid #0000ff !important;
  }
  .tab-btn-inactive button {
      background-color: #eef !important;
      color: #0000ff !important;
      border: 2px solid #0000ff !important;
  }
  .tab-btn-inactive button:hover {
      background-color: #ddf !important;
  }
  /* File uploader — visible against white */
  [data-testid="stFileUploader"] {
      background-color: #eef !important;
      border: 2px solid #0000ff !important;
      padding: 10px !important;
      color: #0000ff !important;
  }
  [data-testid="stFileUploaderDropzone"] {
      background-color: #eef !important;
      border: 2px dashed #0000ff !important;
      border-radius: 0 !important;
      color: #0000ff !important;
  }
  [data-testid="stFileUploaderDropzone"] * {
      color: #0000ff !important;
  }
  /* Force the browse files button inside uploader to be visible */
  [data-testid="stFileUploader"] button {
      background-color: #0000ff !important;
      color: #ffffff !important;
      font-family: 'Share Tech Mono', monospace !important;
      font-size: 12px !important;
      border: none !important;
      padding: 5px 12px !important;
  }
  [data-testid="stFileUploader"] button * {
      color: #ffffff !important;
  }
  
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-thumb { background: #0000ff; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:3px solid #0000ff;padding-bottom:12px;margin-bottom:18px;
            display:flex;justify-content:space-between;align-items:flex-end;">
  <div>
    <div style="font-size:38px;font-weight:900;letter-spacing:.04em;color:#0000ff;line-height:1;">
      LLM THREAT CLASSIFIER
    </div>
    <div style="font-size:13px;letter-spacing:.18em;color:#8888aa;margin-top:6px;">
      BERT TRANSFORMER &nbsp;/&nbsp; 5-CLASS ADVERSARIAL DETECTION
    </div>
  </div>
  <div style="font-size:13px;letter-spacing:.18em;color:#8888aa;text-align:right;">
    BERT BASE<br>ACC: 96.5%
  </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "best_bert_model")

@st.cache_resource
def load_bert_model():
    """Load the fine-tuned BERT model and label mapping from disk."""
    try:
        map_path = os.path.join(MODEL_DIR, "label_mapping.json")
        if not os.path.exists(map_path):
            map_path = os.path.join(BASE_DIR, "results", "label_mapping.json")
        with open(map_path, "r") as f:
            mapping = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        return tokenizer, model, mapping, True, None
    except Exception as e:
        return None, None, None, False, str(e)

@st.cache_data
def load_test_dataset():
    """Load the held-out test CSV from the data folder."""
    try:
        return pd.read_csv(os.path.join(BASE_DIR, "data", "test.csv"))
    except Exception:
        return None

tokenizer, bert_model, label_mapping, loaded, err_msg = load_bert_model()
test_df = load_test_dataset()

if not loaded:
    st.error(f"Cannot load BERT model: {err_msg}")
    st.stop()

id_to_label   = {int(k): v for k, v in label_mapping["id_to_label"].items()}
total_samples = len(test_df) if test_df is not None else 3000

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "actual_label" not in st.session_state:
    st.session_state.actual_label = ""
if "row_index" not in st.session_state:
    st.session_state.row_index = 0
if "probs" not in st.session_state:
    st.session_state.probs = None
if "result" not in st.session_state:
    st.session_state.result = None

# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────
def extract_text(uploaded_file):
    """Pull plain text out of TXT, CSV, PDF, or image files."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith((".txt", ".csv")):
        text = data.decode("utf-8", errors="ignore").strip()
        return text, f"Loaded {uploaded_file.name} ({len(text)} chars)", True

    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
            if pages:
                full = "\n\n".join(pages)
                return full, f"PDF: {len(reader.pages)} pages, {len(full)} chars", True
        except Exception:
            pass
        try:
            raw     = data.decode("latin1", errors="ignore")
            matches = re.findall(r"\((.*?)\)\s*T[jJ]", raw)
            text    = " ".join(matches).strip()
            if len(text) > 5:
                return text, f"PDF stream: {len(text)} chars", True
        except Exception:
            pass
        return "", "No readable text in PDF.", False

    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(data))
        except Exception as e:
            return "", f"Bad image: {e}", False
        for engine in ("tesseract", "easyocr"):
            try:
                if engine == "tesseract":
                    import pytesseract
                    text = pytesseract.image_to_string(img).strip()
                else:
                    import easyocr
                    r    = easyocr.Reader(["en"], gpu=False)
                    text = " ".join([x[1] for x in r.readtext(data)]).strip()
                if text:
                    return text, f"OCR ({engine}): {len(text)} chars", True
            except Exception:
                continue
        return "", "No OCR engine available.", False

    return "", "Unsupported format.", False


# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — Custom tab switcher (no st.tabs — avoids Streamlit opacity bug)
# ════════════════════════════════════════════════════════════════════════════
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "dataset"

# Tab header buttons
tab_col1, tab_col2, tab_spacer = st.columns([1.2, 1.2, 8], vertical_alignment="bottom")
with tab_col1:
    cls1 = "tab-btn-active" if st.session_state.active_tab == "dataset" else "tab-btn-inactive"
    st.markdown(f'<div class="{cls1}">', unsafe_allow_html=True)
    if st.button("TEST DATASET", key="tab_dataset", use_container_width=True):
        st.session_state.active_tab   = "dataset"
        st.session_state.prompt       = ""
        st.session_state.actual_label = ""
        st.session_state.probs        = None
        st.session_state.result       = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
with tab_col2:
    cls2 = "tab-btn-active" if st.session_state.active_tab == "upload" else "tab-btn-inactive"
    st.markdown(f'<div class="{cls2}">', unsafe_allow_html=True)
    if st.button("UPLOAD FILE", key="tab_upload", use_container_width=True):
        st.session_state.active_tab   = "upload"
        st.session_state.prompt       = ""
        st.session_state.actual_label = ""
        st.session_state.probs        = None
        st.session_state.result       = None
        st.session_state.pop('uploaded_file_id', None)
        st.session_state.pop('upload_msg', None)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Blue underline separating tabs from content
st.markdown('<div style="border-top:2px solid #0000ff;margin:0 0 14px 0;"></div>', unsafe_allow_html=True)

# ── Tab content ──────────────────────────────────────────────────────────────
if st.session_state.active_tab == "dataset":
    ci, cb1, cb2, cgt = st.columns([2, 0.8, 0.8, 3], vertical_alignment="bottom")
    with ci:
        idx = st.number_input(
            f"Row index (0 – {total_samples - 1})",
            min_value=0, max_value=total_samples - 1,
            value=st.session_state.row_index, step=1
        )
    with cb1:
        if st.button("LOAD", use_container_width=True):
            row = test_df.iloc[idx]
            st.session_state.prompt       = str(row["prompt"])
            st.session_state.actual_label = str(row["label"])
            st.session_state.row_index    = int(idx)
            st.session_state.probs        = None
            st.session_state.result       = None
            st.rerun()
    with cb2:
        if st.button("RANDOM", use_container_width=True):
            rand = random.randint(0, total_samples - 1)
            row  = test_df.iloc[rand]
            st.session_state.prompt       = str(row["prompt"])
            st.session_state.actual_label = str(row["label"])
            st.session_state.row_index    = rand
            st.session_state.probs        = None
            st.session_state.result       = None
            st.rerun()
    with cgt:
        if st.session_state.actual_label:
            st.markdown(
                f'<div style="background:#000000;color:#fff;padding:10px 20px;'
                f'display:inline-block;font-size:15px;letter-spacing:.06em;">'
                f'<span style="font-size:12px;opacity:.8;">ROW #{st.session_state.row_index} &nbsp;/&nbsp; GROUND TRUTH &nbsp;</span><br>'
                f'<strong style="font-size:22px;">{st.session_state.actual_label.upper()}</strong></div>',
                unsafe_allow_html=True
            )

else:  # upload tab
    cu1, cu2 = st.columns([2, 3], vertical_alignment="bottom")
    with cu1:
        file = st.file_uploader(
            "PDF / image / TXT",
            type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv"],
            label_visibility="collapsed"
        )
    with cu2:
        if file:
            # Auto-load text on new file
            if st.session_state.get('uploaded_file_id') != file.file_id:
                ext_text, msg, ok = extract_text(file)
                st.session_state['uploaded_file_id'] = file.file_id
                st.session_state['upload_msg'] = msg
                if ok and ext_text:
                    st.session_state.prompt       = ext_text
                    st.session_state.actual_label = ""
                    st.session_state.probs        = None
                    st.session_state.result       = None
                    st.rerun()
            # Display the result message
            msg = st.session_state.get('upload_msg', "")
            if msg:
                st.markdown(
                    f'<div style="border-left:4px solid #0000ff;padding:8px 14px;font-size:14px;background:#eef;color:#0000ff;">{msg}</div>',
                    unsafe_allow_html=True
                )

# Thin divider before main panel
st.markdown('<div style="border-top:2px solid #0000ff;margin:14px 0 16px 0;"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — Two columns: [Prompt editor + Scan] | [Result + Probabilities]
# ════════════════════════════════════════════════════════════════════════════
col_prompt, col_result = st.columns([1, 1], gap="large")

# ── LEFT: Prompt editor ───────────────────────────────────────────────────────
with col_prompt:
    st.markdown('<div style="font-size:13px;letter-spacing:.22em;color:#8888aa;margin-bottom:6px;">PROMPT</div>', unsafe_allow_html=True)
    user_prompt = st.text_area(
        "Editable prompt",
        value=st.session_state.prompt,
        height=320,
        label_visibility="collapsed",
        placeholder="Type or paste an adversarial prompt here..."
    )
    scan = st.button("▶  SCAN THREAT / BERT", use_container_width=True)

    if scan:
        if not user_prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = bert_model(**inputs).logits
                probs  = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

            top_idx   = int(np.argmax(probs))
            top_label = id_to_label[top_idx]
            top_conf  = probs[top_idx] * 100.0

            actual    = st.session_state.actual_label
            unchanged = (user_prompt.strip() == st.session_state.prompt.strip())
            if unchanged and actual:
                is_correct = (top_label == actual)
                verdict    = "✓  CORRECT" if is_correct else "✗  WRONG"
                vbg        = "#008800"    if is_correct else "#dd0000"
            else:
                verdict, vbg = "—  CUSTOM INPUT", "#666666"

            descriptions = {
                "jailbreak":        "Persona / roleplay exploit bypassing content filters.",
                "prompt_injection":  "Instruction hijack overriding the system prompt.",
                "harmful_behavior":  "Direct request for dangerous or illegal content.",
                "toxicity":          "Hate speech, harassment, or profanity.",
                "linguistic":        "Subtle semantic framing or social engineering.",
            }

            st.session_state.probs  = probs
            st.session_state.result = {
                "top_idx":   top_idx,
                "top_label": top_label,
                "top_conf":  top_conf,
                "verdict":   verdict,
                "vbg":       vbg,
                "desc":      descriptions.get(top_label, ""),
            }
            st.rerun()

# ── RIGHT: Result + Probability bars ─────────────────────────────────────────
with col_result:
    res   = st.session_state.result
    probs = st.session_state.probs

    if res and probs is not None:
        # Classification block
        st.markdown(f"""
<div style="background:#0000ff;color:#fff;padding:18px 22px;margin-bottom:12px;">
  <div style="font-size:12px;letter-spacing:.22em;opacity:.8;">CLASSIFICATION</div>
  <div style="font-size:36px;font-weight:900;letter-spacing:.04em;line-height:1.2;">{res["top_label"].upper()}</div>
  <div style="font-size:20px;margin-top:4px;">CONFIDENCE: {res["top_conf"]:.2f}%</div>
</div>
""", unsafe_allow_html=True)

        # Verdict block — big coloured banner
        st.markdown(f"""
<div style="background:{res["vbg"]};color:#fff;padding:14px 22px;
            font-size:24px;font-weight:900;letter-spacing:.08em;margin-bottom:12px;">
  {res["verdict"]}
</div>
""", unsafe_allow_html=True)

        # Description
        st.markdown(
            f'<div style="font-size:14px;color:#555;border-left:4px solid #0000ff;'
            f'padding-left:12px;margin-bottom:18px;">{res["desc"]}</div>',
            unsafe_allow_html=True
        )

        # Probability bars
        st.markdown('<div style="font-size:12px;letter-spacing:.2em;color:#8888aa;margin-bottom:8px;">PROBABILITIES</div>', unsafe_allow_html=True)
        for i, label in id_to_label.items():
            pct    = probs[i] * 100.0
            filled = int(pct / 4)
            bar    = "█" * filled + "░" * (25 - filled)
            is_top = (i == res["top_idx"])
            bg, fg, wt = ("#0000ff", "#fff", "900") if is_top else ("#eef", "#0000ff", "400")
            st.markdown(f"""
<div style="background:{bg};color:{fg};padding:8px 14px;margin-bottom:5px;
            font-family:'Share Tech Mono',monospace;">
  <span style="font-size:13px;letter-spacing:.1em;font-weight:{wt};">{label.upper()}</span>
  &nbsp;&nbsp;
  <span style="font-size:16px;font-weight:900;">{pct:.1f}%</span>
  &nbsp;<span style="font-size:12px;">{bar}</span>
</div>
""", unsafe_allow_html=True)

    else:
        # Taxonomy shown before first scan
        st.markdown('<div style="font-size:12px;letter-spacing:.22em;color:#8888aa;margin-bottom:10px;">THREAT TAXONOMY</div>', unsafe_allow_html=True)
        for cat, desc in {
            "JAILBREAK":        "Persona / DAN filter evasion",
            "PROMPT INJECTION":  "System instruction hijack",
            "HARMFUL BEHAVIOR":  "Direct hazardous requests",
            "TOXICITY":          "Hate speech and harassment",
            "LINGUISTIC":        "Social engineering / evasion",
        }.items():
            st.markdown(f"""
<div style="border-left:4px solid #0000ff;padding:8px 14px;margin-bottom:8px;background:#eef;">
  <div style="font-size:14px;font-weight:900;letter-spacing:.1em;color:#0000ff;">{cat}</div>
  <div style="font-size:12px;color:#555;margin-top:2px;">{desc}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:13px;color:#8888aa;margin-top:20px;letter-spacing:.06em;">'
            'LOAD A SAMPLE OR UPLOAD A FILE,<br>'
            'THEN CLICK <strong style="color:#0000ff;">▶ SCAN THREAT</strong> TO CLASSIFY.</div>',
            unsafe_allow_html=True
        )
