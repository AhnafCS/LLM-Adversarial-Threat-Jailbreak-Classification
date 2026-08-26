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
  /* Frutiger Aero / Windows 7 Glass Aesthetic */
  .stApp {
      background: linear-gradient(135deg, #a4d2f0 0%, #e6f2fb 50%, #ffffff 100%);
      background-attachment: fixed;
      color: #333333;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  .block-container {
      max-width: 98vw !important;
      padding: 1.5rem 2rem 2rem 2rem !important;
      background: rgba(255, 255, 255, 0.4);
      border-radius: 15px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      margin-top: 1rem;
      border: 1px solid rgba(255, 255, 255, 0.8);
      backdrop-filter: blur(10px);
  }
  #MainMenu, footer, header { visibility: hidden; }

  /* Quiet labels */
  .stTextArea label, .stFileUploader label, .stNumberInput label {
      color: #306ca0 !important;
      font-weight: 600 !important;
      font-size: 14px !important;
      text-transform: uppercase !important;
      letter-spacing: 1px !important;
      text-shadow: 1px 1px 0px rgba(255, 255, 255, 0.8);
  }
  
  /* Textarea */
  .stTextArea textarea {
      background-color: rgba(255, 255, 255, 0.8) !important;
      color: #333333 !important;
      font-size: 15px !important;
      line-height: 1.5 !important;
      border: 1px solid #99cbee !important;
      border-radius: 8px !important;
      padding: 14px !important;
      box-shadow: inset 0px 2px 4px rgba(0,0,0,0.05);
  }
  .stTextArea textarea:focus {
      border: 1px solid #4a9be0 !important;
      box-shadow: 0 0 8px rgba(74, 155, 224, 0.5), inset 0px 2px 4px rgba(0,0,0,0.05) !important;
  }
  
  /* Number input */
  .stNumberInput input {
      background-color: rgba(255, 255, 255, 0.8) !important;
      border: 1px solid #99cbee !important;
      border-radius: 8px !important;
      color: #333333 !important;
      font-size: 15px !important;
      box-shadow: inset 0px 2px 4px rgba(0,0,0,0.05);
  }
  .stNumberInput input:focus {
      border: 1px solid #4a9be0 !important;
      box-shadow: 0 0 8px rgba(74, 155, 224, 0.5), inset 0px 2px 4px rgba(0,0,0,0.05) !important;
  }

  /* Glossy Buttons (Windows 7 / Aero style) */
  .stButton button {
      background: linear-gradient(to bottom, #dbeaf9 0%, #9bc2ea 49%, #6ca8e0 50%, #4a90d6 100%) !important;
      color: #ffffff !important;
      border: 1px solid #3c77b4 !important;
      border-radius: 20px !important; /* pill shape */
      font-size: 14px !important;
      font-weight: 700 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.1em !important;
      padding: 8px 18px !important;
      box-shadow: 0px 2px 5px rgba(0,0,0,0.2), inset 0px 1px 2px rgba(255,255,255,0.8) !important;
      text-shadow: 1px 1px 1px rgba(0,0,0,0.3) !important;
      transition: all 0.2s ease !important;
  }
  .stButton button:hover { 
      background: linear-gradient(to bottom, #eef5fc 0%, #b8d4f0 49%, #8bbdec 50%, #68a5e0 100%) !important;
      box-shadow: 0px 3px 6px rgba(0,0,0,0.3), inset 0px 1px 2px rgba(255,255,255,0.9), 0px 0px 10px rgba(108,168,224,0.6) !important;
  }
  .stButton button:active {
      background: linear-gradient(to bottom, #4a90d6 0%, #6ca8e0 49%, #9bc2ea 50%, #dbeaf9 100%) !important;
      box-shadow: inset 0px 2px 5px rgba(0,0,0,0.4) !important;
  }

  /* Custom tab switcher buttons */
  .tab-btn-active button {
      background: linear-gradient(to bottom, #8bbdec 0%, #4a9be0 49%, #2884d6 50%, #1c72c2 100%) !important;
      color: #ffffff !important;
      border: 1px solid #145594 !important;
      border-radius: 20px !important;
      box-shadow: 0px 2px 5px rgba(0,0,0,0.2), inset 0px 1px 1px rgba(255,255,255,0.5) !important;
  }
  .tab-btn-active button:hover {
      background: linear-gradient(to bottom, #a0cbf0 0%, #68afe5 49%, #4396dc 50%, #2f83cf 100%) !important;
  }
  .tab-btn-inactive button {
      background: linear-gradient(to bottom, #f7fbfd 0%, #e2eef7 49%, #c9e0f2 50%, #bedbf0 100%) !important;
      color: #3b7cae !important;
      border: 1px solid #a3c7e6 !important;
      border-radius: 20px !important;
      box-shadow: 0px 1px 3px rgba(0,0,0,0.1), inset 0px 1px 2px rgba(255,255,255,0.9) !important;
      text-shadow: 1px 1px 0px rgba(255,255,255,0.8) !important;
  }
  .tab-btn-inactive button:hover {
      background: linear-gradient(to bottom, #ffffff 0%, #eff6fb 49%, #dfedf8 50%, #d5e9f6 100%) !important;
  }
  
  /* File uploader */
  [data-testid="stFileUploader"] {
      background: rgba(255,255,255,0.6) !important;
      border: 1px solid #99cbee !important;
      border-radius: 12px !important;
      padding: 10px !important;
      color: #306ca0 !important;
      box-shadow: inset 0px 1px 3px rgba(0,0,0,0.05);
  }
  [data-testid="stFileUploaderDropzone"] {
      background: rgba(230, 242, 251, 0.4) !important;
      border: 2px dashed #7bc3eb !important;
      border-radius: 10px !important;
      color: #306ca0 !important;
  }
  [data-testid="stFileUploaderDropzone"] * {
      color: #306ca0 !important;
  }
  /* Browse files button inside uploader uses glossy button style too */
  [data-testid="stFileUploader"] button {
      background: linear-gradient(to bottom, #dbeaf9 0%, #9bc2ea 49%, #6ca8e0 50%, #4a90d6 100%) !important;
      color: #ffffff !important;
      font-weight: bold !important;
      border: 1px solid #3c77b4 !important;
      border-radius: 15px !important;
      padding: 5px 12px !important;
      box-shadow: 0px 1px 3px rgba(0,0,0,0.2), inset 0px 1px 2px rgba(255,255,255,0.8) !important;
      text-shadow: 1px 1px 1px rgba(0,0,0,0.3) !important;
  }
  [data-testid="stFileUploader"] button:hover {
      background: linear-gradient(to bottom, #eef5fc 0%, #b8d4f0 49%, #8bbdec 50%, #68a5e0 100%) !important;
  }
  [data-testid="stFileUploader"] button * {
      color: #ffffff !important;
  }
  
  /* Webkit Scrollbar Windows 7 style */
  ::-webkit-scrollbar { width: 12px; }
  ::-webkit-scrollbar-track { background: #e6f0fa; border-left: 1px solid #cce2f4; }
  ::-webkit-scrollbar-thumb { 
      background: linear-gradient(to right, #d3e7f8, #a8cceb); 
      border: 1px solid #8cbde1; 
      border-radius: 6px; 
  }
  ::-webkit-scrollbar-thumb:hover { background: linear-gradient(to right, #eaf3fb, #b9d8f0); }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom: 2px solid #a8d0ef; padding-bottom: 15px; margin-bottom: 20px;
            display: flex; justify-content: space-between; align-items: flex-end;
            background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.6), rgba(255,255,255,0));
            padding: 15px; border-radius: 10px;">
  <div>
    <div style="font-size: 38px; font-weight: 800; color: #206ab0; line-height: 1; text-shadow: 2px 2px 0px #ffffff, 0px 4px 6px rgba(0,0,0,0.1); letter-spacing: 1px;">
      LLM THREAT CLASSIFIER
    </div>
    <div style="font-size: 14px; font-weight: 600; color: #438bc9; margin-top: 8px; letter-spacing: 0.1em; text-shadow: 1px 1px 0px #ffffff;">
      BERT TRANSFORMER &nbsp;/&nbsp; 5-CLASS ADVERSARIAL DETECTION
    </div>
  </div>
  <div style="font-size: 14px; font-weight: 700; color: #438bc9; text-align: right; text-shadow: 1px 1px 0px #ffffff; background: rgba(255,255,255,0.7); padding: 8px 15px; border-radius: 20px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1), 0 1px 0 rgba(255,255,255,1);">
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
# ROW 1 — Custom tab switcher
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

# Aero style underline separating tabs from content (REMOVED)

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
                f'<div style="background: linear-gradient(to bottom, #ffffff, #e6f0fa); color: #1c5e9c; padding: 8px 18px; '
                f'display: inline-flex; align-items: center; border-radius: 20px; border: 1px solid #99cbee; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px;">'
                f'<span style="font-size: 11px; font-weight: 700; text-shadow: 1px 1px 0 #fff; margin-right: 8px; letter-spacing: 0.05em;">ROW #{st.session_state.row_index} &nbsp;|&nbsp; GROUND TRUTH:</span>'
                f'<strong style="font-size: 14px; text-shadow: 1px 1px 2px rgba(255,255,255,0.8); letter-spacing: 0.05em;">{st.session_state.actual_label.upper()}</strong></div>',
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
                    f'<div style="border-left: 4px solid #4a9be0; padding: 10px 16px; font-size: 14px; font-weight: 600; '
                    f'background: linear-gradient(to right, #e2f0fb, rgba(255,255,255,0.5)); color: #1c5e9c; border-radius: 0 8px 8px 0; '
                    f'box-shadow: 0 1px 3px rgba(0,0,0,0.05);">{msg}</div>',
                    unsafe_allow_html=True
                )

# Divider before main panel (REMOVED)

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — Two columns: [Prompt editor + Scan] | [Result + Probabilities]
# ════════════════════════════════════════════════════════════════════════════
col_prompt, col_result = st.columns([1, 1], gap="large")

# ── LEFT: Prompt editor ───────────────────────────────────────────────────────
with col_prompt:
    st.markdown('<div style="font-size: 14px; font-weight: bold; letter-spacing: 1px; color: #306ca0; margin-bottom: 12px; margin-top: 5px; text-shadow: 1px 1px 0 #fff;">PROMPT</div>', unsafe_allow_html=True)
    user_prompt = st.text_area(
        "Editable prompt",
        value=st.session_state.prompt,
        height=320,
        label_visibility="collapsed",
        placeholder="Type or paste an adversarial prompt here..."
    )
    scan = st.button("SCAN THREAT / BERT", use_container_width=True)

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
                verdict    = "CORRECT" if is_correct else "WRONG"
                
                # Frutiger green / red glossy backgrounds
                vbg = "linear-gradient(to bottom, #a8e063, #56ab2f)" if is_correct else "linear-gradient(to bottom, #ff9999, #e63946)"
                border_col = "#4caf50" if is_correct else "#c62828"
            else:
                verdict = "CUSTOM INPUT"
                vbg = "linear-gradient(to bottom, #f0f0f0, #c0c0c0)"
                border_col = "#999999"

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
                "border_col": border_col,
                "desc":      descriptions.get(top_label, ""),
            }
            st.rerun()

# ── RIGHT: Result + Probability bars ─────────────────────────────────────────
with col_result:
    res   = st.session_state.result
    probs = st.session_state.probs

    if res and probs is not None:
        # Classification block (Glossy Blue)
        st.markdown(f"""
<div style="background: linear-gradient(to bottom, #74b9ff, #0984e3); color: #ffffff; padding: 15px 20px; margin-bottom: 10px; 
            border-radius: 15px; border: 1px solid #005f9e; box-shadow: 0 4px 10px rgba(9, 132, 227, 0.4), inset 0 1px 2px rgba(255,255,255,0.8);">
  <div style="font-size: 13px; font-weight: bold; letter-spacing: 0.15em; text-shadow: 1px 1px 1px rgba(0,0,0,0.2);">CLASSIFICATION</div>
  <div style="font-size: 38px; font-weight: 800; letter-spacing: 0.05em; line-height: 1.2; text-shadow: 1px 2px 3px rgba(0,0,0,0.3); margin-top: 5px;">{res["top_label"].upper()}</div>
  <div style="font-size: 20px; font-weight: 600; margin-top: 8px; text-shadow: 1px 1px 1px rgba(0,0,0,0.2);">CONFIDENCE: {res["top_conf"]:.2f}%</div>
</div>
""", unsafe_allow_html=True)

        # Verdict block
        st.markdown(f"""
<div style="background: {res["vbg"]}; color: #ffffff; padding: 10px 20px; border-radius: 10px; border: 1px solid {res["border_col"]};
            font-size: 22px; font-weight: 800; letter-spacing: 0.05em; margin-bottom: 10px; 
            box-shadow: 0 3px 8px rgba(0,0,0,0.2), inset 0 1px 2px rgba(255,255,255,0.7); text-shadow: 1px 1px 2px rgba(0,0,0,0.4);">
  {res["verdict"]}
</div>
""", unsafe_allow_html=True)

        # Probability bars
        st.markdown('<div style="font-size: 14px; font-weight: bold; letter-spacing: 1px; color: #306ca0; margin-bottom: 12px; text-shadow: 1px 1px 0 #fff;">PROBABILITIES</div>', unsafe_allow_html=True)
        for i, label in id_to_label.items():
            pct    = probs[i] * 100.0
            
            is_top = (i == res["top_idx"])
            if is_top:
                bg = "linear-gradient(to right, #4a9be0, #2884d6)"
                txt_col = "#ffffff"
                border = "1px solid #1c72c2"
                shadow = "0 2px 5px rgba(40,132,214,0.4), inset 0 1px 1px rgba(255,255,255,0.5)"
            else:
                bg = "linear-gradient(to right, #f0f7fc, #e2eef7)"
                txt_col = "#3b7cae"
                border = "1px solid #bde0fa"
                shadow = "0 1px 2px rgba(0,0,0,0.05), inset 0 1px 2px rgba(255,255,255,0.9)"

            st.markdown(f"""
<div style="background: {bg}; color: {txt_col}; padding: 12px 14px; margin-bottom: 10px; border-radius: 8px; border: {border}; box-shadow: {shadow}; display: flex; align-items: center; justify-content: space-between;">
  <span style="font-size: 14px; font-weight: {'800' if is_top else '600'}; text-shadow: {'1px 1px 1px rgba(0,0,0,0.2)' if is_top else '1px 1px 0 #fff'};">{label.upper()}</span>
  <div style="display: flex; align-items: center; gap: 12px;">
      <span style="font-size: 16px; font-weight: 800; text-shadow: {'1px 1px 1px rgba(0,0,0,0.2)' if is_top else '1px 1px 0 #fff'};">{pct:.1f}%</span>
      <div style="width: 100px; height: 12px; background: rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
          <div style="width: {pct}%; height: 100%; background: {'linear-gradient(to bottom, #a0e0ff, #56baf0)' if is_top else 'linear-gradient(to bottom, #a0cbf0, #68afe5)'}; border-radius: 6px; box-shadow: inset 0 1px 1px rgba(255,255,255,0.7);"></div>
      </div>
  </div>
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown('<div style="font-size: 14px; font-weight: bold; letter-spacing: 1px; color: #306ca0; margin-bottom: 12px; margin-top: 5px; text-shadow: 1px 1px 0 #fff;">THREAT CLASSES</div>', unsafe_allow_html=True)
        for cat, desc in {
            "JAILBREAK":        "Persona / DAN filter evasion",
            "PROMPT INJECTION":  "System instruction hijack",
            "HARMFUL BEHAVIOR":  "Direct hazardous requests",
            "TOXICITY":          "Hate speech and harassment",
            "LINGUISTIC":        "Social engineering / evasion",
        }.items():
            st.markdown(f"""
<div style="background: linear-gradient(to right, #f0f7fc, #e2eef7); color: #3b7cae; padding: 12px 14px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #bde0fa; box-shadow: 0 1px 2px rgba(0,0,0,0.05), inset 0 1px 2px rgba(255,255,255,0.9); display: flex; align-items: center; justify-content: space-between;">
  <span style="font-size: 14px; font-weight: 600; text-shadow: 1px 1px 0 #fff;">{cat}</span>
  <span style="font-size: 13px; font-weight: 500; color: #5a8ab8;">{desc}</span>
</div>
""", unsafe_allow_html=True)
