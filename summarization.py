import torch
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langdetect import DetectorFactory
import gc
import docx
import time
import warnings
import os

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

warnings.filterwarnings("ignore")

# ============================================================
# 1. GPU / VRAM Optimization
# ============================================================
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

clear_memory()

# ============================================================
# 2. Model Load (Gemma-27B IT Optimized)
# ============================================================
model_id = "google/gemma-3-27b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("🔄 Model yuklanmoqda...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    padding_side="left"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    batch_size=1
)
print("✅ Model yuklandi!")

# ============================================================
# 3. File Reader (PDF + DOCX)
# ============================================================
def read_file(file_path, max_pages=50, max_chars=100000):
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            text_parts, total_chars = [], 0
            for i in range(min(len(doc), max_pages)):
                if total_chars >= max_chars:
                    break
                page_text = doc[i].get_text("text", flags=11)
                if page_text.strip():
                    text_parts.append(page_text)
                    total_chars += len(page_text)
            doc.close()
            text = " ".join(text_parts)
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            text_parts, total_chars = [], 0
            for para in doc.paragraphs:
                if total_chars >= max_chars:
                    break
                if para.text.strip():
                    text_parts.append(para.text.strip())
                    total_chars += len(para.text)
            text = " ".join(text_parts)
        else:
            return None, "❌ Faqat PDF yoki DOCX qo‘llab-quvvatlanadi"
        
        text = " ".join(text.split())
        return text[:max_chars].strip(), None
    except Exception as e:
        return None, f"❌ Fayl o‘qishda xatolik: {e}"

# ============================================================
# 4. Language Detection
# ============================================================
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    sample = text[:1000]
    if any(c in sample for c in ['ў', 'ғ', 'ҳ', 'қ']):
        return 'uz'
    if sum(1 for w in [' va ', ' bilan ', ' uchun '] if w in sample.lower()) >= 2:
        return 'uz'
    if any(c in sample for c in ['ы', 'э', 'ё', 'щ']):
        return 'ru'
    if sum(1 for w in [' и ', ' в ', ' на '] if w in sample.lower()) >= 2:
        return 'ru'
    if sum(1 for w in [' the ', ' is ', ' and '] if w in sample.lower()) >= 2:
        return 'en'
    return 'uz'

# ============================================================
# 5. Prompt Builder
# ============================================================
def build_prompt(text, lang):
    prompts = {
    "uz": f"""Quyidagi matnni yaxlit, tugallangan tarzda qisqa xulosa qilib yozing. 
Matnning uzunligiga qarab xulosa hajmini tanlang: 
- qisqa matnlar uchun 3–5 gap,
- o‘rta hajmdagi matnlar uchun 5–7 gap,
- katta hajmdagi matnlar uchun 7–10 gap yoki 2–3 paragraf yozing. 

Hech qachon jumlani yarimta qoldirmang. 
Sanalarni va faktlarni matnda qanday berilgan bo‘lsa, o‘sha holatda saqlang. 
Agar sanalarda yoki faktlarda qarama-qarshilik bo‘lsa, uni izohlamang va tuzatmang — faqat matndagi variantni xulosa qiling.
Matn:\n\n{text}\n\n📑 Xulosa:""",
    
    "ru": f"""Напишите краткое резюме следующего текста. 
Длина резюме должна зависеть от объёма текста: 
- для коротких текстов — 3–5 предложений,
- для средних — 5–7 предложений,
- для больших документов — 7–10 предложений или 2–3 абзаца. 

Закончите резюме полной фразой. 
Все даты и факты приводите строго в том виде, как они указаны в тексте, без изменений. 
Если в тексте есть противоречия в датах или фактах, не исправляйте и не поясняйте их — просто используйте то, что дано в тексте.
\n\n{text}\n\nРезюме:""",
    
    "en": f"""Write a summary of the following text. 
The length of the summary should depend on the size of the text: 
- for short texts — 3–5 sentences,
- for medium texts — 5–7 sentences,
- for large documents — 7–10 sentences or 2–3 paragraphs. 

Make sure the summary ends with a complete sentence. 
Preserve all dates and facts exactly as they appear in the text, without modification. 
If there are inconsistencies in dates or facts, do not explain or correct them — just summarize the text as it is.
\n\n{text}\n\nSummary:"""
}
    return prompts.get(lang, prompts["uz"])

# ============================================================
# 6. Text Generation
# ============================================================
def generate_summary(prompt, max_tokens=400):
    try:
        with torch.inference_mode():
            output = pipe(
                prompt,
                max_new_tokens=max_tokens,
                min_new_tokens=50,
                temperature=0.3,
                do_sample=False,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True,
                use_cache=True
            )
        text = output[0]["generated_text"].strip()
        return text
    except Exception as e:
        return f"❌ Generation xatosi: {e}"

# ============================================================
# 7. Post-processing (Ensure full sentence)
# ============================================================
def clean_summary(summary: str) -> str:
    # Remove prompt markers if present
    for marker in ["Xulosa:", "Summary:", "Резюме:"]:
        if marker in summary:
            summary = summary.split(marker)[-1].strip()

    # Ensure summary ends with full stop, question mark, or exclamation
    if not summary.endswith((".", "!", "?")):
        last_dot = max(summary.rfind("."), summary.rfind("!"), summary.rfind("?"))
        if last_dot != -1:
            summary = summary[: last_dot + 1].strip()
    return summary

def extract_key_sections(text, max_chars=5000):
    """Extract introduction, middle, and conclusion for better summaries"""
    text_len = len(text)
    if text_len <= max_chars:
        return text
    
    # Take intro (30%), middle sample (40%), conclusion (30%)
    intro_size = int(max_chars * 0.3)
    middle_size = int(max_chars * 0.4)
    conclusion_size = int(max_chars * 0.3)
    
    intro = text[:intro_size]
    middle_start = (text_len // 2) - (middle_size // 2)
    middle = text[middle_start:middle_start + middle_size]
    conclusion = text[-conclusion_size:]
    
    return f"{intro} ... {middle} ... {conclusion}"

# ============================================================
# 8. FastAPI App
# ============================================================
app = FastAPI(title="⚡ Summarizer API", version="0.1.0")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="https://webimzo.uz/openapi.json",
        title="My API Docs"
        # boshqa parametrlar ham joylashtirilishi mumkin
    )

@app.get("/ai/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="https://webimzo.uz/openapi.json",
        title="My API Docs"
        # boshqa parametrlar ham joylashtirilishi mumkin
    )

#@app.post("/summarize-file")
#async def summarize_file(file: UploadFile = File(...)):
#    start_time = time.time()

#    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
#        raise HTTPException(status_code=400, detail="Faqat PDF yoki DOCX qo‘llab-quvvatlanadi")
    
#    temp_path = f"/tmp/{file.filename}"
#    with open(temp_path, "wb") as f:
#        f.write(await file.read())

#    text, error = read_file(temp_path, max_pages=8, max_chars=8000)
#    os.remove(temp_path)

#    if error:
#        raise HTTPException(status_code=500, detail=error)
#    if not text or len(text) < 50:
#        raise HTTPException(status_code=400, detail="Fayl bo‘sh yoki juda qisqa")

#    lang = detect_language(text)
#    word_count = len(text.split())
#    prompt = build_prompt(text, lang)
#    summary_raw = generate_summary(prompt, max_tokens=150)
#    summary = clean_summary(summary_raw)

#    elapsed = time.time() - start_time
#    clear_memory()
    
#    return JSONResponse({
#        "success": True,
#        "summary": summary,
#        "language": lang,
#        "stats": {
#            "original_words": word_count,
#            "summary_words": len(summary.split()),
#            "time_seconds": round(elapsed, 2)
#        }
#    })


#@app.post("/ai/summarize-file")
#async def summarize_file(file: UploadFile = File(...)):
#    start_time = time.time()

#    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
#        raise HTTPException(status_code=400, detail="Faqat PDF yoki DOCX qo‘llab-quvvatlanadi")
    
#    temp_path = f"/tmp/{file.filename}"
#    with open(temp_path, "wb") as f:
#        f.write(await file.read())

#    text, error = read_file(temp_path, max_pages=50, max_chars=100000)
#    os.remove(temp_path)

#    if error:
#        raise HTTPException(status_code=500, detail=error)
#    if not text or len(text) < 50:
#        raise HTTPException(status_code=400, detail="Fayl bo‘sh yoki juda qisqa")
#    lang = detect_language(text)
#    word_count = len(text.split())
#    prompt = build_prompt(text, lang)
#    summary_raw = generate_summary(prompt, max_tokens=400)
#    summary = clean_summary(summary_raw)

#    elapsed = time.time() - start_time
#    clear_memory()
    
#    return JSONResponse({
#        "success": True,
#        "summary": summary,
#        "language": lang,
#        "stats": {
#            "original_words": word_count,
#            "summary_words": len(summary.split()),
#            "time_seconds": round(elapsed, 2)
#        }
#    })

@app.post("/ai/summarize-file")
async def summarize_file(file: UploadFile = File(...)):
    start_time = time.time()

    # Validate file type
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Faqat PDF yoki DOCX qo'llab-quvvatlanadi")
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Read file with reduced limits for initial speed boost
    # Reduce max_pages to 20 and max_chars to 50000 for faster processing
    text, error = read_file(temp_path, max_pages=20, max_chars=50000)
    os.remove(temp_path)

    if error:
        raise HTTPException(status_code=500, detail=error)
    if not text or len(text) < 50:
        raise HTTPException(status_code=400, detail="Fayl bo'sh yoki juda qisqa")
    
    # Store original metrics
    original_word_count = len(text.split())
    
    # Extract key sections to reduce processing time
    # This is crucial for performance - reduces text from 50000 to 5000 chars
    processed_text = extract_key_sections(text, max_chars=5000)
    
    # Detect language on the processed text for better accuracy
    lang = detect_language(processed_text)
    
    # Build prompt with the extracted key sections
    prompt = build_prompt(processed_text, lang)
    
    # Generate summary with optimized parameters
    summary_raw = generate_summary(prompt, max_tokens=300)  # Reduced from 400
    summary = clean_summary(summary_raw)

    # Calculate processing time
    elapsed = time.time() - start_time
    
    # Clear GPU memory
    clear_memory()
    
    return JSONResponse({
        "success": True,
        "summary": summary,
        "language": lang,
        "stats": {
#            "original_words": original_word_count,
#            "processed_chars": len(processed_text),
            "summary_words": len(summary.split()),
            "compression_ratio": round(original_word_count / len(summary.split()), 2),
            "time_seconds": round(elapsed, 2)
        }
    })