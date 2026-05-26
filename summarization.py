import fitz  # PyMuPDF
import docx
import time
import os
import httpx
from typing import Optional
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from langdetect import DetectorFactory
from diagnoze_analizer import DialogueAnalyzer
DetectorFactory.seed = 0

# ============================================================
# 1. Konfiguratsiya
# ============================================================
VLLM_API_URL = "http://localhost:8008/v1/chat/completions"
#VLLM_API_URL = "http://vllm-server:8000/v1/chat/completions"
MODEL_ID = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
ihma_analyzer = DialogueAnalyzer(model=MODEL_ID,api_url=VLLM_API_URL)

app = FastAPI(title="⚡ vLLM Summarizer API", version="1.0.0")

# ============================================================
# 2. Fayl O'qish (Avvalgi kodingizdan)
# ============================================================
def read_file(file_path, max_pages=20, max_chars=50000):
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            text_parts, total_chars = [], 0
            for i in range(min(len(doc), max_pages)):
                if total_chars >= max_chars: break
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
                if total_chars >= max_chars: break
                if para.text.strip():
                    text_parts.append(para.text.strip())
                    total_chars += len(para.text)
            text = " ".join(text_parts)
        else:
            return None, "❌ Faqat PDF yoki DOCX qo'llab-quvvatlanadi"
        
        return " ".join(text.split())[:max_chars].strip(), None
    except Exception as e:
        return None, f"❌ Fayl o'qishda xatolik: {e}"

def detect_language(text: str) -> str:
    sample = text[:1000]
    if any(c in sample for c in ['ў', 'ғ', 'ҳ', 'қ']): return 'uz'
    if sum(1 for w in [' va ', ' bilan ', ' uchun '] if w in sample.lower()) >= 2: return 'uz'
    if any(c in sample for c in ['ы', 'э', 'ё', 'щ']): return 'ru'
    if sum(1 for w in [' и ', ' в ', ' на '] if w in sample.lower()) >= 2: return 'ru'
    if sum(1 for w in [' the ', ' is ', ' and '] if w in sample.lower()) >= 2: return 'en'
    return 'uz'

# ============================================================
# Til kodi mapping (ixtiyoriy `language` parami uchun)
# ============================================================
# Hech narsa yuborilmasa / noma'lum kod -> default o'zbek (lotin).
#   1 -> ru (rus)
#   2 -> uz_cyrl (o'zbek kirill)
#   3 -> uz_latn (o'zbek lotin)
#   4 -> en (ingliz)
DEFAULT_LANG = "uz_latn"
LANG_MAP = {
    1: "ru",
    2: "uz_cyrl",
    3: "uz_latn",
    4: "en",
}

def resolve_language(code: Optional[int]) -> str:
    if code is None:
        return DEFAULT_LANG
    return LANG_MAP.get(code, DEFAULT_LANG)

def extract_key_sections(text, max_chars=5000):
    text_len = len(text)
    if text_len <= max_chars: return text
    intro_size = int(max_chars * 0.3)
    middle_size = int(max_chars * 0.4)
    conclusion_size = int(max_chars * 0.3)
    intro = text[:intro_size]
    middle_start = (text_len // 2) - (middle_size // 2)
    middle = text[middle_start:middle_start + middle_size]
    conclusion = text[-conclusion_size:]
    return f"{intro} ... {middle} ... {conclusion}"

# ============================================================
# 3. vLLM ga so'rov yuborish (YANGI QISM)
# ============================================================
async def generate_summary_vllm(text, lang):
    system_prompts = {
        "uz_latn": "Sen professional matn xulosa qiluvchi yordamchisan. Asosiy faktlarni yo'qotmagan holda, aniq va lo'nda xulosa yozishing kerak. Javobni faqat o'zbek tilida, lotin alifbosida yoz.",
        "uz_cyrl": "Сен профессионал матн хулоса қилувчи ёрдамчисан. Асосий фактларни йўқотмаган ҳолда, аниқ ва лўнда хулоса ёзишинг керак. Жавобни фақат ўзбек тилида, кирилл алифбосида ёз.",
        "ru": "Вы профессиональный ИИ-ассистент для создания резюме текстов. Ваша задача — писать точные и краткие выводы, сохраняя факты.",
        "en": "You are a professional AI text summarizer. You must write clear and concise summaries while preserving key facts."
    }

    user_prompts = {
        "uz_latn": f"Quyidagi matnni hajmiga qarab 3-10 ta gap bilan xulosa qilib ber. Xulosani lotin alifbosida yoz. Matn:\n\n{text}",
        "uz_cyrl": f"Қуйидаги матнни ҳажмига қараб 3-10 та гап билан хулоса қилиб бер. Хулосани кирилл алифбосида ёз. Матн:\n\n{text}",
        "ru": f"Сделайте краткое резюме следующего текста (от 3 до 10 предложений в зависимости от длины). Текст:\n\n{text}",
        "en": f"Summarize the following text in 3 to 10 sentences depending on its length. Text:\n\n{text}"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompts.get(lang, system_prompts[DEFAULT_LANG])},
            {"role": "user", "content": user_prompts.get(lang, user_prompts[DEFAULT_LANG])}
        ],
        "temperature": 0.3,
        "max_tokens": 400,
        "top_p": 0.9
    }

    # vLLM ga so'rov yuborish
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(VLLM_API_URL, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content']

# ============================================================
# 4. API Endpointlar
# ============================================================
@app.get("/ai/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Summarizer API Docs")

@app.post("/ai/summarize-file")
async def summarize_file(
    file: UploadFile = File(...),
    language: Optional[int] = Form(None),
):
    start_time = time.time()

    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Faqat PDF yoki DOCX qo'llab-quvvatlanadi")
    
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    text, error = read_file(temp_path, max_pages=20, max_chars=50000)
    os.remove(temp_path)

    if error: raise HTTPException(status_code=500, detail=error)
    if not text or len(text) < 50: raise HTTPException(status_code=400, detail="Fayl bo'sh yoki juda qisqa")
    
    original_word_count = len(text.split())
    processed_text = extract_key_sections(text, max_chars=5000)
    # Xulosa tili ixtiyoriy `language` kodidan aniqlanadi.
    # Hech narsa yuborilmasa / noma'lum kod -> default o'zbek (lotin).
    lang = resolve_language(language)

    # ⚡ VLLM orqali ultra-tez generatsiya
    summary = await generate_summary_vllm(processed_text, lang)

    elapsed = time.time() - start_time
    
    return JSONResponse({
        "success": True,
        "summary": summary,
        "language": lang,
        "stats": {
            "original_words": original_word_count,
            "summary_words": len(summary.split()),
            "time_seconds": round(elapsed, 2)
        }
    })


@app.post("/ai/ihma-summary")
async def ihma_summary_endpoint(file: UploadFile = File(...)):
    """
    Siz taqdim etgan DialogueAnalyzer klassi orqali 
    dialoglarni chuqur tahlil qilish (vazifalar, qarorlar, statistika).
    """
    start_time = time.time()
    
    # Faylni vaqtincha saqlash
    temp_path = f"/tmp/ihma_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # 1. Ma'lumotni yuklash
        # DialogueAnalyzer.load_dialogue_data metodidan foydalanamiz
        dialogue_data = ihma_analyzer.load_dialogue_data(temp_path)
        print(dialogue_data)        
        # Agar bu oddiy matn bo'lsa (JSON bo'lmasa), klass ichidagi mantiqni hisobga olib
        # uni dialog formatiga o'tkazishga harakat qilamiz
        if not dialogue_data or 'dialogue' not in dialogue_data:
            content, error = read_file(temp_path)
            if error: raise HTTPException(status_code=400, detail=error)
            # Oddiy matnni bitta spikerli dialog sifatida o'raymiz
            dialogue_data = {
                "success": True,
                "dialogue": [{"speaker": "User", "text": content, "start": 0, "end": 10}],
                "filename": file.filename
            }

        # 2. DialogueAnalyzer orqali tahlil qilish
        # Bu metod ichida: xulosa, sentiment, mavzular, qarorlar va vazifalar bor
        analysis_results = ihma_analyzer.analyze_dialogue(dialogue_data)
        
        if 'error' in analysis_results:
            raise HTTPException(status_code=500, detail=analysis_results['error'])

        elapsed = time.time() - start_time
        
        # 3. Natijani qaytarish
        return JSONResponse({
            "success": True,
            "processing_time": round(elapsed, 2),
            "data": analysis_results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahlil jarayonida xatolik: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)