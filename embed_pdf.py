# STEP 1: ติดตั้ง dependencies (ให้รันใน Python)
# pip install pymupdf openai

import fitz  # PyMuPDF
import json
from sentence_transformers import SentenceTransformer

TITLE = "MyPDFAI"
PDF_PATH = "iHuris-KPI+Training_GoldCity.pdf"
OUTPUT_JSON = "pdf_embedding_igenco_3.json"

model = SentenceTransformer('intfloat/multilingual-e5-large')


# --- STEP 2: อ่าน PDF แล้วแปลงเป็นข้อความ ---
def pdf_to_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- STEP 3: แบ่งข้อความเป็น chunk ---
def split_text(text, max_length=300):
    sentences = text.split(".")
    chunks = []
    current = ""
    for s in sentences:
        if len(current + s) < max_length:
            current += s + "."
        else:
            chunks.append(current.strip())
            current = s + "."
    if current:
        chunks.append(current.strip())
    return chunks

# --- STEP 4: สร้าง Embedding ด้วย OpenRouter ---
def create_embedding(text):
    embedding = model.encode(text, convert_to_tensor=False).tolist()
    return embedding

# --- STEP 5: ประมวลผล PDF แล้วบันทึก ---
def process_pdf_to_embedding():
    text = pdf_to_text(PDF_PATH)
    chunks = split_text(text)
    embedded_data = []

    for i, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)
        embedded_data.append({
            "id": i,
            "content": chunk,
            "embedding": embedding
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(embedded_data, f, ensure_ascii=False, indent=2)

    print(f"Embedding saved to {OUTPUT_JSON} ({len(embedded_data)} chunks)")

# รันกระบวนการทั้งหมด
if __name__ == "__main__":
    process_pdf_to_embedding()