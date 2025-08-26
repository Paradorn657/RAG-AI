from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import numpy as np
import os

app = FastAPI()

KB_FILES = [
        "pdf_embedding_igenco.json",
        "pdf_embedding_igenco_2.json",
        "pdf_embedding_igenco_3.json"
    ]

model = SentenceTransformer('intfloat/multilingual-e5-large') 

# โหลดฐาน embedding
# ตรวจสอบให้แน่ใจว่าไฟล์นี้ถูกสร้างขึ้นอย่างถูกต้องและมีคุณภาพ
with open('pdf_embedding.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
   api_key="api",
    default_headers={
        "HTTP-Referer": "https://your-site.com", # ควรเปลี่ยนเป็นโดเมนจริงของคุณ
        "X-Title": "MyPDFAI"
    }
)

class Query(BaseModel):
    question: str

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    # เพิ่มการจัดการกรณีที่เวกเตอร์เป็นศูนย์เพื่อป้องกัน ZeroDivisionError
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def find_top_k_context(question_emb, json_file_paths, k=3, min_score_threshold=0.6):
    """
    Finds the top-k relevant contexts from multiple JSON knowledge base files
    based on cosine similarity with the question embedding.
    """
    all_scores_across_files = [] # <<< นี่คือลิสต์ใหม่ที่จะเก็บคะแนนและเนื้อหาจากทุกไฟล์

    for json_file_path in json_file_paths: # <<< วนลูปผ่านพาธไฟล์ JSON แต่ละไฟล์
        if not os.path.exists(json_file_path):
            continue
        with open(json_file_path, 'r', encoding='utf-8') as f:
            kb_data_from_file = json.load(f) # <<< โหลดข้อมูลจากไฟล์ปัจจุบัน
            
            for entry in kb_data_from_file: # <<< วนลูปผ่านแต่ละ entry ในไฟล์ปัจจุบัน
                score = cosine_similarity(question_emb, entry['embedding'])
                
                if score >= min_score_threshold: 
                    all_scores_across_files.append({
                        'score': score, 
                        'content': entry['content'],
                        'file': json_file_path # เพิ่มเพื่อให้รู้ว่ามาจากไฟล์ไหน (ไม่บังคับ แต่มีประโยชน์)
                    })
            

    # <<< หลังจากวนลูปครบทุกไฟล์แล้ว ค่อยมาเรียงลำดับและเลือก top-k จากข้อมูลทั้งหมด
    all_scores_across_files.sort(key=lambda x: x['score'], reverse=True)
    top_contexts = [item['content'] for item in all_scores_across_files[:k]]

    return "\n---\n".join(top_contexts).strip()

@app.post("/ask")
async def ask(query: Query):
    question_embedding = model.encode(query.question).tolist()
    
    # ดึง Top-K context แทนที่จะเป็นแค่ 1
    context = find_top_k_context(question_embedding,KB_FILES,k=3, min_score_threshold=0.3) # ปรับ k และ threshold ตามความเหมาะสม

    if not context:
        return {"answer": "ขออภัยค่ะ ฉันไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลของฉันสำหรับคำถามนี้"}

    messages = [
        {"role": "system", "content": """คุณคือผู้ช่วย AI ที่เชี่ยวชาญด้านข้อมูลภายในองค์กรของบริษัท
        จงตอบคำถามของผู้ใช้โดย **อ้างอิงจากข้อมูลที่ให้มาเท่านั้น** อย่างกระชับ ชัดเจน และเป็นธรรมชาติ
        **หากข้อมูลที่ให้มาไม่เพียงพอที่จะตอบคำถาม หรือคำถามไม่ได้เกี่ยวข้องกับข้อมูลที่ให้มา**
        ให้ตอบว่า "ขออภัยค่ะ ฉันไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลของฉันสำหรับคำถามนี้" หรือ "ฉันไม่สามารถตอบคำถามนี้ได้จากข้อมูลที่มีอยู่"
        **ห้ามสร้างข้อมูลใดๆ ขึ้นมาเองโดยเด็ดขาด**
        """},
        {"role": "user", "content": f"ข้อมูล:\n{context}\n\nคำถาม: {query.question}"}
    ]
    
    print("----- Context ที่ส่งให้ LLM -----")
    print(context)
    print("----- คำถามที่ส่งให้ LLM -----")
    print(query.question)
    print("---------------------------------")

    r = f"""System: คุณคือผู้ช่วย AI ที่เชี่ยวชาญด้านข้อมูลภายในองค์กรของบริษัท
    จงตอบคำถามของผู้ใช้โดย **อ้างอิงจากข้อมูลที่ให้มาเท่านั้น** อย่างกระชับ ชัดเจน และเป็นธรรมชาติ
    **หากข้อมูลที่ให้มาไม่เพียงพอที่จะตอบคำถาม หรือคำถามไม่ได้เกี่ยวข้องกับข้อมูลที่ให้มา**
    ให้ตอบว่า "ขออภัยค่ะ ฉันไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลของฉันสำหรับคำถามนี้" หรือ "ฉันไม่สามารถตอบคำถามนี้ได้จากข้อมูลที่มีอยู่"
    **ห้ามสร้างข้อมูลใดๆ ขึ้นมาเองโดยเด็ดขาด**
    User: ข้อมูล:
    {context}
    คำถาม: {query.question}"""


    import requests
    import json

    try:
        
        # payload = {
        #     "model": "sailor2:20b",
        #     "prompt": r, 
        #     "stream": False
        # }
        # print(r)
        # response = requests.post('http://103.107.53.251/api/generate', json=payload) # requests.post(url, json=...) จะจัดการ headers ให้
        # response.raise_for_status() # ตรวจสอบสถานะ HTTP errors (เช่น 4xx, 5xx)

        # # พิมพ์เฉพาะข้อความตอบกลับ
        
        # answer = response.json().get('response', 'ไม่พบข้อความตอบกลับ')


        response = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=messages,
            temperature=0.2 # ปรับค่านี้ให้น้อยลงเพื่อความแม่นยำสูงขึ้น แต่ก็อาจทำให้คำตอบซ้ำซาก
        )
        answer = response.choices[0].message.content
    except Exception as e:
        try:
            # ลองใช้โมเดลสำรอง
            response = client.chat.completions.create(
                model='deepseek/deepseek-r1-0528:free',
                messages=messages,
                temperature=0.2
            )
            answer = response.choices[0].message.content
            print(e)
        except Exception as fallback_e:
            # ถ้าโมเดลสำรองก็ยังล้มเหลว
            answer = "ขออภัยค่ะ ระบบ AI ไม่สามารถประมวลผลคำถามของคุณได้ในขณะนี้ กรุณาลองใหม่ในภายหลัง"
            print(fallback_e)
    print(answer)
    return {"answer": answer}