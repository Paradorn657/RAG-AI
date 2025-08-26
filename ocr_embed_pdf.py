# process_ocr_pdfs_and_append.py
# สคริปต์นี้ใช้สำหรับประมวลผลไฟล์ PDF ที่เป็นรูปภาพ/สแกน (ต้อง OCR)
# แปลงหน้าเป็นรูปภาพ, ใช้ Tesseract OCR, แปลงข้อความเป็น chunk,
# สร้าง embedding และเพิ่มข้อมูลลงใน pdf_embedding.json (โดยไม่ลบข้อมูลเดิม)

import os
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageFilter,ImageEnhance # Pillow for image manipulation (ยังจำเป็นสำหรับการบันทึกรูปภาพชั่วคราว)
import logging
import re
import subprocess # สำหรับเรียก Tesseract CLI
import numpy as np

OUTPUT_JSON = "pdf_embedding_tint.json"
PDF_FOLDER_FOR_OCR = "data_needtoOcr" # โฟลเดอร์สำหรับ PDF ที่ต้องการทำ OCR
TEMP_OCR_DIR = "tmp_ocr" # โฟลเดอร์สำหรับไฟล์ชั่วคราวที่ Tesseract สร้าง

# สร้างโฟลเดอร์ชั่วคราวถ้ายังไม่มี
if not os.path.exists(TEMP_OCR_DIR):
    os.makedirs(TEMP_OCR_DIR)

try:
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
    print("Successfully loaded SentenceTransformer model: intfloat/multilingual-e5-large")
except Exception as e:
    logging.error(f"Error loading SentenceTransformer model: {e}")
    exit("Failed to load embedding model. Exiting.")

def process_page_with_tesseract_cli(page, page_num,pdf_filename_base,dpi=600, lang='tha', psm=6, oem=1):
    TESSERACT_CMD = r'C:\Users\danai\Downloads\Tesseract-OCR\tesseract.exe' 

    text_from_ocr = ""

    pdf_output_folder = os.path.join(TEMP_OCR_DIR, pdf_filename_base)
    if not os.path.exists(pdf_output_folder):
        os.makedirs(pdf_output_folder)

    img_filename = f"page_{page_num}.png" 
    img_path = os.path.join(pdf_output_folder, img_filename)
    

    output_filename_base = f"ocr_output_{page_num}"
    output_path_base = os.path.join(pdf_output_folder, output_filename_base)
    output_txt_path = output_path_base + '.txt'

    try:
        # บันทึกหน้า PDF เป็นรูปภาพ

        if not os.path.exists(img_path):
            pix = page.get_pixmap(dpi=dpi)

                    # แปลง pixmap เป็นภาพ PIL เพื่อประมวลผล
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # แปลงเป็นขาวดำ (Grayscale)
            img = img.convert('L')
            
            # เพิ่มคอนทราสต์
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # ปรับคอนทราสต์ (อาจต้องทดลองปรับค่า)
            
            # ลด noise ด้วย Gaussian Blur
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            
            # แปลงเป็นภาพไบนารี (ขาว-ดำ) เพื่อให้ข้อความชัดเจน
            threshold = 128  # ค่าธรณี (อาจต้องปรับตามภาพ)
            img = img.point(lambda p: 255 if p > threshold else 0)

            img.save(img_path)
            #pix.save(img_path)
        
        # สร้างคำสั่ง Tesseract
        if not os.path.exists(output_txt_path):
            cmd = [
                TESSERACT_CMD,
                img_path,          # ไฟล์รูปภาพ Input
                output_path_base,  # ชื่อไฟล์พื้นฐานสำหรับ Output Text (Tesseract จะเพิ่ม .txt)
                '-l', lang,        # ภาษาที่ใช้ OCR (tha+eng)
                '--psm', str(psm), # Page segmentation mode
                '--oem', str(oem)  # OCR Engine mode
            ]
            
            print(f"Running Tesseract command: {' '.join(cmd)}")
        
            # รันคำสั่ง Tesseract
            process = subprocess.run(cmd, capture_output=True, text=True, check=False) 
        
            if process.returncode != 0:
                logging.error(f"Tesseract command failed with error code {process.returncode}: {process.stderr}")
                # ถ้า Tesseract ล้มเหลว อาจไม่มีไฟล์ .txt ถูกสร้าง
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                return ""

            # อ่านผลลัพธ์จากไฟล์ .txt
            if os.path.exists(output_txt_path):
                with open(output_txt_path, 'r', encoding='utf-8') as f:
                    text_from_ocr = f.read()
                print(f"OCR successful for {img_path}. Extracted {len(text_from_ocr)} characters.")
            else:
                logging.error(f"Tesseract did not create expected output file: {output_txt_path}")
                return ""
        # อ่านผลลัพธ์จากไฟล์ .txt
        if os.path.exists(output_txt_path):
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                text_from_ocr = f.read()
            print(f"reading  {img_path}. Extracted {len(text_from_ocr)} characters.")
        else:
            logging.error(f"not found txt: {output_txt_path}")
            return ""
            
    except Exception as ocr_e:
        logging.error(f"Error during OCR process for {img_path}: {ocr_e}")
        return ""  
    return text_from_ocr



# --- ฟังก์ชันทำความสะอาดข้อความ ---
def clean_text(text):
    """ทำความสะอาดข้อความ เช่น ลบช่องว่างเกินหนึ่งช่อง"""
    text = re.sub(r'\s+', ' ', text).strip()
    # คุณอาจเพิ่มการลบอักขระพิเศษอื่นๆ ที่มักจะมาจาก OCR ได้ที่นี่
    # เช่น: text = re.sub(r'[^a-zA-Z0-9\u0E00-\u0E7F\s.,!?]', '', text)
    return text

# --- STEP 3: แบ่งข้อความเป็น chunk ---
def split_text(text, max_length=300):
    """
    แบ่งข้อความยาวๆ เป็น chunk เล็กๆ โดยพยายามคงบริบทของประโยค/ย่อหน้า
    และมีขนาดไม่เกิน max_length
    """
    # แยกด้วย .!? ตามด้วยช่องว่าง หรือ double/single newline
    sentences = re.split(r'(?<=[.!?])\s+|\n\n|\n', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: # ข้ามประโยคว่างเปล่า
            continue
        
        # เพิ่มประโยคเข้า chunk ปัจจุบัน ถ้ายังไม่เกิน max_length
        if len(current_chunk) + len(sentence) + 1 <= max_length: # +1 for a space
            current_chunk += sentence + " "
        else:
            # ถ้าเกิน ให้เพิ่ม chunk ปัจจุบัน (ถ้ามี) แล้วเริ่ม chunk ใหม่
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    # เพิ่ม chunk สุดท้าย (ถ้ามี)
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [chunk for chunk in chunks if chunk] # กรอง chunks ที่ว่างเปล่า

# --- STEP 4: สร้าง Embedding ---
def create_embedding(text):
    """สร้างเวกเตอร์ Embedding จากข้อความที่กำหนด"""
    embedding = embedding_model.encode(text, convert_to_tensor=False).tolist()
    return embedding

# --- Main Logic: วนลูปผ่านไฟล์ PDF ที่เป็นรูปภาพและประมวลผล OCR และเพิ่มข้อมูล ---
def process_ocr_pdfs_and_append_embeddings(pdf_folder_path=PDF_FOLDER_FOR_OCR, output_json_path=OUTPUT_JSON):
    """
    ประมวลผลไฟล์ PDF ทั้งหมดในโฟลเดอร์ 'ocr_pdfs' ด้วย OCR, 
    สร้าง Embedding และเพิ่มลงในไฟล์ JSON ที่มีอยู่ (หรือสร้างใหม่)
    """
    all_embeddings_data = []

    # โหลดข้อมูลเก่าถ้ามี
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                all_embeddings_data = json.load(f)
            print(f"Loaded existing embeddings from {output_json_path}")
        except json.JSONDecodeError:
            logging.warning(f"decoding {output_json_path}. Starting with empty embeddings list (this might be an issue if you expect old data).")
            all_embeddings_data = []
    
    # เก็บชื่อไฟล์และประเภทที่ประมวลผลไปแล้วเพื่อหลีกเลี่ยงการประมวลผลซ้ำ
    # (filename, type) tuple เช่น ("doc1.pdf", "ocr_layer")
    processed_entries = {(entry.get("file"), entry.get("type")) for entry in all_embeddings_data}
    
    # ตรวจสอบว่าโฟลเดอร์มีอยู่
    if not os.path.exists(pdf_folder_path):
        logging.error(f"Folder '{pdf_folder_path}' not found. Please create it and put your OCR-based PDFs there.")
        return

    # วนลูปผ่านไฟล์ PDF ทั้งหมดในโฟลเดอร์สำหรับ OCR
    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_folder_path, filename)
            
            # ตรวจสอบว่าไฟล์นี้ถูกประมวลผลด้วย OCR แล้วหรือยัง
            if (filename, "ocr_layer") in processed_entries:
                print(f"Skipping {filename}: Already processed with OCR.")
                continue

            print(f"Processing OCR PDF: {file_path}")
            
            doc = fitz.open(file_path)
            full_ocr_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"  Performing OCR on page {page_num + 1}/{len(doc)} of {filename}")
                # เรียกใช้ฟังก์ชัน OCR ที่สร้างไว้
                ocr_text = process_page_with_tesseract_cli(page, page_num,os.path.splitext(filename)[0])
                full_ocr_text += ocr_text + "\n"
            doc.close() # ปิดเอกสาร PDF เมื่อใช้เสร็จ
            
            cleaned_text = clean_text(full_ocr_text)

            if not cleaned_text.strip():
                logging.warning(f"No meaningful text extracted from {filename} via OCR. Skipping.")
                continue

            chunks = split_text(cleaned_text, max_length=300)

            if not chunks:
                logging.warning(f"No chunks created for {filename} from OCR. Skipping.")
                continue

            # สร้าง Embedding สำหรับแต่ละ chunk และเพิ่มลงในลิสต์
            for i, chunk in enumerate(chunks):
                print(f"Creating embedding for chunk {i+1}/{len(chunks)} (OCR) of {filename}")
                try:
                    embedding = create_embedding(chunk)
                    all_embeddings_data.append({
                        "id": i,
                        "content": chunk,
                        "embedding": embedding
                    })
                except Exception as emb_e:
                    logging.error(f"Failed to create embedding for OCR chunk {i} in {filename}: {emb_e}")
            
            print(f"Processed {len(chunks)} OCR chunks from {filename}.")

    # บันทึกข้อมูล Embedding ทั้งหมดลงในไฟล์ JSON (ซึ่งจะรวมข้อมูลเก่าและใหม่)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_embeddings_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully processed all OCR PDFs and appended embeddings to {output_json_path}")

# --- รันฟังก์ชันหลักเมื่อสคริปต์ถูกเรียกใช้โดยตรง ---
if __name__ == "__main__":
    process_ocr_pdfs_and_append_embeddings()