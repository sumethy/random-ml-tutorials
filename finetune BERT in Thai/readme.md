# วิธีการ finetune BERT สำหรับภาษาไทย

วิธีการ finetune model BERT สำหรับภาษาไทย โดยใช้ repo อันนี้ https://github.com/ThAIKeras/bert เสร็จแล้วนำไป deploy ด้วย bert-as-service https://github.com/hanxiao/bert-as-service.git โดยการ finetune ใช้ Colab แล้วโหลดโมเดลกลับมา deploy บน local

## 1. Setup

### 1.1 สร้าง directory ขึ้นมาใหม่ และ clone repo สองอันนี้ลงมา

https://github.com/ThAIKeras/bert

https://github.com/wongnai/wongnai-corpus.git

### 1.2 Download BERT thai model and Thai BPE files
(link อยู่ใน https://github.com/ThAIKeras/bert อยู่แล้ว แค่แยกออกมาให้หาง่ายขึ้น)

https://drive.google.com/open?id=1J3uuXZr_Se_XIFHj7zlTJ-C9wzI9W_ot

https://drive.google.com/file/d/1F7pCgt3vPlarI9RxKtOZUrC_67KMNQ1W/

### 1.3 Upload to GCP
ใน tutorial อันนี้จะรันบน Colab ดังนั้นให้ upload file ที่โหลดมาจากข้อ (bert_base_th.zip, th_wiki_bpe.zip) 1.2 ใส่ bucket บน GCP เพื่อให้โหลดใส่ VM Colab ได้อย่างรวดเร็ว เสร็จแล้วให้แตกไฟล์ bert_base_th.zip, th_wiki_bpe.zip ไว้ใน local ด้วย

## 2 Finetune
เปิดไฟล์ `bert_wongnai_gpu.ipynb` บน Colab รันจนจบ จะได้ไฟล์ output_last.zip อยู่บน bucket GCP ให้ download ไฟล์นั้นกลับลงมาที่ local

## 3. Setup สำหรับการ deploy ด้วย bert-as-service

### 3.1 ติดตั้ง bert-as-service
ทำตามวิธีติดตั้งการใน https://github.com/hanxiao/bert-as-service

### 3.2 สร้างไฟล์ vocab.txt สำหรับการ deploy
เปิดไฟล์ /th_wiki_bpe/th.wiki.bpe.op25000.vocab จะเห็นไฟล์หน้าตาแบบนี้

