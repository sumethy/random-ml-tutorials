# วิธีการ finetune BERT สำหรับภาษาไทย

วิธีการ finetune model BERT สำหรับภาษาไทย โดยใช้ repo อันนี้ https://github.com/ThAIKeras/bert เสร็จแล้วนำไป deploy ด้วย bert-as-service https://github.com/hanxiao/bert-as-service.git

## 1. Setup

### 1.1 สร้าง directory ขึ้นมาใหม่ และ clone repo สองอันนี้ลงมา

https://github.com/ThAIKeras/bert

https://github.com/wongnai/wongnai-corpus.git

### 1.2 Download BERT thai model and Thai BPE files
(link อยู่ใน https://github.com/ThAIKeras/bert อยู่แล้ว แค่แยกออกมาให้หาง่ายขึ้น)

https://drive.google.com/open?id=1J3uuXZr_Se_XIFHj7zlTJ-C9wzI9W_ot

https://drive.google.com/file/d/1F7pCgt3vPlarI9RxKtOZUrC_67KMNQ1W/

### 1.3 Upload to GCP
ใน tutorial อันนี้จะรันบน Colab ดังนั้นให้ upload file ที่โหลดมาจากข้อ (bert_base_th.zip, th_wiki_bpe.zip) 1.2 ใส่ bucket บน GCP เพื่อให้โหลดใส่ VM Colab ได้อย่างรวดเร็ว

### 1.4 Finetune
