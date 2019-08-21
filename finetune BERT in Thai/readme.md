# วิธีการ finetune BERT สำหรับภาษาไทย

วิธีการ finetune model BERT สำหรับภาษาไทย โดยใช้ repo อันนี้ https://github.com/ThAIKeras/bert เสร็จแล้วนำไป deploy ด้วย bert-as-service https://github.com/hanxiao/bert-as-service.git โดยการ finetune ใช้ Colab แล้วโหลดโมเดลกลับมา deploy บน local

## 1. Setup

### 1.1 สร้าง directory ขึ้นมาใหม่ และ clone repo สองอันนี้ลงมา

https://github.com/ThAIKeras/bert

https://github.com/wongnai/wongnai-corpus.git

ต่อไปนี้จะเรียก directory ที่สร้างขึ้นมาใหม่นี้ว่า root

### 1.2 Download BERT thai model and Thai BPE files
(link อยู่ใน https://github.com/ThAIKeras/bert อยู่แล้ว แค่แยกออกมาให้หาง่ายขึ้น)

https://drive.google.com/open?id=1J3uuXZr_Se_XIFHj7zlTJ-C9wzI9W_ot

https://drive.google.com/file/d/1F7pCgt3vPlarI9RxKtOZUrC_67KMNQ1W/

BPE (byte pair encoding) คือ tokenizer ที่โมเดลพวก transformer (BERT, XLNET, GPT1/2) ใช้กัน หลักการคือมองสตริงเป็น unicode หลายๆตัวมาต่อกัน ข้อดีคือใช้ได้กับทุกภาษาโดยไม่ต้องมี algorithm แยกสำหรับแต่ละภาษา ส่วน sentencepiece คือการ implement BPE ให้เร็ว สามาอ่านเพิ่มเติมได้จาก https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46

### 1.3 Upload to GCP
ใน tutorial อันนี้จะรันบน Colab ดังนั้นให้ upload file ที่โหลดมาจากข้อ (bert_base_th.zip, th_wiki_bpe.zip) 1.2 ใส่ bucket บน GCP เพื่อให้โหลดใส่ VM Colab ได้อย่างรวดเร็ว เสร็จแล้วให้แตกไฟล์ bert_base_th.zip, th_wiki_bpe.zip ไว้ใน local ด้วย

## 2 Finetune

### 2.1 Run finetune
เปิดไฟล์ `bert_wongnai_gpu.ipynb` บน Colab รันจนจบ (ใช้เวลาประมาณ 1 ชม.) จะได้ไฟล์ output_last.zip อยู่บน bucket GCP ให้ download ไฟล์นั้นกลับลงมาที่ local จากนั้นแตกไฟล์ออกมาไว้ที่ใดก็ได้ ต่อไปนี้จะเรียกไฟล์เดอร์ที่เพิ่งแตกออกมานี้ว่า `finetuned_dir`

### 2.2 แก้ชื่อไฟล์ของโมเดล
เปิดโฟลเดอร์ `finetuned_dir` แล้วให้แก้ชื่อไฟล์ โดยเติม *bert_* ข้างหน้า *model* ทุกไฟล์ เช่น `model.ckpt.data-00000-of-00001` ให้แก้ชื่อไฟล์เป็น `bert_model.ckpt.data-00000-of-00001` เป็นต้น

## 3. Setup สำหรับการ deploy ด้วย bert-as-service

### 3.1 ติดตั้ง bert-as-service
ทำตามวิธีติดตั้งการใน https://github.com/hanxiao/bert-as-service

### 3.2 สร้างไฟล์ vocab.txt สำหรับการ deploy
เปิดไฟล์ /th_wiki_bpe/th.wiki.bpe.op25000.vocab จะเห็นไฟล์หน้าตาแบบนี้

![vocab dot text original](https://github.com/sumethy/random-ml-tutorials/blob/master/finetune%20BERT%20in%20Thai/images/vocab_txt_original.png)

เราต้องการลบตัวเลขสีม่วงๆออก ให้เหลือแต่ token บรรทัดละ 1 ตัวเท่านั้น วิธีการทำ ให้เอาไฟล์ `make-vocab-dot-txt.py` ไปรันในโฟลเดอร์ root (โครงสร้าง directory เป็นแบบนี้)

```
root
  |----th_wiki_bpe/ 
  |----bert_base_th/
  |----make-vocab-dot-txt.py
```

เมื่อรัน `make-vocab-dot-txt.py` แล้วจะได้ไฟล์ใหม่ `vocab.txt` ออกมาที่ root หน้าตาเป็นแบบนี้

![vocab dot text new](https://github.com/sumethy/random-ml-tutorials/blob/master/finetune%20BERT%20in%20Thai/images/vocab_txt_new.png)

**ตอนแรกบรรดทัดที่ 5 จะเป็น [unk] ตัวเล็ก ให้แก้เป็น [UNK] ตัวใหญ่ ตามภาพ**

### 3.3 ก๊อป vocab.txt ไปไว้ที่ `finetuned_dir`
### 3.4 ก๊อป /bert_base_th/bert_config.json ไปไว้ที่ `finetuned_dir`
โฟล์เดอร์ของโมเดลที่พร้อมจะ deploy จะมีไฟล์ดังต่อไปนี้

![ready-for-deploy](https://github.com/sumethy/random-ml-tutorials/blob/master/finetune%20BERT%20in%20Thai/images/ready-for-deploy.png)


## 4. Deploy ด้วย bert-as-service
เปิด terminal/cmd รัน `bert-serving-start -model_dir <finetuned_dir>` 

ตอนนี้สามารถใช้โมเดลใน `finetuned_dir` เป็น encoder ได้แล้ว (วิธีการ encode ดูจาก https://github.com/hanxiao/bert-as-service) การจะนำไปเทรน classifier ให้ encode ดาต้าทั้งหมดด้วย bert-as-service แล้วค่อยเอาไปเทรน dense+softmax เป็นโมเดลแยกเอา พอจะ inference ก็เอา string ไปผ่าน bert-as-service ก่อน พอได้ hidden state vector ออกมาแล้วก็ค่อยเอาไปผ่านโมเดล classifier อีกรอบ

ถ้าจะใช้โมเดลจากใน `finetuned_dir` ทำ inference โดยตรงเลยนั้นใช้ bert-as-service ไม่ได้ เพราะ output ของ bert-as-service เป็น hidden layer ไม่ใช่ softmax ตรงนี้เดี๋ยวมาอัปเดททีหลัง

## 5. ปรับใช้กับ classification problem อื่นๆ
สำหรับการนำไปใช้กับปัญหาอื่นๆที่ไม่ใช่ Wongnai นั้นต้อง implement คลาส `DataProcessor` ใหม่เอา ให้ศึกษาตัวอย่างได้จากคลาส `WongnaiProcessor` ใน `https://github.com/ThAIKeras/bert/run_classifier.py` ส่วนปัญหาประเภทอื่นๆเลยเช่น question answer เดี๋ยวมาอัปเดท
