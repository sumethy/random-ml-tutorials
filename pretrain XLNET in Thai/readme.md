# วิธีการ pretrain XLNET สำหรับภาษาไทย

## 1. Get and prepare training data

สร้าง working directory ใหม่แล้ว cd เข้าไป

clone repo ของ xlnet `git clone https://github.com/zihangdai/xlnet` แล้วก็ download training data (วิกิไทยที่คลีนเรียบร้อยแล้ว) `https://drive.google.com/file/d/1QZSOpikO6Qc02gRmyeb_UiRLtTmUwGz1/view?usp=sharing`

แตกไฟล์ preprocessed_thaiwikitext.zip ที่โหลดมา เปลี่ยนชื่อไฟล์ thaiwikitext_sentseg ข้างในให้เป็น .txt แล้วลบอีกไฟล์นึง (.xml) ทิ้งไป

download `https://drive.google.com/file/d/1F7pCgt3vPlarI9RxKtOZUrC_67KMNQ1W/view?usp=sharing` (sentencepiece ภาษาไทย) แล้วก็แตกไฟล์

สร้างโฟล์เดอร์ใหม่ชื่อ tf_record_out

ตอนนี้โครงสร้างโฟล์เดอร์เราจะเป็นแบบนี้

```
xlnet\ (อันที่โคลนมา ข้างในมีไฟล์เต็มไปหมด)
preprocess_thaiwikitext\
    |--->thaiwikitext_sentseg.txt
th_wiki_bpe\
    |--->th.wiki.bpe.op25000.model
    |--->th.wiki.bpe.op25000.vocab
tf_record_out
```
