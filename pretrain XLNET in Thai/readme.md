# วิธีการ pretrain XLNET สำหรับภาษาไทย

## 1. Get and prepare training data

สร้าง working directory ใหม่แล้ว cd เข้าไป

clone repo ของ xlnet `git clone https://github.com/zihangdai/xlnet` แล้วก็ download training data (วิกิไทยที่คลีนเรียบร้อยแล้ว) `https://drive.google.com/file/d/1QZSOpikO6Qc02gRmyeb_UiRLtTmUwGz1/view?usp=sharing`

แตกไฟล์ preprocessed_thaiwikitext.zip ที่โหลดมา เปลี่ยนชื่อไฟล์ thaiwikitext_sentseg ข้างในให้เป็น .txt แล้วลบอีกไฟล์นึง (.xml) ทิ้งไป

download `https://drive.google.com/file/d/1F7pCgt3vPlarI9RxKtOZUrC_67KMNQ1W/view?usp=sharing` (sentencepiece ภาษาไทย) แล้วก็แตกไฟล์

สร้างโฟล์เดอร์ใหม่ชื่อ tf_record_out

ตอนนี้โครงสร้างโฟล์เดอร์เราจะเป็นแบบนี้

```
xlnet/ (อันที่โคลนมา ข้างในมีไฟล์เต็มไปหมด)
preprocess_thaiwikitext/
    |--->thaiwikitext_sentseg.txt
th_wiki_bpe/
    |--->th.wiki.bpe.op25000.model
    |--->th.wiki.bpe.op25000.vocab
tf_record_out
```
เสร็จแล้ว cd เข้าไปใน `xlnet/` แล้วรัน (python env ลง numpy, tensorflow, sentencepiece ให้เรียบร้อยก่อน)

```
python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=16 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=../preprocessed_thaiwikitext/*.txt \
	--save_dir=../tf_record_out \
	--num_passes=20 \
	--bi_data=True \
	--sp_path=../th_wiki_bpe/th.wiki.bpe.op25000.model \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85
```

เครื่องต้องแรงพอสมควร แนะนำให้ทำบน server จำนวน core 16+ (Colab ไม่ไหวมันให้ cpu มานิดเดียว) แต่เนื่องจากไฟล์วิกิพีเดียไทยไม่ค่อยใหญ่เท่าไหร่ server 16 core ใช้เวลาไม่ถึง 1 ชม เสร็จแล้วไฟล์ tf record จะอยู่ใน `tf_record_out`
