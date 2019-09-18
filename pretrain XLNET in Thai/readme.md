# วิธีการ pretrain XLNET สำหรับภาษาไทย

ใช้ Colab TPU เทรน เพราะว่าถ้าเป็น GPU แรมไม่พอ T_T ต้องปรับเป็นโมเดลเล็ก (เล็กกว่า xlnet-base) และความยาวของ sequence input เหลือแค่ 128
แต่ถ้าใช้ TPU (Colab เป็น TPU v2) สามารถเทรนโมเดลขนาด(เกือบ)เท่า xlnet-base ได้ ที่ความยาวอินพุต 512 เต็มๆ ถ้าจะเทรน xlnet-large คงต้องใช้ Cloud TPU v3 แบบจ่ายตังค์

## 1.1 Get and prepare training data

download training data (วิกิไทยที่คลีนเรียบร้อยแล้ว) `https://drive.google.com/file/d/1QZSOpikO6Qc02gRmyeb_UiRLtTmUwGz1/view?usp=sharing`

แตกไฟล์ preprocessed_thaiwikitext.zip ที่โหลดมา เปลี่ยนชื่อไฟล์ thaiwikitext_sentseg ข้างในให้เป็น .txt แล้วลบอีกไฟล์นึง (.xml) ทิ้งไป

## 1.2 Train sentencepiece model

Install sentencepiece: `pip install sentencepiece`

เซฟโค้ดด้านล่างนี้เป็น .py อย่าลืมเปลี่ยน --input ให้ตรงกับที่ๆเก็บไฟล์ `thaiwikitext_sentseg.txt` ไว้ รันบนโน๊คบุคก็ได้เพราะว่าไม่ต้องใช้ GPU พอรันจบแล้วจะได้ไฟล์ออกมาสองไฟล์ `thaiwiki.model` และ `thaiwiki.vocab` สองไฟล์นี้คือ sentencepiece model ที่จะใช้ในขั้นตอนต่อไป
```
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=preprocessed_thaiwikitext/thaiwikitext_sentseg.txt \
--vocab_size=32000 --character_coverage=0.99995 --model_type=unigram \
--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> --user_defined_symbols=<eop>,.,(,),",-,–,£,€ \
--shuffle_input_sentence --model_prefix=thaiwiki')
```
## 2.1 Create tfrecord

clone xlnet repo: git clone https://github.com/zihangdai/xlnet.git

โครงสร้าง directory

```
preprocessed_thaiwikitext/
   |----->thaiwikitext_sentseg.txt (มาจากข้อ 1.1)
xlnet/ (repo ที่ clone มา)
tf_record_out/ (directory ว่าง)
thaiwiki.model (มาจากข้อ 1.2)
thaiwiki.vocab (มาจากข้อ 1.2)
```

สิ่งสำคัญในการสร้าง tfrecord คือตอนเทรน ต้องใช้ option ให้ตรงกันกับตอนที่สร้าง tfrecord ไม่งั้นจะเจอ error เต็มไปหมดเหมือนที่เราเจอมาแล้ว T_T option ที่สำคัญมีดังต่อไปนี้: `bsz_per_host (int), seq_len (int), reuse_len (int), bi_data (boolean), uncased (boolean), mask_alpha (int), mask_beta (int), num_predict (int)` คำสั่งที่ใช้ในการสร้าง tfrecord ที่เราใช้คือ 

```
python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=8 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=../preprocessed_thaiwikitext/*.txt \
	--save_dir=../tf_record_out \
	--num_passes=20 \
	--bi_data=True \
	--sp_path=../thaiwiki.model \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85 \
	--uncased=False
```
เสร็จแล้วจะได้ไฟล์ออกมาในโฟลเดอร์ `tf_record_out` แบบนี้
```
tf_record_out\
    |--->corpus_info.json	
    |--->tfrecords/
            |---->record_info-train-0-0.bsz-32.seqlen-512.reuse-256.bi.alpha-6.beta-1.fnp-85.json
            |---->train-0-0.bsz-32.seqlen-512.reuse-256.bi.alpha-6.beta-1.fnp-85.tfrecords
```
สังเกตว่าชื่อไฟล์ มันจะบ่งบอกถึง option ที่เรากำหนดตอนสร้าง tfrecord `bsz-32.seqlen-512.reuse-256.bi.alpha-6.beta-1.fnp-85` หมายถึง `bsz_per_host=32`, `seq_len=512`, `reuse_len=256`, `bi_data=True`, `mask_alpha=6`, `mask_beta=1`, `num_predict=85` และ `uncased=False` (ถ้าเป็น True ชื่อไฟล์จะมี -uncased ด้วย) ดังนั้นตอนเทรน ถ้าเรากำหนด option ไม่ตรงกัน มันจะไปหาไฟล์ชื่ออื่น เช่น ถ้าตอนเทรนเราดันไปกำหนด batch size เป็น 64 มันก็จะพยายามไปหาไฟล์ที่ชื่อ ...bsz-64... แทน ทำให้เกิด error (เจอมากับตัวเองแล้ว ใครเคย debug tensorflow ก็จะรู้ว่า error message นี่นะ เอิ่ม )

## 2.2 เอา tfrecord ใส่ bucket

การที่จะใช้ TPU ได้นั้น ทั้งดาต้าและโฟล์เดอร์สำหรับ checkpoint จะต้องอยู่ใน GCP bucket (ใครไม่เคยใช้ ไปหัดใช้ก่อน) ก่อนอื่นให้สร้างโฟลเดอร์ใหม่ใน bucket ชื่อ xlnet/ เอาไว้เก็บ checkpoint

ที่นี้มาถึงขั้นตอนที่ hack สุดๆแล้วครับ กว่าจะหาวิธีได้ debug อยู่นานมาก เรื่องของเรื่อง ถ้าเราเปิดไฟล์ `record_info-train-0-0.bsz-32.seqlen-512.reuse-256.bi.alpha-6.beta-1.fnp-85.json` ม้นจะหน้าตาแบบนี้

```
{"filenames": ["../tf_record_out/tfrecords/train-0-0.bsz-32.seqlen-512.reuse-256.bi.alpha-6.beta-1.fnp-85.tfrecords"], "num_batch": 13329}
```
ฟิวด์ `filenames` เป็นตัวบอกว่าไฟล์ tfrecord ของเราอยู่ตรงไหน ซึ่งก็ตรงกับ option `--save_dir` จากข้อ 2.1 แต่ช้าก่อน เพราะว่ามันเป็น path local (local หมายถึงใน VM Colab นะ) ดังนั้นถ้าเราก๊อปปี้ใส่ bucket ไปทั้งๆแบบนี้ สิ่งที่จะเกิดขึ้นคือมันจะเกิด error ว่า "[local] filesystem not implemented" เพราะว่า TPU ไม่สามารถอ่านไฟล์ใน local ได้ อ่านได้จากใน bucket เท่านั้น
