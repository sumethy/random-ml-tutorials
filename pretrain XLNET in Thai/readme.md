# วิธีการ pretrain XLNET สำหรับภาษาไทย

ใช้ Colab TPU เทรน เพราะว่าถ้าเป็น GPU แรมไม่พอ T_T ต้องปรับเป็นโมเดลเล็ก (เล็กกว่า xlnet-base) และความยาวของ sequence input เหลือแค่ 128
แต่ถ้าใช้ TPU (Colab เป็น TPU v2) สามารถเทรนโมเดลขนาด(เกือบ)เท่า xlnet-base ได้ ที่ความยาวอินพุต 512 เต็มๆ ถ้าจะเทรน xlnet-large คงต้องใช้ Cloud TPU v3 แบบจ่ายตังค์ (ขอติดไว้ภาค 2 ถ้าทำสำเร็จนะ)

## 1.1 Get and prepare training data

โหลดดาต้า (วิกิไทยที่คลีนเรียบร้อยแล้ว) `https://drive.google.com/file/d/1QZSOpikO6Qc02gRmyeb_UiRLtTmUwGz1/view?usp=sharing`

แตกไฟล์ preprocessed_thaiwikitext.zip ที่โหลดมา เปลี่ยนชื่อไฟล์ `thaiwikitext_sentseg` ข้างในให้เป็น .txt แล้วลบอีกไฟล์นึง (.xml) ทิ้งไป

## 1.2 Train sentencepiece model

ลง sentencepiece: `pip install sentencepiece`

เซฟโค้ดด้านล่างนี้เป็น .py อย่าลืมเปลี่ยน --input ให้ตรงกับที่ๆเก็บไฟล์ `thaiwikitext_sentseg.txt` ไว้แล้วก็รัน บนโน๊คบุคก็ได้เพราะว่าไม่ต้องใช้ GPU พอรันจบแล้วจะได้ไฟล์ออกมาสองไฟล์ `thaiwiki.model` และ `thaiwiki.vocab` สองไฟล์นี้คือ sentencepiece model ที่จะใช้ในขั้นตอนต่อไป
```
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=preprocessed_thaiwikitext/thaiwikitext_sentseg.txt \
--vocab_size=32000 --character_coverage=0.99995 --model_type=unigram \
--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> --user_defined_symbols=<eop>,.,(,),",-,–,£,€ \
--shuffle_input_sentence --model_prefix=thaiwiki')
```
## 2.1 Create tfrecord

clone xlnet repo: git clone https://github.com/zihangdai/xlnet.git เสร็จแล้วสร้างโฟลเดอร์ใหม่ชื่อ tf_record_out

จัดโครงสร้าง directory ให้เป็นแบบนี้

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
รันเสร็จแล้วจะได้ไฟล์ออกมาในโฟลเดอร์ `tf_record_out` แบบนี้
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

ฟิวด์ `filenames` เป็นตัวบอกว่าไฟล์ tfrecord ของเราอยู่ตรงไหน ซึ่งก็ตรงกับ option `--save_dir` จากข้อ 2.1 แต่ช้าก่อน เพราะว่ามันเป็น path local (local หมายถึงใน VM Colab นะ) ดังนั้นถ้าเราอัปโหลดใส่ bucket ไปทั้งๆแบบนี้ สิ่งที่จะเกิดขึ้นคือมันจะเกิด error ว่า "[local] filesystem not implemented" เพราะว่า TPU ไม่สามารถอ่านไฟล์ใน local ได้ อ่านได้จากใน bucket เท่านั้น ดังนั้นเราจะต้องแก้เป็นแบบนี้ก่อน (ไม่สามารถแก้ไขไฟล์บน Colab โดยตรงได้ ต้องโหลดลงมาที่เครื่องเราก่อน ลบไฟล์อันบน VM ทิ้ง แก้ไขที่เครื่องเราแล้วอัปโหลดกลับขึ้นไป)

```
{"filenames": ["gs://<bucket_name>/tf_record_out/tfrecords/train-0-0.bsz-32.seqlen-512.reuse-256.bi.alpha-6.beta-1.fnp-85.tfrecords"], "num_batch": 13329}
```

คือเปลี่ยน path ให้เป็น path บน bucket นั่นเอง แก้ไฟล์ให้เรียบร้อยใน local ก่อนแล้วค่อยอัปโหลดใส่ bucket ไปทั้งโฟลเดอร์ `tf_record_out` เลยครับ ส่วนโฟล์เดอร์บน local อย่าลบต้องเก็บไว้สำหรับตอนเทรน

## 3.1 เตรียมพร้อมสำหรับการเทรน
ยังครับ ยังเทรนไม่ได้ เราจะต้องแก้ไขไฟล์ใน repo xlnel บน VM อีกสองไฟล์ ก็คือ `modeling.py` และ `data_utils.py` วิธีที่ง่ายที่สุดคือ clone repo ไว้ที่เครื่องเราแล้วแก้ไขในเครื่องเราเลย ลบอันที่อยู่ใน VM ทิ้งแล้วอัปโหลดอันที่แก้ไขแล้วขึ้นไปแทน

สำหรับ `modeling.py` ให้ comment บรรทัดที่ 236 ออก อันนี้เค้า assert ว่า batch size ต้องหารสองลงตัวแต่มันเป็นบั๊ก (https://github.com/zihangdai/xlnet/issues/132) ก็ comment ทิ้งเลยเพราะว่า batch size เราเป็น 32 อยู่แล้ว

ส่วน `data_utils.py` ให้ comment บรรทัด 828-833 ออก (อันนี้แปลกใจว่าไม่มี issue ใน Github สงสัยไม่มีใครเทรนเองบน TPU 555) เรื่องของเรื่องคือมันดันไปเอา local path ของ tfrecord กลับมาใช้แทนที่จะเป็นอันที่เราแก้ไขเป็น bucket path ใน json ตะกี้นี้แล้ว (เพื่ออะไร ไม่เข้าใจเหมือนกัน) ทำให้เกิด error "[local] filesystem not implemented" อีกแล้ว ดังนั้นเม้นท์มันออกไปโลด

## 3.2 หาที่อยู่ของ TPU
อย่าลืมเปลี่ยน runtime type เป็น TPU นะจ๊ะ เสร็จแล้วก็รัน cell ต่อไปนี้ (ก๊อปมาจากตัวอย่าง TPU ของ Google เอง)

```
import os
import datetime
import json
import os
import pprint
import random
import string
import sys
import tensorflow as tf
assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!'
TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
print('TPU address is', TPU_ADDRESS) # copy the TPU address to the cell below

from google.colab import auth
auth.authenticate_user()
with tf.Session(TPU_ADDRESS) as session:
  print('TPU devices:')
  pprint.pprint(session.list_devices())

  # Upload credentials to TPU.
  with open('/content/adc.json', 'r') as f:
    auth_info = json.load(f)
  tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
  # Now credentials are set for all future sessions on this TPU.
```
สิ่งที่เราต้องการคือสตริง `TPU_ADDRESS` ที่จะใช้ตอนเทรน

## 3.3 เทรน
คำสั่งเทรน โมมาจาก https://github.com/ymcui/Chinese-PreTrained-XLNet/blob/master/README_EN.md
```
python train.py \
	--record_info_dir=../tf_record_out/tfrecords \
	--model_dir='gs://<bucket_name>/xlnet' \
	--train_batch_size=32 \
	--num_core_per_host=8 \
	--seq_len=512 \
	--reuse_len=256 \
	--mem_len=384 \
	--perm_size=256 \
	--n_layer=12 \
	--d_model=768 \
	--d_embed=768 \
	--n_head=8 \
	--d_head=64 \
	--d_inner=3072 \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85 \
	--uncased=False \
	--bi_data=True \
	--untie_r=True \
	--train_steps=2000000 \
	--save_steps=20000 \
	--warmup_steps=20000 \
	--max_save=20 \
	--weight_decay=0.01 \
	--adam_epsilon=1e-6 \
	--learning_rate=1e-4 \
	--dropout=0.1 \
	--dropatt=0.1 \
	--tpu='grpc://10.27.119.98:8470' \ 
	--use_tpu=True
```

โปรดสังเกต `train_batch_size` จะเท่ากับ `bsz_per_host` ตอนสร้าง tfrecord นอกจากนี้ `seq_len`, `reuse_len`, `mask_alpha`, `mask_beta`, `num_predict`, `uncased`, `bi_data` ก็จะตรงกันเป๊ะกับตอนสร้าง tfrecord เช่นเดียวกัน ต้องตรงกันนะครับไม่งั้น error แน่ๆ ส่วน `tpu` ให้ก๊อปมาจาก `TPU_ADDRESS` ในขั้นตอนที่แล้ว option อื่นๆเป็นโครงสร้างของโมเดลเช่นจำนวนเลเยอร์ และพวก hyperparameter ในการเทรน โครงสร้างโมเดลนี้จะประมาณขนาดเดียวกันกับ xlnet-base เลยยกเว้นจำนวน attention head เหลือ 8 จาก 12 เพราะว่าแรมไม่พอ ขนาด TPU นะเนี่ย

และสังเกตตรง `record_info_dir` มันเป็น local path ไปหาโฟลเดอร์ที่เราสร้าง tfrecord เอาไว้ อันนี้ต้องเป็น path local นะ ถึงแม้ว่าเราจะอัปโหลดดาต้าไปไว้บน bucket แล้วก็ตาม ดูตรงบรรทัดที่ 45 ใน `train.py` เค้าจะเขียนไว้เลยว่า "Path to local directory containing `record_info-lm.json`" ก็คือไอ้ไฟล์ชื่อ `record_info-train-0-0.bsz-32.seqlen-512.reuse-256.bi.alpha-6.beta-1.fnp-85.json` นั้นเอง ไฟล์ .json อันนี้ต้องอยู่ใน local ส่วนไฟล์ .tfrecords อ่ะต้องไปอยู่บน bucket นั่นแหละครับ debug กันเป็นวันๆ ก็ไอ้ตรงนี้แหละ T_T

เทรนด้วย config แบบนี้จะได้ความเร็วที่ประมาณ 1.85 steps/s. เทรนทั้งหมด 2 ล้าน step ก็ไม่นานเท่าไหร่แค่ประมาณ 12 วัน แต่เอาจริงๆคอยดู loss ถ้ามันไม่ลงแล้วก็เอา checkpoint ล่าสุดไปใช้ได้เลยครับ ถ้าเทรนเสร็จจะเอา weight มาปล่อยนะครับ
