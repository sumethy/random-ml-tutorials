# วิธีการ pretrain XLNET สำหรับภาษาไทย

## 1.1 Get and prepare training data

download training data (วิกิไทยที่คลีนเรียบร้อยแล้ว) `https://drive.google.com/file/d/1QZSOpikO6Qc02gRmyeb_UiRLtTmUwGz1/view?usp=sharing`

แตกไฟล์ preprocessed_thaiwikitext.zip ที่โหลดมา เปลี่ยนชื่อไฟล์ thaiwikitext_sentseg ข้างในให้เป็น .txt แล้วลบอีกไฟล์นึง (.xml) ทิ้งไป

## 1.2 Train sentencepiece model

Install sentencepiece: `pip install sentencepiece`

```
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=preprocessed_thaiwikitext/thaiwikitext_sentseg.txt \
--vocab_size=32000 --character_coverage=0.99995 --model_type=unigram \
--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> --user_defined_symbols=<eop>,.,(,),",-,–,£,€ \
--shuffle_input_sentence --model_prefix=thaiwiki')
```
