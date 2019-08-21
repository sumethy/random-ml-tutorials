import tensorflow as tf
import csv

def _read_vocab(input_file):
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter = "\t")
        lines = []
        tokens = []
        for line in reader:
            lines.append(line)
            tokens.append(line[0])
        return lines,tokens

lines, vocab = _read_vocab('th_wiki_bpe/th.wiki.bpe.op25000.vocab')

with open('vocab.txt', 'w', encoding="utf-8") as outfile:
    for v in vocab:
        outfile.write(v)
        outfile.write('\n')