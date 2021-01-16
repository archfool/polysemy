import os
import fastBPE
from init_path import src_path, data_path, src_xml_path, model_xml_path, task_corpus_path

if __name__ == '__main__':

    codes_path = os.path.join(model_xml_path, 'codes_xnli_17.txt')
    vocab_path = os.path.join(model_xml_path, 'vocab_xnli_17.txt')
    fastbpe_path = os.path.join(src_xml_path, 'tools', 'fastBPE', 'fast')
    input_file = os.path.join(data_path, 'SemEval2021_Task2_corpus.txt')
    output_file = os.path.join(data_path, 'SemEval2021_Task2_corpus_bpe.txt')

    # 读取原始语料
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = f.readlines()

    # 进行BPE分词转换
    corpus_bpe = []
    bpe = fastBPE.fastBPE(codes_path, vocab_path)
    for idx, sent in enumerate(corpus):
        sent_bpe = bpe.apply([sent])[0]
        corpus_bpe.append(sent_bpe)
        if idx % 10000 == 0:
            print("{}:{}".format(idx, sent_bpe))

    # 输出BPE语料
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(corpus_bpe))

    print('END')
