import os
import torch
import pandas as pd

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

from init_path import *

batch_size = 10

# 加载模型并预测
def infer(model_path, sentences):
    # Reload a pretrained model
    if torch.cuda.is_available():
        print("torch.cuda.is_available={}".format(torch.cuda.is_available()))
        reloaded = torch.load(model_path)
    else:
        print("torch.cuda.is_available={}".format(torch.cuda.is_available()))
        reloaded = torch.load(model_path, map_location=torch.device('cpu'))
    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    model.eval()
    model.load_state_dict(reloaded['model'])

    # check how many tokens are OOV
    n_w = len([w for w in ' '.join(sentences).split()])
    n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])
    print('Number of out-of-vocab words: %s/%s' % (n_oov, n_w))

    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]

    # Create batch
    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])

    # NOTE: No more language id (removed it in a later version)
    # langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs) if params.n_langs > 1 else None
    langs = None

    # Forward
    tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
    print(tensor.size())
    return tensor


if __name__ == '__main__':
    corpus_bpe_path = os.path.join(data_path, 'SemEval2021_Task2_corpus.txt')
    with open(corpus_bpe_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    # corpus_df = pd.read_csv(os.path.join(data_path, 'SemEval2021_Task2_corpus.csv'), sep='\001', encoding='utf-8')

    model_path = os.path.join(model_xml_path, u'mlm_17_1280.pth')
    tensor = infer(model_path, sentences[-batch_size:])
    tensor_couple_list = [(tensor[:, i, :], tensor[:, i + 1, :]) for i in range(0, batch_size, 2)]

    print('END')
