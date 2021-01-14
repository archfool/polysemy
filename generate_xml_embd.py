import os
import torch

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

from init_path import src_path, data_path, src_xml_path, model_xml_path, task_corpus_path


# 加载模型并预测
def infer(model_path, sentences):
    # Reload a pretrained model
    if torch.cuda.is_available():
        reloaded = torch.load(model_path)
    else:
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

    # list of (sentences, lang)
    sentences = [
        'once he had worn trendy italian leather shoes and jeans from paris that had cost three hundred euros .',  # en
        'Le français est la seule langue étrangère proposée dans le système éducatif .',  # fr
        'El cadmio produce efectos tóxicos en los organismos vivos , aun en concentraciones muy pequeñas .',  # es
        'Nach dem Zweiten Weltkrieg verbreitete sich Bonsai als Hobby in der ganzen Welt .',  # de
        'وقد فاز في الانتخابات في الجولة الثانية من التصويت من قبل سيدي ولد الشيخ عبد الله ، مع أحمد ولد داداه في المرتبة الثانية .',
        # ar
        '羅伯特 · 皮爾 斯 生於 1863年 , 在 英國 曼徹斯特 學習 而 成為 一 位 工程師 . 1933年 , 皮爾斯 在 直布羅陀去世 .',  # zh
    ]

    model_path = os.path.join(model_xml_path, u'mlm_17_1280.pth')
    tensor = infer(model_path, sentences)
    print(tensor[0, :, :])

    print('END')
