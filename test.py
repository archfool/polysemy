import pandas as pd
import copy
from init_path_config import *

def generate_bpe_tag(row, columns_name):
    sent = row[columns_name[0]].split()
    sent_bpe = row[columns_name[1]].split()
    sent_bpe_bak = copy.deepcopy(sent_bpe)
    keyword_tags = row[columns_name[2]].split()
    bpe_keyword_tags = []
    for word, keyword_tag in zip(sent, keyword_tags):
        word_bpe = ""
        while True:
            token_bpe_tmp = sent_bpe.pop(0)
            if token_bpe_tmp.endswith('@@'):
                keyword_tag_tmp = keyword_tag[:len(token_bpe_tmp)-2]+"00"
                keyword_tag = keyword_tag[len(token_bpe_tmp)-2:]
            else:
                keyword_tag_tmp = keyword_tag[:len(token_bpe_tmp)]
                keyword_tag = keyword_tag[len(token_bpe_tmp):]
            word_bpe += token_bpe_tmp
            word_bpe = word_bpe.rstrip('@@')
            bpe_keyword_tags.append(keyword_tag_tmp)
            # # 异常判断
            # if token_bpe_tmp.endswith('@@'):
            #     keyword_tag_tmp = keyword_tag_tmp[:-2]
            # if token_bpe_tmp == '@@':
            #     pass
            # if token_bpe_tmp.endswith('.') \
            #         or token_bpe_tmp.endswith('-') \
            #         or token_bpe_tmp.endswith(')') \
            #         or token_bpe_tmp.endswith("'") \
            #         or token_bpe_tmp.endswith('"'):
            #     pass
            # elif len(set(keyword_tag_tmp)) != 1:
            #     print("ERROR: [{}/{}/{}] multi-tag: {} in {}".format(
            #         columns_name[0],
            #         len(keyword_tag_tmp),
            #         ''.join(keyword_tag_tmp),
            #         token_bpe_tmp,
            #         sent))
            if word == word_bpe:
                break
    return ' '.join(bpe_keyword_tags)


def keyword_tag_check_single(idx, s, t, s_bpe, t_bpe, t_bpe_simple, t_idx):
    k = ''
    for word, tag in zip(list(s), list(t)):
        if tag == '1':
            k += word
    k_bpe = ''
    for word, tag in zip(list(s_bpe), list(t_bpe)):
        if tag == '1':
            k_bpe += word
    s_bpe_list = s_bpe.split(' ')
    print("{}\n{}\n{}\n".format(k, k_bpe, ''.join([s_bpe_list[int(idx)] for idx in t_idx.split(' ')])))
    # 检查BPE前后，提取的文本是否一致
    if k != k_bpe:
        print("ERROR:{}\n{}\n{}".format(idx, k, k_bpe))
    # 检查BPE的seg和tag的长度是否一致
    if len(s_bpe.split(' ')) != len(t_bpe_simple):
        print("ERROR(len_unequal):[{}]{}".format(idx, len(t_bpe_simple)))
    # 检查是否会受到strip()影响
    if len(s_bpe.strip().split(' ')) != len(t_bpe_simple):
        print("ERROR(strip):[{}]{}".format(idx, len(t_bpe_simple)))


def keyword_tag_check(df: pd.DataFrame):
    idx = -1
    for idx, row in df.iterrows():
        # if row['sent1_lang'] != 'zh':
        #     continue
        if idx % 1000 == 0:
            print(idx)
        s = row['sent1']
        t = row['sent1_keyword_tags']
        s_bpe = row['sent1_bpe']
        t_bpe = row['sent1_bpe_keyword_tags']
        t_bpe_simple = row['sent1_bpe_keyword_tags_simple']
        t_idx = row['sent1_bpe_keyword_idx']
        keyword_tag_check_single(idx, s, t, s_bpe, t_bpe, t_bpe_simple, t_idx)
        s = row['sent2']
        t = row['sent2_keyword_tags']
        s_bpe = row['sent2_bpe']
        t_bpe = row['sent2_bpe_keyword_tags']
        t_bpe_simple = row['sent2_bpe_keyword_tags_simple']
        t_idx = row['sent2_bpe_keyword_idx']
        keyword_tag_check_single(idx, s, t, s_bpe, t_bpe, t_bpe_simple, t_idx)
    print(idx)



if __name__=="__main__":
    # with open(os.path.join(src_path, 'log.txt'), 'r', encoding='utf-8') as f:
    #     x = f.readlines()

    corpus_df = pd.read_csv(os.path.join(data_path, 'SemEval2021_Task2_corpus.csv'), sep='\001', encoding='utf-8')
    keyword_tag_check(corpus_df)
    sent1 = corpus_df['sent1']
    tag1 = corpus_df['sent1_keyword_tags']
    sent2 = corpus_df['sent2']
    tag2 = corpus_df['sent2_keyword_tags']
    # row = corpus_df.iloc[6000]
    # generate_bpe_tag(row, ['sent2', 'sent2_bpe', 'sent2_keyword_tags'])
    print('END')
