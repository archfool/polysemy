import pandas as pd
import copy
from init_path import *

def generate_bpe_tag(row, columns_name):
    sent = row[columns_name[0]].split()
    sent_bpe = row[columns_name[1]].split()
    sent_bpe_bak = copy.deepcopy(sent_bpe)
    keyword_tags = row[columns_name[2]].split()
    bpe_keyword_tags = []
    for word, keyword_tag in zip(sent, keyword_tags):
        word_bpe = ""
        while True:
            token_bpe_tmp = sent_bpe.pop(0).rstrip('@')
            keyword_tag_tmp = keyword_tag[:len(token_bpe_tmp)]
            keyword_tag = keyword_tag[len(token_bpe_tmp):]
            if token_bpe_tmp.endswith('.')\
                    or token_bpe_tmp.endswith('-')\
                    or token_bpe_tmp.endswith(')')\
                    or token_bpe_tmp.endswith("'")\
                    or token_bpe_tmp.endswith('"'):
                pass
            elif len(set(keyword_tag_tmp)) != 1:
                print("ERROR: [{}/{}/{}] multi-tag: {} in {}".format(
                    columns_name[0],
                    len(keyword_tag_tmp),
                    ''.join(keyword_tag_tmp),
                    token_bpe_tmp,
                    sent))
            word_bpe += token_bpe_tmp
            bpe_keyword_tags.append(keyword_tag_tmp)
            if word == word_bpe:
                break
    return bpe_keyword_tags


def keyword_tag_check_single(s, t, s_bpe, t_bpe):
    if len(s.split()) != len(t.split()):
        print('ERROR:keyword_tag_check')
    if len(s_bpe.split()) != len(t_bpe.split()):
        print('ERROR:keyword_tag_check')
    s = ''.join(s)
    t = ''.join(t)
    k = []
    for word, tag in zip(s, t):
        if tag == '1':
            k += word
    s_bpe = ''.join(s_bpe).replace('@', '')
    t_bpe = ''.join(t_bpe)
    k_bpe = ''
    for word, tag in zip(s_bpe, t_bpe):
        if tag == '1':
            k_bpe += word
    print("{}\n{}".format(k, k_bpe))
    return None


def keyword_tag_check(df: pd.DataFrame):
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(idx)
        s = row['sent1']
        t = row['sent1_keyword_tags']
        s_bpe = row['sent1_bpe']
        t_bpe = row['sent1_bpe_keyword_tags']
        keyword_tag_check_single(s, t, s_bpe, t_bpe)
        s = row['sent2']
        t = row['sent2_keyword_tags']
        s_bpe = row['sent2_bpe']
        t_bpe = row['sent2_bpe_keyword_tags']
        keyword_tag_check_single(s, t, s_bpe, t_bpe)


if __name__=="__main__":
    with open(os.path.join(src_path, 'log.txt'), 'r', encoding='utf-8') as f:
        x = f.readlines()

    corpus_df = pd.read_csv(os.path.join(data_path, 'SemEval2021_Task2_corpus.csv'), sep='\001', encoding='utf-8')
    keyword_tag_check(corpus_df)
    sent1 = corpus_df['sent1']
    tag1 = corpus_df['sent1_keyword_tags']
    sent2 = corpus_df['sent2']
    tag2 = corpus_df['sent2_keyword_tags']
    row = corpus_df.iloc[6025]
    generate_bpe_tag(row, ['sent2', 'sent2_bpe', 'sent2_keyword_tags'])
    print('END')
