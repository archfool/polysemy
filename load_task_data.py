import os
import json
import pandas as pd
import jieba
import copy
import fastBPE

from init_config import *

jieba.lcut("机器学习")


# 解析单条语料
def single_corpus_process(corpus, corpus_label):
    # 判断是否有标注文件（test数据集无标注文件）
    if corpus_label is None:
        label = None
    else:
        if corpus['id'] != corpus_label['id']:
            print('id mismatch in {} and {} !!!'.format(corpus['id'], corpus_label['id']))
            return None
        label = 1 if corpus_label['tag'] == 'T' else 0
    # 读取id
    id = corpus['id']
    if id == 'training.en-en.6025':
        print("")
    # 读取数据集类型：dev/training/trial/test
    # 读取任务赛道：en-en/en-zh等
    data_set, task_type, _ = id.split('.')
    # 读取语言类型：en/zh/ru/fr/ar
    sent1_lang, sent2_lang = task_type.split('-')
    # 判定语料类型：multilingual/crosslingual
    corpus_type = 'multilingual' if sent1_lang == sent2_lang else 'crosslingual'
    # 读取前一句语料，处理语料（中文语料要分词处理），给预测词打上序列标签
    sent1_bak = corpus['sentence1']
    if 'ranges1' in corpus.keys():
        sent1, sent1_keyword_tags = sent_keyword_tag(corpus['sentence1'], sent1_lang, ranges=corpus['ranges1'],
                                                     start=None, end=None)
    else:
        sent1, sent1_keyword_tags = sent_keyword_tag(corpus['sentence1'], sent1_lang, ranges=None,
                                                     start=corpus['start1'],
                                                     end=corpus['end1'])
    # 读取后一句语料，处理语料（中文语料要分词处理），给预测词打上序列标签
    sent2_bak = corpus['sentence2']
    if 'ranges2' in corpus.keys():
        sent2, sent2_keyword_tags = sent_keyword_tag(corpus['sentence2'], sent2_lang, ranges=corpus['ranges2'],
                                                     start=None, end=None)
    else:
        sent2, sent2_keyword_tags = sent_keyword_tag(corpus['sentence2'], sent2_lang, ranges=None,
                                                     start=corpus['start2'],
                                                     end=corpus['end2'])
    # 读取待预测词
    lemma = corpus['lemma']
    # 读取待预测词的词性
    pos = corpus['pos']
    corpus_dict = {
        'id': id,
        'data_set': data_set,
        'task_type': task_type,
        'sent1_lang': sent1_lang,
        'sent2_lang': sent2_lang,
        'corpus_type': corpus_type,
        'lemma': lemma,
        'pos': pos,
        'sent1': sent1,
        'sent1_keyword_tags': sent1_keyword_tags,
        'sent2': sent2,
        'sent2_keyword_tags': sent2_keyword_tags,
        'label': label,
    }
    return corpus_dict


# 给目标词打标记，返回标记序列
def sent_keyword_tag(sent: str, lang: str, ranges: str, start: str, end: str):
    # 将特殊的空格替换成正常空格
    sent = sent.replace('\xa0', ' ')
    # 提取目标词span信息
    span_list = []
    if ranges is None:
        span_list.append((int(start), int(end)))
    else:
        for r in ranges.split(','):
            s, e = r.split('-')
            span_list.append((int(s), int(e)))
        if len(span_list) >= 2:
            # print("本条语料包含两段以上的目标词：{}:{}".format(sent, ranges))
            pass
    span_list.sort()

    # 生成目标词tag标记序列
    # keyword_tags = [' ' if (token == ' ' or token == '\xa0') else '0' for token in list(sent)]
    keyword_tags = [' ' if (token == ' ') else '0' for token in list(sent)]
    for span in span_list:
        for i in range(span[0], span[1]):
            if keyword_tags[i] != ' ':
                keyword_tags[i] = '1'
    keyword_tags = ''.join(keyword_tags)

    # 对中文语料进行分词和修正
    if lang == 'zh':
        if True:
            word_list = jieba.lcut(sent)
            word_tag_list = []
            for word in word_list:
                word_tag_tmp = keyword_tags[:len(word)]
                word_tag_list.append(word_tag_tmp)
                keyword_tags = keyword_tags[len(word):]
            sent = ' '.join(word_list)
            keyword_tags = ' '.join(word_tag_list)
        else:
            word_tag_list = []
            token_list_0_tmp = []
            token_list_1_tmp = []
            for token, tag in zip(sent, keyword_tags):
                if tag == '0':
                    token_list_0_tmp.append(token)
                    if len(token_list_1_tmp) > 0:
                        word_tag_list.extend(cut_and_tag(token_list_1_tmp, '1'))
                        token_list_1_tmp = []
                else:
                    token_list_1_tmp.append(token)
                    if len(token_list_0_tmp) > 0:
                        word_tag_list.extend(cut_and_tag(token_list_0_tmp, '0'))
                        token_list_0_tmp = []
            if len(token_list_1_tmp) > 0:
                word_tag_list.extend(cut_and_tag(token_list_1_tmp, '1'))
            if len(token_list_0_tmp) > 0:
                word_tag_list.extend(cut_and_tag(token_list_0_tmp, '0'))
            sent = ' '.join([word for word, tag in word_tag_list])
            keyword_tags = ' '.join([tag for word, tag in word_tag_list])
        if len(sent) != len(keyword_tags):
            print("len(sent)!=len(keyword_tags):{}".format(sent))

    return sent, keyword_tags


# 中文语料的分词和打目标词标签
def cut_and_tag(token_list, tag):
    sent = ''.join(token_list)
    word_list = jieba.lcut(sent)
    word_tag_list = [(word, tag * len(word) if word != ' ' else ' ') for word in word_list]
    return word_tag_list


# 根据原语料、BPE语料、原语料tag，生成BPE语料tag
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
                keyword_tag_tmp = keyword_tag[:len(token_bpe_tmp) - 2] + "00"
                keyword_tag = keyword_tag[len(token_bpe_tmp) - 2:]
            else:
                keyword_tag_tmp = keyword_tag[:len(token_bpe_tmp)]
                keyword_tag = keyword_tag[len(token_bpe_tmp):]
            word_bpe += token_bpe_tmp
            word_bpe = word_bpe.rstrip('@@')
            bpe_keyword_tags.append(keyword_tag_tmp)
            # # 异常判断
            # if token_bpe_tmp.endswith('@@'):
            #     keyword_tag_tmp = keyword_tag_tmp[:-2]
            #     token_bpe_tmp = token_bpe_tmp[:-2]
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


# 语料读取和处理
def corpus_process(corpus_path, data_sets=None):
    if data_sets is None:
        data_sets = ['training', 'dev', 'trial', 'test']
    corpus_types = ['crosslingual', 'multilingual']
    corpus_list = []
    # 遍历数据集：训练集/测试集/验证集
    for data_set in data_sets:
        # 遍历语料类型：单语言/跨语言
        for corpus_type in corpus_types:
            data_path = os.path.join(corpus_path, data_set, corpus_type)
            if os.path.exists(data_path):
                print('loading data in {}'.format(data_path))
                # 遍历文件夹下的所有语料文件
                for file_name in os.listdir(data_path):
                    # 以data文件为锚点，同时读取data文件和gold文件
                    if file_name.endswith('data'):
                        # 读取data文件
                        with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
                            corpuses = json.loads(f.read())
                            print("{}:{}".format(file_name.split('.')[1], len(corpuses)))
                        # 读取gold文件（若为test数据集，则不读取）
                        if file_name.startswith('test'):
                            labels = [None] * len(corpuses)
                        else:
                            with open(os.path.join(data_path, file_name[:-4]) + 'gold', 'r', encoding='utf-8') as f:
                                labels = json.loads(f.read())
                        # 循环遍历处理单个语料文件下的所有语料
                        for corpus, label in zip(corpuses, labels):
                            corpus_dict = single_corpus_process(corpus, label)
                            # 异常判断：data和gold不匹配
                            if not isinstance(corpus_dict, dict):
                                print(corpus_dict)
                            corpus_list.append(corpus_dict)
    corpus_df = pd.DataFrame(corpus_list)
    return corpus_df


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
    # print("{}\n{}\n{}\n".format(k, k_bpe, ''.join([s_bpe_list[int(idx)] for idx in t_idx.split(' ')])))
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


if __name__ == '__main__':
    # codes_path = os.path.join(model_xml_path, 'codes_xnli_17.txt')
    # vocab_path = os.path.join(model_xml_path, 'vocab_xnli_17.txt')
    codes_path = os.path.join(model_xml_path, 'codes_xnli_15.txt')
    vocab_path = os.path.join(model_xml_path, 'vocab_xnli_15.txt')
    fastbpe_path = os.path.join(src_xml_path, 'tools', 'fastBPE', 'fast')
    corpus_csv_path = os.path.join(data_path, 'SemEval2021_Task2_corpus.csv')
    # corpus_txt_path = os.path.join(data_path, 'SemEval2021_Task2_corpus.txt')

    # 读取语料数据
    # data_sets = ['trial']
    data_sets = ['training', 'dev', 'trial', 'test']
    corpus_df = corpus_process(task_corpus_path, data_sets=data_sets)

    # 进行BPE分词转换
    bpe = fastBPE.fastBPE(codes_path, vocab_path)
    corpus_df['sent1_bpe'] = corpus_df['sent1'].apply(lambda sent: bpe.apply([sent])[0])
    corpus_df['sent2_bpe'] = corpus_df['sent2'].apply(lambda sent: bpe.apply([sent])[0])
    corpus_df['sent1_bpe_keyword_tags'] = corpus_df.apply(
        lambda row: generate_bpe_tag(row, ['sent1', 'sent1_bpe', 'sent1_keyword_tags']), axis=1)
    corpus_df['sent2_bpe_keyword_tags'] = corpus_df.apply(
        lambda row: generate_bpe_tag(row, ['sent2', 'sent2_bpe', 'sent2_keyword_tags']), axis=1)
    corpus_df['sent1_bpe_keyword_tags_simple'] = corpus_df['sent1_bpe_keyword_tags'].apply(
        lambda ss: ''.join(['1' if '1' in list(s) else '0' for s in ss.split()]))
    corpus_df['sent2_bpe_keyword_tags_simple'] = corpus_df['sent2_bpe_keyword_tags'].apply(
        lambda ss: ''.join(['1' if '1' in list(s) else '0' for s in ss.split()]))
    corpus_df['sent1_bpe_keyword_idx'] = corpus_df['sent1_bpe_keyword_tags'].apply(
        lambda ss: ' '.join([str(idx) for idx, tag in enumerate(['1' if '1' in list(s) else '0' for s in ss.split()]) if tag=='1']))
    corpus_df['sent2_bpe_keyword_idx'] = corpus_df['sent2_bpe_keyword_tags'].apply(
        lambda ss: ' '.join([str(idx) for idx, tag in enumerate(['1' if '1' in list(s) else '0' for s in ss.split()]) if tag=='1']))
    # keyword_tag_check(corpus_df)

    # 存储数据到文件
    corpus_df.to_csv(corpus_csv_path, sep='\001', index=None, encoding='utf-8')
    # with open(corpus_txt_path, 'w', encoding='utf-8') as f:
    #     for sent1, sent2 in zip(corpus_df['sent1_bpe'].to_list(), corpus_df['sent2_bpe'].to_list()):
    #         f.write(sent1 + '\n' + sent2 + '\n')

    print('END')
