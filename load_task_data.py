import os
import json
import pandas as pd
import jieba
from init_path import src_path, data_path, src_xml_path, model_xml_path, task_corpus_path

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
    # 读取数据集类型：dev/training/trial/test
    # 读取任务赛道：en-en/en-zh等
    data_set, task_type, _ = id.split('.')
    # 读取语言类型：en/zh/ru/fr/ar
    sent1_lang, sent2_lang = task_type.split('-')
    # 判定语料类型：multilingual/crosslingual
    corpus_type = 'multilingual' if sent1_lang == sent2_lang else 'crosslingual'
    # 读取前一句语料，处理语料（中文语料要分词处理），给预测词打上序列标签
    if 'ranges1' in corpus.keys():
        sent1, sent1_keyword_tags = sent_keyword_tag(corpus['sentence1'], sent1_lang, ranges=corpus['ranges1'],
                                                     start=None, end=None)
    else:
        sent1, sent1_keyword_tags = sent_keyword_tag(corpus['sentence1'], sent1_lang, ranges=None,
                                                     start=corpus['start1'],
                                                     end=corpus['end1'])
    # 读取后一句语料，处理语料（中文语料要分词处理），给预测词打上序列标签
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
    keyword_tags = ['0' for _ in list(sent)]
    for span in span_list:
        for i in range(span[0], span[1]):
            keyword_tags[i] = '1'
    keyword_tags = ''.join(keyword_tags)

    # 对中文语料进行分词和修正
    if lang == 'zh':
        word_tag_list = []
        token_list_0 = []
        token_list_1 = []
        for token, tag in zip(sent, keyword_tags):
            if tag == '0':
                token_list_0.append(token)
                if len(token_list_1) > 0:
                    word_tag_list.extend(cut_and_tag(token_list_1, '1'))
                    token_list_1 = []
            else:
                token_list_1.append(token)
                if len(token_list_0) > 0:
                    word_tag_list.extend(cut_and_tag(token_list_0, '0'))
                    token_list_0 = []
        if len(token_list_1) > 0:
            word_tag_list.extend(cut_and_tag(token_list_1, '1'))
        if len(token_list_0) > 0:
            word_tag_list.extend(cut_and_tag(token_list_0, '0'))
        sent = ' '.join([word for word, tag in word_tag_list])
        keyword_tags = ' '.join([tag for word, tag in word_tag_list])
        if len(sent) != len(keyword_tags):
            print("len(sent)!=len(keyword_tags):{}".format(sent))

    return sent, keyword_tags


# 中文语料的分词和打目标词标签
def cut_and_tag(token_list, tag):
    sent = ''.join(token_list)
    word_list = jieba.lcut(sent)
    word_tag_list = [(word, tag * len(word)) for word in word_list]
    return word_tag_list


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


if __name__ == '__main__':

    # data_sets = ['trial']
    data_sets = ['training', 'dev', 'trial', 'test']
    corpus_df = corpus_process(task_corpus_path, data_sets=data_sets)

    corpus_df.to_csv(os.path.join(data_path, 'SemEval2021_Task2_corpus.csv'), sep='\001', index=None, encoding='utf-8')
    with open(os.path.join(data_path, 'SemEval2021_Task2_corpus.txt'), 'w', encoding='utf-8') as f:
        for sent1, sent2 in zip(corpus_df['sent1'].to_list(), corpus_df['sent2'].to_list()):
            f.write(sent1 + '\n' + sent2 + '\n')

    print('END')
