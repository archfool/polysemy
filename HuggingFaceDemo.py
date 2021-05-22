# -*- coding: utf-8 -*-
"""
@author: ruanzhihao
Created on 2021/4/4 0004 下午 19:39
"""

import os
import sys
import torch
from scipy.spatial.distance import cosine
# sys.path.append("../transformers/src")
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import AutoModel, AutoTokenizer


def demo1():
    # 这里我们调用bert-base模型，同时模型的词典经过小写处理
    model_name = 'bert-base-uncased'
    # 读取模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # 载入模型
    model = BertModel.from_pretrained(model_name)
    # 输入文本
    input_text = "Here is some text to encode"
    # 通过tokenizer把文本变成 token_id
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
    input_ids = torch.tensor([input_ids])
    # 获得BERT模型最后一个隐层结果
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples


def demo2():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    # 从下载好的文件夹中加载tokenizer
    # 这里你需要改为自己的实际文件夹路径
    tokenizer = GPT2Tokenizer.from_pretrained('/dfsdata2/yucc1_data/models/huggingface/gpt2')
    text = 'Who was Jim Henson ? Jim Henson was a'
    # 编码一段文本
    # 编码后为[8241, 373, 5395, 367, 19069, 5633, 5395, 367, 19069, 373, 257]
    indexed_tokens = tokenizer.encode(text)
    # 转换为pytorch tensor
    # tensor([[ 8241,   373,  5395,   367, 19069,  5633,  5395,   367, 19069,   373, 257]])
    # shape为 torch.Size([1, 11])
    tokens_tensor = torch.tensor([indexed_tokens])
    # 从下载好的文件夹中加载预训练模型
    model = GPT2LMHeadModel.from_pretrained('/dfsdata2/yucc1_data/models/huggingface/gpt2')

    # 设置为evaluation模式，去取消激活dropout等模块。
    # 在huggingface/transformers框架中，默认就是eval模式
    model.eval()

    # 预测所有token
    with torch.no_grad():
        # 将输入tensor输入，就得到了模型的输出，非常简单
        # outputs是一个元组，所有huggingface/transformers模型的输出都是元组
        # 本初的元组有两个，第一个是预测得分（没经过softmax之前的，也叫作logits），
        # 第二个是past，里面的attention计算的key value值
        # 此时我们需要的是第一个值
        outputs = model(tokens_tensor)
        # predictions shape为 torch.Size([1, 11, 50257])，
        # 也就是11个词每个词的预测得分（没经过softmax之前的）
        # 也叫做logits
        predictions = outputs[0]

    # 我们需要预测下一个单词，所以是使用predictions第一个batch，最后一个词的logits去计算
    # predicted_index = 582，通过计算最大得分的索引得到的
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    # 反向解码为我们需要的文本
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    # 解码后的文本：'Who was Jim Henson? Jim Henson was a man'
    # 成功预测出单词 'man'
    print(predicted_text)


def demo3():
    model = AutoModel.from_pretrained('bert-base-uncased')


def demo4():
    from transformers import BertTokenizer, BertForQuestionAnswering
    import torch

    MODEL_PATH = r"D:\transformr_files\bert-base-uncased/"
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(r"D:\transformr_files\bert-base-uncased\bert-base-uncased-vocab.txt")
    # 导入bert的model_config
    model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
    # 首先新建bert_model
    bert_model = transformers.BertModel.from_pretrained(MODEL_PATH, config=model_config)
    # 最终有两个输出，初始位置和结束位置（下面有解释）
    model_config.num_labels = 2
    # 同样根据bert的model_config新建BertForQuestionAnswering
    model = BertForQuestionAnswering(model_config)
    model.bert = bert_model

    # 设定模式
    model.eval()
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    # 获取input_ids编码
    input_ids = tokenizer.encode(question, text)
    # 手动进行token_type_ids编码，可用encode_plus代替
    # input_ids = tokenizer.encode_plus("i like you", "but not him")
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    # 得到评分,
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    # 进行逆编码，得到原始的token
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'nice', 'puppet', '[SEP]']
    # 对输出的答案进行解码的过程
    answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
    # assert answer == "a nice puppet"
    # 这里因为没有经过微调，所以效果不是很好，输出结果不佳。
    print(answer)
    # 'was jim henson ? [SEP] jim henson was a nice puppet [SEP]'


def demo5():
    from transformers import XLNetConfig, XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
    import torch
    # 定义路径，初始化tokenizer
    XLN_PATH = r"D:\transformr_files\XLNetLMHeadModel"
    tokenizer = XLNetTokenizer.from_pretrained(XLN_PATH)
    # 加载配置
    model_config = XLNetConfig.from_pretrained(XLN_PATH)
    # 设定类别数为3
    model_config.num_labels = 3
    # 直接从xlnet的config新建XLNetForSequenceClassification(和上一节方法等效)
    cls_model = XLNetForSequenceClassification.from_pretrained(XLN_PATH, config=model_config)
    # 设定模式
    model.eval()
    token_codes = tokenizer.encode_plus("i like you, what about you")


def demo():
    from transformers import BertTokenizer, BertForQuestionAnswering
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo_simcse():
    path_model_simcse_sup = r'E:\data\huggingface\sup-simcse-bert-base-uncased'
    path_model_simcse_unsup = r'E:\data\huggingface\unsup-simcse-bert-base-uncased'
    if True:
        model = AutoModel.from_pretrained(path_model_simcse_unsup)
        tokenizer = AutoTokenizer.from_pretrained(path_model_simcse_unsup)

        # Tokenize input texts
        texts = [
            "There's a kid on a skateboard.",
            "A kid is skateboarding.",
            "A kid is inside the house."
        ]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
        cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

        print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
        print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))

    else:
        from simcse import SimCSE
        model = SimCSE(path_model_simcse_unsup)
        embeddings = model.encode("A woman is reading.")
        sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
        sentences_b = ['He plays guitar.', 'A woman is making a photo.']
        similarities = model.similarity(sentences_a, sentences_b)

    return model, tokenizer




if __name__ == "__main__":
    model, tokenizer = demo_simcse()
    model_paras = [(para_name, para_value) for para_name, para_value in model.named_parameters()]

    path_model_bert = r'E:\data\huggingface\bert-base-uncased'

    tokenizer = transformers.BertTokenizer.from_pretrained(path_model_bert)
    model_config = transformers.BertConfig.from_pretrained(path_model_bert)
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    model = transformers.BertModel.from_pretrained(path_model_bert, config=model_config)

    input_ids = tokenizer.encode("I love machine learning!", add_special_tokens=True)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_ids)  # Models outputs are now tuples
        last_hidden_states = outputs[0]
        print(last_hidden_states)

    print('END')
