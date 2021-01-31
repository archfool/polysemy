# -*- coding: utf-8 -*-
"""
@author: ruanzhihao
Created on 2021/1/28 0028 下午 14:11
"""
import tensorflow as tf
import tokenization
from siamese_utils import InputFeatures_Polysemy, convert_single_example_polysemy
from init_config import *


# 从输入中构造迭代器，用于在线预测。
def gen_data_from_input(examples, tokenizer, max_seq_length):
    # seq_length = len(features[0].input_ids_a, max_seq_length)
    feature_info = {
        "input_ids_a": ([None, max_seq_length], tf.int64),
        "input_mask_a": ([None, max_seq_length], tf.int64),
        "segment_ids_a": ([None, max_seq_length], tf.int64),
        "input_ids_b": ([None, max_seq_length], tf.int64),
        "input_mask_b": ([None, max_seq_length], tf.int64),
        "segment_ids_b": ([None, max_seq_length], tf.int64),
    }

    # 构造输入
    def generate_input():
        for example in examples:
            feature = convert_single_example_polysemy(example=example, max_seq_length=max_seq_length,
                                                      tokenizer=tokenizer, random_mask=0)
            # print(feature.tokens_a, feature.tokens_b)
            yield {name: [feature.__getattribute__(name)] for name in feature_info.keys()}
            # yield {
            #     'text': [feature],
            #     # 'label': None
            # }

    # 根据输入，构造tf的数据生成器。
    def generate_iterator():
        dataset = tf.data.Dataset.from_generator(
            generate_input,
            output_shapes={key: value[0] for key, value in feature_info.items()},
            output_types={key: value[1] for key, value in feature_info.items()}
            # output_types={
            #     'text': tf.float32,
            #     # 'label': tf.int32
            # },
            # output_shapes={
            #     'text': (None, 4),
            #     # 'label': (None, 3)
            # }
        )
        iterator = dataset.make_one_shot_iterator()
        result = iterator.get_next()
        return result

    return lambda: generate_iterator()


if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_bert_path, 'vocab.txt'),
                                           do_lower_case=True)
    # feature = convert_single_example_polysemy(example="我爱机器学习", max_seq_length=128, tokenizer=tokenizer, random_mask=0)
    examples = [("我爱机器学习", "机器学习爱我"), ("我爱机器学习", "机器学习爱我")]
    gen_data_from_input(examples=examples, tokenizer=tokenizer, max_seq_length=128)
    print("END")
