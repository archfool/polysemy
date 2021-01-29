#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : mc_lih
# @Time    : 2021/1/14 0014 11:04
# @File    : model.py
# 模型文件，定义模型的结构

import tensorflow as tf
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
import numpy as np


# 数据生成器
class DataReader:
    def __init__(self, data_file):
        self.data_file = data_file

    @staticmethod
    def read_data(row: str):
        """
        :param row: 一行数据，是一个string
        :return: 返回处理后的特征，以及标签
        """
        columns = tf.decode_csv(
            row,                             # 一行记录
            record_defaults=[[''], ['']],    # 文件形式
            field_delim='\t',                # 分隔符
            use_quote_delim=False
        )

        tmp_feature = {name: values for name, values in zip(['label', 'text'], columns) if name not in []}

        feature = {}

        input_text = tf.string_split([tmp_feature['text']], '|').values
        input_text = tf.string_to_number(input_text, out_type=tf.float32)

        label = tf.string_split([tmp_feature['label']], '|').values
        label = tf.string_to_number(label, out_type=tf.int32)

        feature['text'] = input_text

        return feature, label

    # 定义从文件中读取到dataset
    def input_fn_from_file(self, data_file: str, data_type: str = 'train', batch_size: int = 10):
        dataset = tf.data.TextLineDataset(data_file).skip(1)  #跳过首行
        if data_type == 'train':
            # 如果是训练数据，打乱（数值越大，混乱度越高）
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.map(DataReader.read_data).batch(batch_size)
        # 创建一个迭代器
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        features, labels = iterator.get_next()
        return features, labels

    # 返回的是一个数据迭代器
    def gen_data_from_file(self, data_file: str, data_type: str = 'train', batch_size: int = 10):
        data = lambda: self.input_fn_from_file(data_file=data_file, data_type=data_type, batch_size=batch_size)
        return data

    # 从输入中构造迭代器，用于在线预测。
    def gen_data_from_input(self, text):
        # 构造输入
        def generate_input():
            for each in text:
                yield {
                    'text': [each],
                    # 'label': None
                }

        # 根据输入，构造tf的数据生成器。
        def generate_iterator():
            dataset = tf.data.Dataset.from_generator(
                generate_input,
                output_types={
                    'text': tf.float32,
                    # 'label': tf.int32
                },
                output_shapes={
                    'text': (None, 4),
                    # 'label': (None, 3)
                }
            )
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()
            label = None
            return features, label

        return lambda: generate_iterator()


# 模型
class ModelName:

    def __init__(self, model_file: str, serving_file: str):
        # 初始化模型常量or其它参数。
        self.model_file = model_file
        self.serving_file = serving_file

    # 创建一个estimator高级api
    def model(self):
        # 定义模型所需要的参数
        param = {}
        # 设置运行配置，每10步保存一次
        run_config = tf.estimator.RunConfig(save_summary_steps=10)
        estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,         # 模型结构的定义函数
            model_dir=self.model_file,          # 模型训练后的tmp保存目录
            params=param,                   # 模型训练过程中可能需要的参数
            config=run_config               # 模型接口配置
        )
        return estimator

    # 定义模型训练以及结构函数
    def model_fn(self, features, labels, mode, params):
        # TODO: 编写模型结构
        text = features['text']
        # tf是静态图，所以入参需要reshape下
        text = tf.reshape(text, shape=[-1, 4])
        logits = tf.layers.dense(text, 3, name='dense_layer')
        # 模拟模型输出
        pre_label = tf.argmax(tf.nn.softmax(logits), 1, name='pred')

        predictions = {'pred': pre_label}
        # 预测, 必须放在第一个位置
        if mode == tf.estimator.ModeKeys.PREDICT:
            # 定义需要返回的数据，一般设置为字典
            export_outputs = {DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )

        # TODO：定义loss等损失函数，以及评估矩阵等
        # 模拟loss和评估矩阵(用字典)
        labels = tf.reshape(labels, shape=[-1, 3])
        true_label = tf.argmax(labels, axis=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name='loss', axis=0)

        eval_metric = {
            'accuracy': tf.metrics.accuracy(pre_label, true_label),
            'auc': tf.metrics.auc(pre_label, true_label),
            'recall': tf.metrics.recall(pre_label, true_label),
            'precision': tf.metrics.precision(pre_label, true_label)
        }

        # 评估模型, 放在第二个位置
        if mode == tf.estimator.ModeKeys.EVAL:
            # 定义需要返回的数据，一般设置为字典
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric
            )

        # TODO: 定义优化方式（梯度下降等）
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss, global_step=tf.train.get_global_step())
        # 训练模型, 放在最后位置
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 定义需要返回的数据，一般设置为字典
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=optimizer
            )

    # 定义模型导出的结构，用于serving
    @staticmethod
    def customer_serving_fn(**params):
        def serving_input_receiver_fn():
            # 这里定义的结构，跟model_fn中的features
            receiver = {
                'text': tf.placeholder(dtype=tf.float32, shape=(None, 4)),
                'label': tf.placeholder(dtype=tf.int32, shape=(None, 3))
            }
            features = {
                'text': receiver['text'],
                'label': receiver['label']
            }
            return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=receiver)
        return serving_input_receiver_fn


# 训练、评估、预测、在线预测模型
class RunModel:

    def __init__(self, reader, model):
        # 定义数据
        self.data_reader = reader
        self.data = self.data_reader.gen_data_from_file(data_file=self.data_reader.data_file)
        # 定义模型
        self.model = model
        self.estimator = self.model.model()

    def train(self, epoch):
        for i in range(epoch):
            self.estimator.train(input_fn=self.data)
            self.eval()
        self.save_model_for_serving()

    def eval(self):
        result = self.estimator.evaluate(input_fn=self.data)
        print(result)

    def predict(self):
        result = self.estimator.predict(input_fn=self.data)
        for i in result:
            print(i)

    def predict_online(self, text):
        result = self.estimator.predict(input_fn=self.data_reader.gen_data_from_input(text=text))
        for i in result:
            print(i)

    # 保存模型，tf-serving
    def save_model_for_serving(self):
        self.estimator.export_savedmodel(self.model.serving_file, self.model.customer_serving_fn(), as_text=True)


if __name__ == '__main__':
    data_reader = DataReader(data_file=r'./test.txt')
    my_model = ModelName(model_file=r'./model', serving_file=r'./serving')
    run_model = RunModel(data_reader, my_model)
    run_model.train(epoch=1)
    run_model.eval()
    run_model.predict()
    run_model.predict_online(np.asarray([[5, 12, 145, 100], [5, 12, 145, 100]], dtype=np.float))
    run_model.predict_online(np.asarray([[5, 12, 145, 100], [443, 13, 10, 1432534]], dtype=np.float))

