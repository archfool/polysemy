import os

if os.path.exists(u'c:\\'):
    root_path = u'E:\\'
elif os.path.exists(u'/media/archfool/data'):
    root_path = u'/media/archfool/data/'
elif os.path.exists(u'/mnt/hgfs/'):
    root_path = u'/mnt/hgfs/'

src_root_path = os.path.join(root_path, u'src')
data_root_path = os.path.join(root_path, u'data')

src_path = os.path.join(src_root_path, 'polysemy')
data_path = os.path.join(data_root_path, 'polysemy')

src_xml_path = os.path.join(src_root_path, 'XLM')
# model_xml_path = os.path.join(data_root_path, 'XLM', 'mlm_17_1280')
model_xml_path = os.path.join(data_root_path, 'XLM', 'mlm_tlm_xnli15_1024')
task_corpus_path = os.path.join(data_root_path, 'SemEval2021', 'task 2', 'ver2')

model_bert_path = os.path.join(data_root_path, 'bert', 'uncased_L-24_H-1024_A-16')
model_bert_file = os.path.join(model_bert_path, 'bert_model.ckpt')
# model_bert_path = os.path.join(data_root_path, 'BERT-flow', 'exp', 'exp_large_1234')
# model_bert_file = os.path.join(model_bertflow_path, 'model.ckpt-60108')

model_bertflow_path = os.path.join(data_root_path, 'BERT-flow', 'exp', 'exp_t_STS-B_ep_1.00_lr_5.00e-05_e_avg-last-2_f_11_1.00e-03_allsplits')
model_bertflow_file = os.path.join(model_bertflow_path, 'model.ckpt-269')

corpus_embd_file_path = os.path.join(data_path, "corpus_embd.npy")
corpus_embd_file_path_format = os.path.join(data_path, "corpus_embd_{}.npy")
corpus_token_file_path = os.path.join(data_path, "corpus_token.txt")

corpus_csv_path = os.path.join(data_path, 'SemEval2021_Task2_corpus.csv')

# print(tensor.device)
# print(torch.cuda.device_count())


if __name__ == "__main__":
    import pickle
    from util_tools import logger
    logger.info("BEGIN")
    corpus_list = []
    f = open(corpus_embd_file_path, "rb")
    i = 0
    while True:
        try:
            corpus_list.append(pickle.load(f))
            i += 1
            # print(i)
        except:
            f.close()
            break
    logger.info("END")

