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
model_xml_path = os.path.join(data_root_path, 'XLM', 'mlm_17_1280')
task_corpus_path = os.path.join(data_root_path, 'SemEval2021', 'task 2', 'ver2')


# print(tensor.device)
# print(torch.cuda.device_count())


