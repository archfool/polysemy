import os

if os.path.exists(u'c:\\'):
    root_path = u'E:\\'
else:
    root_path = u'/home/archfool'

src_root_path = os.path.join(root_path, u'src')
data_root_path = os.path.join(root_path, u'data')

src_path = os.path.join(src_root_path, 'polysemy')
data_path = os.path.join(data_root_path, 'polysemy')

src_xml_path = os.path.join(src_root_path, 'XLM')
model_xml_path = os.path.join(data_root_path, 'XLM', 'mlm_17_1280')
task_corpus_path = os.path.join(data_root_path, 'SemEval2021', 'task 2', 'ver2')

