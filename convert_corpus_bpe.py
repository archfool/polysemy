import os
import fastBPE
from init_path import src_path, data_path, src_xml_path, model_xml_path, task_corpus_path

# def to_bpe(file_input, file_output, fastbpe_path, codes_path, vocab_path, data_path=''):
#
#     # apply bpe to tmp file
#     comment_line = "{fastbpe_path} applybpe {output} {input} {codes} {vocab}".format(
#         fastbpe_path=fastbpe_path,
#         output=file_output,
#         input=file_input,
#         codes=codes_path,
#         vocab=vocab_path
#     )
#     os.system(comment_line)
#     # os.system('%s applybpe %s %s %s' % (fastbpe_path, file_input, file_output, data_path))
#     # os.system('%s applybpe %s %s %s' % (fastbpe_path, file_input, file_output, codes))
#
#     # load bpe-ized sentences
#     sentences_bpe = []
#     # todo change file_input to file_output
#     with open(file_input, 'r', encoding='utf-8') as f:
#         for line in f:
#             sentences_bpe.append(line.rstrip())
#             print(line)
#
#     return sentences_bpe


if __name__ == '__main__':
    # if os.path.exists(u'c:\\'):
    #     root_path = u'E:\\'
    # else:
    #     root_path = u'~'
    #
    # data_root_path = os.path.join(root_path, u'data')
    # src_root_path = os.path.join(root_path, u'src')
    #
    # data_path = os.path.join(data_root_path, 'XLM')
    # src_path = os.path.join(src_root_path, 'XLM')

    # data_subpath = os.path.join(data_path, 'mlm_tlm_xnli15_1024')
    # model_file_name = u'mlm_tlm_xnli15_1024.pth'
    # model_path = os.path.join(data_subpath, model_file_name)

    codes_path = os.path.join(model_xml_path, 'codes_xnli_17.txt')
    vocab_path = os.path.join(model_xml_path, 'vocab_xnli_17.txt')
    fastbpe_path = os.path.join(src_xml_path, 'tools', 'fastBPE', 'fast')
    input_file = os.path.join(data_path, 'SemEval2021_Task2_corpus.txt')
    output_file = os.path.join(data_path, 'SemEval2021_Task2_corpus_bpe.txt')

    # apply bpe to tmp file
    # comment_line = "{fastbpe_path} applybpe {output} {input} {codes} {vocab}".format(
    #     fastbpe_path=fastbpe_path,
    #     output=input_file,
    #     input=output_file,
    #     codes=codes_path,
    #     vocab=vocab_path
    # )
    # os.system(comment_line)

    # with open(input_file, 'r', encoding='utf-8') as f:
    #     sentences = f.readlines()
    #
    # # bpe-ize sentences
    # fastbpe_path = os.path.join(src_path, 'tools', 'fastBPE', 'fast')
    # sentences = to_bpe(sentences, fastbpe_path, codes_path, vocab_path, data_subpath)
    # print('\n\n'.join(sentences))

    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = f.readlines()

    corpus_bpe = []
    bpe = fastBPE.fastBPE(codes_path, vocab_path)
    for idx, sent in enumerate(corpus):
        sent_bpe = bpe.apply([sent])[0]
        corpus_bpe.append(sent_bpe)
        if idx % 1000 == 0:
            print("{}:{}".format(idx, sent_bpe))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(corpus_bpe))

    print('END')
