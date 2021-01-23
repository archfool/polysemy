# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:25:39 2021

@author: yaoxiaoyuan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:01:04 2020

@author: yaoxiaoyuan
"""
import os
import torch
from torch import optim
import re
import random
import logging
from datetime import datetime
import numpy as np



def train(params, model_fn, generator_fn, loss_fn, eval_fn=None):
    """
    """
    logger = build_logger()
    
    model = model_fn(params)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("%s" % model)
    logger.info("Total Model Params:%s" % total_params)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           params["lr"])

    model_dir = params["model_dir"]
    if not os.path.exists(model_dir):
        try:
            os.mkdir(model_dir)
            logger.info("Create model dir success!")
        except:
            model_dir = "./"
            logger.info("Change model dir to current dir.")
    else:
        logger.info("Model dir already exists!")
    
    start_epoch = 1
    start_steps = 1
    total_steps = 1
    if params["reload"] == True:
        model.load_state_dict(torch.load(
                params["reload_model"],
                map_location=lambda storage, loc: storage),
                False)
        logger.info("Reload model complete!")
        
        opti_path = params["reload_model"] + ".optimizer"
        if os.path.exists(opti_path):
            optimizer.load_state_dict(torch.load(
                    opti_path,
                    map_location=lambda storage, loc: storage))
            logger.info("Reload optimizer complete!")
    
    logger.info("Train Start!")
    
    generator = generator_fn(params)

    def save_model(model_name=None):
        """
        """
        if model_name is None:
            model_name = "%d.%d.%d.%s" % (epoch, 
                                          steps,
                                          total_steps,
                                          params["save_model"])
        model_path = os.path.join(params["model_dir"], model_name)
        torch.save(model.state_dict(), model_path, 
                   _use_new_zipfile_serialization=False)
        torch.save(optimizer.state_dict(), model_path + ".optimizer", 
                   _use_new_zipfile_serialization=False)
        logger.info("Save model to %s" % model_path)

    accumulate_steps = params.get("accumulate_steps", 1)
    save_steps = params["save_steps"]
    history_loss = []
    for epoch in range(start_epoch, params["max_epoch"] + 1): 
        model.train()
        
        steps = 1
        if epoch == start_epoch:
            steps = start_steps
            
        for inputs,target in generator(params["train_dir"], params["batch_size"]):
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, target, params)
            
            history_loss = history_loss[-999:] + [loss.item()]
            ma_loss = sum(history_loss) / len(history_loss)
            logger.info(
                    "%d epoch %d step total %d steps loss: %.3f" % 
                    (epoch, steps, total_steps, ma_loss)
                    )
            
            loss = loss / accumulate_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           params["grad_clip"])
           
            if total_steps % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if save_steps > 0 and total_steps % save_steps == 0:
                save_model()
            
            steps += 1
            total_steps += 1
    
        if eval_fn:
            eval_fn(model, generator, params)
    
        logger.info("shuffle train data...")
        shuffle_data(params["data_dir"], params["train_dir"])
        logger.info("shuffle train data completed!")
        
    save_model()
    logger.info("Train Completed!")

def shuffle_data(data_dir, dest_dir):
    """
    """
    for f in os.listdir(data_dir):
        lines = [line for line in open(os.path.join(data_dir, str(f)), "rb")]
        random.shuffle(lines)
        fo = open(os.path.join(dest_dir, str(f)), "wb")
        for line in lines:
            fo.write(line)
        fo.close()
        
        
def build_logger():
    """
    """
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    filename = datetime.today().strftime('%Y-%m-%d-%H-%M-%S.log')
    logging.basicConfig(filename=filename, 
                        level=logging.INFO,
                        format=format_str)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    
    return logger

def build_transformer_model(train_params):
    """
    """
    src_vocab_size = train_params["src_vocab_size"]
    src_max_len = train_params["src_max_len"]
    trg_vocab_size = train_params["trg_vocab_size"]
    trg_max_len = train_params["trg_max_len"]
    n_heads = train_params["n_heads"]
    d_model = train_params["d_model"]
    d_ff = train_params["d_ff"]
    n_enc_layers = train_params["n_enc_layers"]
    n_dec_layers = train_params["n_dec_layers"]
    dropout = train_params.get("dropout", 0)
    share_src_trg_emb = train_params["share_src_trg_emb"]
    share_emb_out_proj = train_params.get("share_emb_out_proj", False)
    
    transformer = Transformer(constants["symbols"],
                              src_vocab_size, src_max_len, trg_vocab_size, 
                              trg_max_len, n_heads, d_model, d_ff, 
                              n_enc_layers, n_dec_layers, dropout, 
                              share_src_trg_emb, share_emb_out_proj)
    if train_params["use_cuda"]:
        transformer = transformer.cuda()
            
    return transformer


class BatchGenerator():
    """
    """
    def __init__(self, src_max_len, trg_max_len, src_vocab, trg_vocab,
                 src_tokenizer, trg_tokenizer, symbols, use_cuda):
        """
        """
        self._src_max_len = src_max_len
        self._trg_max_len = trg_max_len
        self._src_word2id = load_vocab(real_path(src_vocab))
        self._trg_word2id = load_vocab(real_path(trg_vocab))
        self._src_tokenizer = Tokenizer(tokenizer=src_tokenizer,
                                        vocab=self._src_word2id)
        self._trg_tokenizer = Tokenizer(tokenizer=trg_tokenizer,
                                        vocab=self._trg_word2id)
        self.PAD = symbols["PAD"]
        self.BOS = symbols["BOS"]
        self.EOS = symbols["EOS"]
        self.UNK = symbols["UNK"]
        self.use_cuda = use_cuda
    
    
    def _vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        x = self.PAD + np.zeros((batch_size, self._src_max_len), dtype=np.long)
        y = self.PAD + np.zeros((batch_size, self._trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, self._trg_max_len), 
                                       dtype=np.long)
        
        for i, (xx, yy, src, trg) in enumerate(batch_data):
            x[i, :len(xx)] = xx
            y[i, :len(yy) - 1] = yy[:-1]
            y_target[i, :len(yy) - 1] = yy[1:]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)
        if self.use_cuda == True:
            x = x.cuda()
            y = y.cuda()
            y_target = y_target.cuda()
        
        return [x, y], y_target
        
    
    def __call__(self, data_dir, batch_size, start_steps=0):
        """
        """
        data = []
        files = os.listdir(data_dir)
        files.sort()
        
        steps = 0
        for fi in files:
            fi = os.path.join(data_dir, fi)
            for line in open(fi, "rb"):
                steps += 1
                if steps <= start_steps * batch_size:
                    continue
                arr = line.decode("utf-8").strip().split("\t")
                
                src,trg = arr[0],arr[1]
                
                src = self._src_tokenizer.tokenize(src)[:self._src_max_len]
                trg = self._trg_tokenizer.tokenize(trg)[:self._trg_max_len - 1]
                x = [self._src_word2id.get(ww, self.UNK) for ww in src]
                y = [self._trg_word2id.get(ww, self.UNK) for ww in trg]
                y = [self.BOS] + y + [self.EOS]
                data.append([x, y, src, trg])
                if len(data) % (20 * batch_size) == 0:
                    batch_data = data[:batch_size]
                    data = data[batch_size:]
                    yield self._vectorize(batch_data)
        
        while len(data) > 0:
            batch_data = data[:batch_size]
            yield self._vectorize(batch_data)              
            data = data[batch_size:]


def generator_fn(params):
    """
    """
    return BatchGenerator(params["src_max_len"], 
                          params["trg_max_len"], 
                          params["src_vocab"], 
                          params["trg_vocab"],
                          params.get("src_tokenizer", "default"),
                          params.get("trg_tokenizer", "default"),
                          constants["symbols"],
                          params["use_cuda"])


def loss_fn(outputs, target, params):
    """
    """
    loss = seq_cross_entropy(outputs[0], target, params.get("eps", 0), 
                             constants["symbols"]["PAD"])
    return loss