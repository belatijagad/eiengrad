"""
  References:
  - https://github.com/hkproj/pytorch-transformer/blob/main/train.py
"""
from dataset import BilingualDataset, causal_mask

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
  for item in ds:
    yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else: 
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer

def get_dataset(config):
  raw_dataset = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='train')
  tokenizer_src = get_or_build_tokenizer(config, raw_dataset, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, raw_dataset, config['lang_tgt'])
  train_ds_size = int(0.9*len(raw_dataset))
  val_ds_size = len(raw_dataset) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(raw_dataset, [train_ds_size, val_ds_size])
  train_dataset = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  val_dataset   = BilingualDataset(val_ds_raw,   tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
  