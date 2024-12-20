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
  
def train_model(config, model):
  device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
  for epoch in range(1, config['num_epochs']):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch: {epoch:02d}')
    for batch in batch_iterator:
      encoder_input = batch['encoder_input'].to(device)
      decoder_input = batch['decoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)
      decoder_mask = batch['decoder_mask'].to(device)
      encoder_output = model.encode(encoder_input, encoder_mask)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
      proj_output = model.project(decoder_output)
      label = batch['label'].to(device)
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({'Loss': f'{loss.item():6.3f}'})
      loss.backward()
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg))
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, config['model_name'])
    
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=2):
  model.eval()
  count = 0
  src_texts, expected, predicted = [], [], []
  console_width = 80
  with torch.no_grad():
    for batch in validation_ds:
      count += 1
      encoder_input = batch['encoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)
      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
      src_text = batch['src_text'][0]
      tgt_text = batch['tgt_text'][0]
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
      src_texts.append(src_text)
      expected.append(tgt_text)
      predicted.append(model_out_text)
      print_msg('-'*console_width)
      print_msg(f"{f'SOURCE: ':>12}{src_text}")
      print_msg(f"{f'TARGET: ':>12}{tgt_text}")
      print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
      if count == num_examples:
        print_msg('-'*console_width)
        break
      
def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
  sos_idx = tokenizer_tgt.token_to_id('[SOS]')
  eos_idx = tokenizer_tgt.token_to_id('[EOS]')
  encoder_output = model.encode(src, src_mask)
  decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)
  next_word = ''
  while decoder_input.size(1) != max_len and next_word != eos_idx:
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(src).to(device)
    out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
    prob = model.project(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device)], dim=1)
  return decoder_input.squeeze(0)