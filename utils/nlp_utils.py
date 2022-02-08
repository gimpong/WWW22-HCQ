# Copyright 2022 Jinpeng Wang
# Copyright 2020 Valentin Gabeur
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenizer creation.
Code based on the implementation of "Multi-Modal Transformer":
https://github.com/gabeur/mmt
"""

from model.module.txt_embeddings import WeTokenizer
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer, XLNetTokenizer


def create_tokenizer(tokenizer_type, txt_bert_type):
  """Creates a tokenizer given a tokenizer type."""
  if tokenizer_type.endswith('frz'):
    freeze = True
  elif tokenizer_type.endswith('ftn'):
    freeze = False
  if tokenizer_type.startswith('bert'):
    assert txt_bert_type in ['bert-base','bert-large',
                             'distilbert-base',
                             'roberta-base','roberta-large',
                             'xlnet-base','xlnet-large']
    if txt_bert_type in ['bert-base', 'bert-large']:
      model_name_or_path = f'{txt_bert_type}-cased'
      do_lower_case = True
      cache_dir = 'data/cache_dir'
      tokenizer_class = BertTokenizer
      tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                  do_lower_case=do_lower_case,
                                                  cache_dir=cache_dir)
    elif txt_bert_type == 'distilbert-base':
      model_name_or_path = f'{txt_bert_type}-cased'
      do_lower_case = True
      cache_dir = 'data/cache_dir'
      tokenizer_class = DistilBertTokenizer
      tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                  do_lower_case=do_lower_case,
                                                  cache_dir=cache_dir)
    elif txt_bert_type in ['roberta-base', 'roberta-large']:
      model_name_or_path = f'{txt_bert_type}'
      do_lower_case = True
      cache_dir = 'data/cache_dir'
      tokenizer_class = RobertaTokenizer
      tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                  do_lower_case=do_lower_case,
                                                  cache_dir=cache_dir)
    elif txt_bert_type in ['xlnet-base', 'xlnet-large']:
      model_name_or_path = f'{txt_bert_type}-cased'
      do_lower_case = True
      cache_dir = 'data/cache_dir'
      tokenizer_class = XLNetTokenizer
      tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                  do_lower_case=do_lower_case,
                                                  cache_dir=cache_dir)
    
  elif tokenizer_type.startswith('wo2v'):
    we_filepath = 'data/word_embeddings/word2vec/GoogleNews-vectors-negative300.bin'
    tokenizer = WeTokenizer(we_filepath, freeze=freeze)
  elif tokenizer_type.startswith('grvl'):
    we_filepath = 'data/word_embeddings/GrOVLE/mt_grovle.txt'
    tokenizer = WeTokenizer(we_filepath, freeze=freeze)
  else:
    tokenizer = None

  return tokenizer
