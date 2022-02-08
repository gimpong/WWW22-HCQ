# Copyright 2022 Jinpeng Wang
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
"""Getting dense text and video embeddings from HCT model.
Code based on the implementation of "Collaborative Experts" and "Multi-Modal Transformer":
https://github.com/albanie/collaborative-experts
https://github.com/gabeur/mmt
"""

import argparse
import os
import random
import time

import data_loader.data_loaders as module_data
import model.arch as module_arch
import numpy as np
import h5py as h5
from parse_config import ConfigParser
import torch
from utils.nlp_utils import create_tokenizer
from utils.util import move_dict_to_device, compute_dims


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PyTorch Template")
  parser.add_argument("--config",
                      default=None,
                      type=str,
                      help="config file path (default: None)")
  parser.add_argument(
      "--resume",
      default=None,
      type=str,
      help="path to the experiment dir to resume (default: None)")
  parser.add_argument("--load_checkpoint",
                      default=None,
                      type=str,
                      help="path to the checkpoint to load (default: None)")
  parser.add_argument("--device", type=str, help="indices of GPUs to enable")
  parser.add_argument("--only_eval", action="store_true")
  parser.add_argument("-v",
                      "--verbose",
                      help="increase output verbosity",
                      action="store_true")
  args = parser.parse_args()
  assert args.load_checkpoint is not None, "please input a checkpoint"
  config = ConfigParser(args)
  
  # Get the list of experts and their dimensions
  expert_dims = compute_dims(config)
  raw_input_dims = {}
  for expert, expert_dic in expert_dims.items():
    raw_input_dims[expert] = expert_dic["dim"]
  device = torch.device('cuda:0' if config['n_gpu'] > 0 else 'cpu')

  # Set the random initial seeds
  tic = time.time()
  seed = config["seed"]
  cross_seed = config.get("cross_seed", seed)
  print("Setting experiment random seed to %d", seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

  # Tokenizer to parse sentences into tokens
  tokenizer = create_tokenizer(config["arch"]["args"]["txt_inp"])

  # Create the datasets
  print("Preparing training and testing sets ...")
  train_loader = getattr(module_data, config["train_sets"][0]["type"])(
    **config["train_sets"][0]["args"],
    raw_input_dims=raw_input_dims,
    training=False,
    tokenizer=tokenizer,
    loaded_data={},
    cross_seed=cross_seed,
  )
  test_loader = getattr(module_data, config["final_eval_sets"][0]["type"])(
    **config["final_eval_sets"][0]["args"],
    raw_input_dims=raw_input_dims,
    training=False,
    tokenizer=tokenizer,
    loaded_data={},
    cross_seed=cross_seed,
  )

  # Setup the cross-modal architecture
  model = config.init(
      name="arch",
      module=module_arch,
      expert_dims=expert_dims,
      tokenizer=tokenizer,
  )

  # Load checkpoint
  resume_path = str(config.resume)
  print('Loading checkpoint from: %s ...', resume_path)
  checkpoint = torch.load(resume_path, map_location=device)
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)

  # Compute embeddings ...
  print("Compute training embeddings ...")
  text_cls_embd_list = []
  vid_cls_embd_list = []
  text_vlad_embds_list = []
  vid_vlad_embds_list = []
  with torch.no_grad():
    for batch_idx, minibatch in enumerate(train_loader["loader"]):
      minibatch = move_dict_to_device(minibatch, device)
      output = model(**minibatch, out="embds", device=device, debug=False)
      text_cls_embd_list.append(output['text_cls_embd'])
      vid_cls_embd_list.append(output['vid_cls_embd'])
      text_vlad_embds_list.append(output['text_vlad_embds'])
      vid_vlad_embds_list.append(output['vid_vlad_embds'])
  train_text_cls_embd = torch.cat(text_cls_embd_list)
  train_vid_cls_embd = torch.cat(vid_cls_embd_list)
  train_text_vlad_embds = torch.cat(text_vlad_embds_list)
  train_vid_vlad_embds = torch.cat(vid_vlad_embds_list)

  print("Compute testing embeddings ...")
  text_cls_embd_list = []
  vid_cls_embd_list = []
  text_vlad_embds_list = []
  vid_vlad_embds_list = []
  with torch.no_grad():
    for batch_idx, minibatch in enumerate(test_loader["loader"]):
      minibatch = move_dict_to_device(minibatch, device)
      output = model(**minibatch, out="embds", device=device, debug=False)
      text_cls_embd_list.append(output['text_cls_embd'])
      vid_cls_embd_list.append(output['vid_cls_embd'])
      text_vlad_embds_list.append(output['text_vlad_embds'])
      vid_vlad_embds_list.append(output['vid_vlad_embds'])
  test_text_cls_embd = torch.cat(text_cls_embd_list)
  test_vid_cls_embd = torch.cat(vid_cls_embd_list)
  test_text_vlad_embds = torch.cat(text_vlad_embds_list)
  test_vid_vlad_embds = torch.cat(vid_vlad_embds_list)
  
  embedding_dict = {
    'train_text_cls_embd': train_text_cls_embd,
    'train_vid_cls_embd': train_vid_cls_embd,
    'train_text_vlad_embds': train_text_vlad_embds,
    'train_vid_vlad_embds': train_vid_vlad_embds,
    'test_text_cls_embd': test_text_cls_embd,
    'test_vid_cls_embd': test_vid_cls_embd,
    'test_text_vlad_embds': test_text_vlad_embds,
    'test_vid_vlad_embds': test_vid_vlad_embds
  }
  move_dict_to_device(embedding_dict, device="cpu", only_tensors=False)
  
  # Saving embeddings
  print("Saving embeddings ...")
  h5file = h5.File(config.save_dir / 'embeddings.h5', 'w')
  for key, val in embedding_dict.items():
    h5file[key] = val  
  h5file.close()
  os.system(f"chmod 777 {config.save_dir / 'embeddings.h5'}")
  
  duration = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - tic))
  print("Script took %s", duration)
