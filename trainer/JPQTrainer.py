# Copyright 2022 Jinpeng Wang
# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
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
"""Training process for "Jointly optimizing query encoder and product quantization to improve retrieval performance".
Code based on the implementation of "Collaborative Experts" and "Multi-Modal Transformer":
https://github.com/albanie/collaborative-experts
https://github.com/gabeur/mmt
"""

import logging
import pathlib
import time

from base import BaseTrainer
from model.module.util import cross_view_cls_inner_product
import numpy as np
import pytorch_warmup as warmup
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from utils.util import move_dict_to_device, compress_predictions

logger = logging.getLogger(__name__)


class JPQTrainer(BaseTrainer):

  def __init__(self,
               model,
               loss,
               metrics,
               optimizer,
               config,
               data_loaders,
               lr_scheduler,
               visualizer,
               skip_first_n_saves,
               codebook_fixed_epoches,
               update_codebooks_by_clustering,
               include_optim_in_ckpts,
               expert_dims,
               num_keep_ckpts=1,
               tokenizer=None,
               warmup_iterations=-1):
    super().__init__(model, loss, metrics, optimizer, lr_scheduler, config)
    self.config = config
    self.data_loaders = data_loaders
    self.num_keep_ckpts = num_keep_ckpts
    self.train_loaders = [
        lo["loader"] for lo in self.data_loaders["train_sets"]
    ]
    self.train_datasets = [
        lo["dataset"] for lo in self.data_loaders["train_sets"]
    ]
    self.continuous_eval_loaders = [
        lo["loader"] for lo in self.data_loaders["continuous_eval_sets"]
    ]
    self.continuous_eval_datasets = [
        lo["dataset"] for lo in self.data_loaders["continuous_eval_sets"]
    ]
    self.log_step = int(np.sqrt(self.train_loaders[0].batch_size))
    self.visualizer = visualizer
    self.skip_first_n_saves = skip_first_n_saves
    self.codebook_fixed_epoches = codebook_fixed_epoches
    self.update_codebooks_by_clustering = update_codebooks_by_clustering
    self.include_optim_in_ckpts = include_optim_in_ckpts

    self.batch_size = self.train_loaders[0].batch_size
    self.n_pairs = self.train_datasets[0].n_pairs

    self.modalities = list(expert_dims.keys())

    cfg_trainer = config["trainer"]
    self.max_samples_per_epoch = cfg_trainer["max_samples_per_epoch"]
    self.max_batches_per_epoch = int(self.max_samples_per_epoch / self.n_pairs
                                     / self.batch_size)
    self.batches_per_epoch = min(len(self.train_loaders[0]),
                                 self.max_batches_per_epoch)
    self.samples_per_epoch = self.batches_per_epoch * self.batch_size * self.n_pairs

    self.debug_dataloader = False
    self.tokenizer = tokenizer

    if warmup_iterations > 0:
      self.warmup_scheduler = warmup.LinearWarmup(
          optimizer, warmup_period=warmup_iterations)
    else:
      self.warmup_scheduler = None

  def _train_epoch(self, epoch, scaler):
    # There is no training at epoch 0, only evaluation
    if epoch == 0:
      logger.debug("Performing initial evaluation...")
      total_loss = 0
      log = {"loss": total_loss}
      learning_rate = self.lr_scheduler.get_last_lr()[0]
      log["learning_rate"] = learning_rate
      log["n_samples"] = self.n_samples
      log["n_steps"] = self.n_steps
      return log

    self.model.train()
    total_loss = 0
    total_quant_reg = 0
    total_quant_err = 0
    out = "embds" if isinstance(self.model, torch.nn.DataParallel) else "conf"
    if self.update_codebooks_by_clustering:
      out = "embds"
      text_cls_embd_cache = []
      vid_cls_embd_cache = []

    # Choose which of the trainsets to use (pretraining or finetuning)
    i = 0
    while self.data_loaders["train_sets"][i].until_epoch < epoch:
      i += 1

    self.batch_size = self.data_loaders["train_sets"][i].batch_size
    self.n_pairs = self.data_loaders["train_sets"][i].n_pairs
    self.source = self.data_loaders["train_sets"][i]["dataset"].dataset_name

    msg = (f"Out: {out}, source: {self.source}, batch_size: {self.batch_size}, "
           f"n_pairs: {self.n_pairs}")
    logger.debug(msg)

    data_start = time.time()
    for batch_idx, minibatch in enumerate(self.train_loaders[i]):
      # We limit the number of samples per epoch
      if (batch_idx + 1) * self.batch_size * self.n_pairs > self.max_samples_per_epoch:
        break

      self.timer.update("train_batch.data_loading", time.time() - data_start)
      dataset_name = self.train_loaders[0].dataset.dataset_name

      transfer_start = time.time()
      # Print the minibatch data to debug the dataloader
      if self.debug_dataloader and batch_idx < 1:
        self.display_minibatch(minibatch, dataset_name)
        debug = True
      else:
        debug = False

      minibatch = move_dict_to_device(minibatch, self.device)

      self.n_samples += self.batch_size * self.n_pairs
      self.n_steps += 1

      if self.warmup_scheduler:
        self.warmup_scheduler.dampen()

      self.model.codebook_normalization()
      self.optimizer.zero_grad()
      self.timer.update("train_batch.transfer", time.time() - transfer_start)
      forward_start = time.time()
      
      with autocast():
        output = self.model(**minibatch, out=out, device=self.device, debug=debug)

        self.timer.update("train_batch.forward", time.time() - forward_start)
        loss_start = time.time()
        if out == "conf":
          loss = self.loss(quant_cls_t2v_sims=output['quant_vid_t2v_cls_conf_matrix'],  # -> B x B
                          quant_cls_v2t_sims=output['quant_text_t2v_cls_conf_matrix'].transpose(-1,-2),
                          epoch=epoch)
          quant_err = output['quant_err']
        else:
          if self.update_codebooks_by_clustering:
            text_cls_embd_cache.append(output['text_cls_embd'].detach())
            vid_cls_embd_cache.append(output['vid_cls_embd'].detach())

          # Reassemble the tensors from multiple GPUs to one dict
          # -> B x B
          quant_vid_t2v_cls_conf_matrix = cross_view_cls_inner_product(
            text_cls_embd=output['text_cls_embd'],
            vid_cls_embd=output['quant_vid_cls_embd'],
            merge_caption_similarities=self.model.merge_caption_similarities
          )
          # -> B x B
          quant_text_t2v_cls_conf_matrix = cross_view_cls_inner_product(
            text_cls_embd=output['quant_text_cls_embd'],
            vid_cls_embd=output['vid_cls_embd'],
            merge_caption_similarities=self.model.merge_caption_similarities
          )
          
          with torch.no_grad():
            # B x Caps x D -> 1
            quant_text_cls_err = F.mse_loss(output["text_cls_embd"], 
                                            output["quant_text_cls_embd"], 
                                            reduction='none').sum(-1).mean(-1).mean(0)
            # B x D -> 1
            quant_vid_cls_err = F.mse_loss(output["vid_cls_embd"], 
                                          output["quant_vid_cls_embd"], 
                                          reduction='none').sum(-1).mean(0)
            quant_err = (quant_text_cls_err + quant_vid_cls_err) / 2  # -> 1

          loss = self.loss(quant_cls_t2v_sims=quant_vid_t2v_cls_conf_matrix,  # -> B x B
                          quant_cls_v2t_sims=quant_text_t2v_cls_conf_matrix.transpose(-1,-2),
                          epoch=epoch)
        loss += ((1 - self.loss.smoothing_schedule(epoch)) * self.loss.pq_reg_loss_weight * output['quant_reg'])

        self.timer.update("train_batch.loss", time.time() - loss_start)
      
      backward_start = time.time()
      # replace 'loss.backward()'
      scaler.scale(loss).backward()
      # loss.backward()

      # cancel some gradients
      if epoch < self.codebook_fixed_epoches:
        success_flag = False
        for name, p in self.model.named_parameters():
          if "pq_layer" in name:
            success_flag = True
            p.grad = None
        assert success_flag, "No module about 'pq_layer'."

      # replace 'self.optimizer.step()'
      scaler.step(self.optimizer)
      scaler.update()
      # self.optimizer.step()

      loss_value = loss.item()
      total_loss += loss_value
      quant_reg_value = output['quant_reg'].item()
      total_quant_reg += quant_reg_value
      quant_err_value = quant_err.item()
      total_quant_err += quant_err_value

      self.timer.update("train_batch.backward", time.time() - backward_start)
      self.timer.update("train_batch.total", time.time() - data_start)

      logging_start = time.time()
      display_timers = False
      if batch_idx % self.log_step == 0:
        prog = self._progress(batch_idx)
        data_loading_time = self.timer.dic["train_batch.data_loading"]["val"]
        forward_time = self.timer.dic["train_batch.forward"]["val"]
        loss_time = self.timer.dic["train_batch.loss"]["val"]
        backward_time = self.timer.dic["train_batch.backward"]["val"]
        batch_time = self.timer.dic["train_batch.total"]["val"]
        msg = (f"Train Epoch: {epoch} {prog} "
               f"Loss: {loss_value:.5f} (QuantReg: {quant_reg_value:.5f}) "
               f"QuantErr: {quant_err_value:.5f} "
               f"batch_time={batch_time:.5f} ")
        if display_timers:
          msg = msg + (f"data_loading={data_loading_time:.5f}"
                       f"({data_loading_time * 100 / batch_time:.0f}%) "
                       f"forward={forward_time:.5f}"
                       f"({forward_time * 100 / batch_time:.0f}%) "
                       f"loss={loss_time:.5f}"
                       f"({loss_time * 100 / batch_time:.0f}%) "
                       f"backward={backward_time:.5f}"
                       f"({backward_time * 100 / batch_time:.0f}%) ")
        logger.info(msg)

      self.timer.update("train.logging", time.time() - logging_start)
      data_start = time.time()

    # update codebooks
    if self.update_codebooks_by_clustering:
      clustering_start = time.time()
      self.model.update_codebooks(text_cls_embd=torch.cat(text_cls_embd_cache),
                                  vid_cls_embd=torch.cat(vid_cls_embd_cache))
      clustering_end = time.time()
      clustering_time = clustering_end - clustering_start
      msg = (f"Train Epoch: {epoch} codebook_update_time={clustering_time:.5f}")
      logger.info(msg)

    scheduler_start = time.time()
    log = {"loss": total_loss / self.batches_per_epoch, 
           "quant_reg": total_quant_reg / self.batches_per_epoch, 
           "quant_err": total_quant_err / self.batches_per_epoch}

    learning_rate = self.lr_scheduler.get_last_lr()[0]
    log["learning_rate"] = learning_rate
    log["n_samples"] = self.n_samples
    log["n_steps"] = self.n_steps

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

    self.timer.update("train_batch.scheduler", time.time() - scheduler_start)
    return log

  def display_minibatch(self, minibatch, dataset_name):
    b = len(minibatch["token_ids"])
    for i in range(b):
      logger.debug()
      msg = (f"Sample {i}: dataset: {dataset_name}, "
             f"source: {minibatch['sources'][i]}")
      logger.debug(msg)


  def log_metrics(self, metric_store, epoch, metric_name, dataset_name):
    for key, value in metric_store.items():
      if key != "cols":
        self.writer.add_scalar(f"{dataset_name}/{metric_name}/{key}", value,
                               epoch)

  def _get_embeddings(self, val_loader):
    out = "embds"
    self.model.codebook_normalization()
    with torch.no_grad():
      text_cls_embd_list = []
      vid_cls_embd_list = []
      quant_text_cls_embd_list = []
      quant_vid_cls_embd_list = []
      quant_reg_list = []
      query_masks_list = []
      raw_captions_list = []
      token_ids_list = []
      paths_list = []
      logger.debug("Running batches through model ...")
      data_start = time.time()
      for batch_idx, minibatch in enumerate(val_loader):
        self.timer.update("valid_batch.data_loading", time.time() - data_start)

        transfer_start = time.time()
        if "raw_captions" in minibatch.keys():
          raw_captions_list.extend(minibatch["raw_captions"])
          paths_list.extend(minibatch["paths"])
        else:
          b, captions_per_video, _, _ = minibatch["token_ids"].size()
          raw_caps = [[np.array(["unprovided", "words"])] * captions_per_video
                     ] * b
          raw_captions_list.extend(raw_caps)
          paths = [pathlib.Path("unprovided/path")] * b
          paths_list.extend(paths)

        if "token_ids" in minibatch.keys():
          token_ids_list.extend(minibatch["token_ids"])

        query_masks_list.append(torch.from_numpy(minibatch["query_masks"]))

        # Print the minibatch data to debug the dataloader
        if self.debug_dataloader and batch_idx == 0:
          dataset_name = val_loader.dataset.dataset_name
          self.display_minibatch(minibatch, dataset_name)
          debug = True
        else:
          debug = False

        minibatch = move_dict_to_device(minibatch, self.device)

        self.timer.update("valid_batch.transfer", time.time() - transfer_start)

        forward_start = time.time()

        output = self.model(**minibatch,
                            out=out,
                            device=self.device,
                            debug=debug)

        text_cls_embd_list.append(output['text_cls_embd'])
        vid_cls_embd_list.append(output['vid_cls_embd'])
        quant_text_cls_embd_list.append(output['quant_text_cls_embd'])
        quant_vid_cls_embd_list.append(output['quant_vid_cls_embd'])
        quant_reg_list.append(output['quant_reg'])
        self.timer.update("valid_batch.forward", time.time() - forward_start)
        self.timer.update("valid_batch.total", time.time() - data_start)

        data_start = time.time()
      # end for

      query_masks = torch.cat(query_masks_list)
      text_cls_embd = torch.cat(text_cls_embd_list)
      vid_cls_embd = torch.cat(vid_cls_embd_list)
      quant_text_cls_embd = torch.cat(quant_text_cls_embd_list)
      quant_vid_cls_embd = torch.cat(quant_vid_cls_embd_list)
      quant_reg = torch.as_tensor(quant_reg_list)
      token_ids = np.concatenate(token_ids_list)

      res = {
        'text_cls_embd': text_cls_embd,
        'vid_cls_embd': vid_cls_embd,
        'quant_text_cls_embd': quant_text_cls_embd,
        'quant_vid_cls_embd': quant_vid_cls_embd,
        'quant_reg': quant_reg,
        "raw_captions": raw_captions_list,
        "token_ids": token_ids,
        "query_masks": query_masks,
        "paths": paths_list,
      }

      move_dict_to_device(res, device="cpu", only_tensors=False)

      return res

  def _valid_epoch(self, epoch=None, sets="continuous_eval"):
    embds_start = time.time()
    loaders = [lo["loader"] for lo in self.data_loaders[f"{sets}_sets"]]
    datasets = [lo["dataset"] for lo in self.data_loaders[f"{sets}_sets"]]

    self.model.eval()

    result_dict = {}
    result_dict["metrics"] = {}

    loaders_embds = {}

    with torch.no_grad():
      for loader_idx, val_loader in enumerate(loaders):
        dataset = datasets[loader_idx]
        split_name = dataset.split_name
        dataset_name = dataset.dataset_name
        logger.debug("Obtaining embeddings for dataset %s ...", dataset_name)
        loaders_embds[dataset_name] = self._get_embeddings(val_loader)

      self.timer.update("valid.embds", time.time() - embds_start)
      for dataset_name, embds in loaders_embds.items():
        conf_mat_start = time.time()
        logger.debug("Computing confusion matrix ...")
        
        quant_vid_t2v_sims = cross_view_cls_inner_product(
          text_cls_embd=embds['text_cls_embd'],
          vid_cls_embd=embds['quant_vid_cls_embd'],
          merge_caption_similarities=self.model.merge_caption_similarities
        )
        
        quant_text_t2v_sims = cross_view_cls_inner_product(
          text_cls_embd=embds['quant_text_cls_embd'],
          vid_cls_embd=embds['vid_cls_embd'],
          merge_caption_similarities=self.model.merge_caption_similarities
        )
        
        quant_vid_t2v_sims = quant_vid_t2v_sims.data.cpu().float().numpy()
        quant_text_t2v_sims = quant_text_t2v_sims.data.cpu().float().numpy()
        query_masks = embds["query_masks"].numpy()

        dataset_basename = dataset_name.split("_")[0]
        cut_name = dataset_name.split("_")[1]
        split_name = dataset_name.split("_")[2]

        if sets == "final_eval":
          # Challenge mode, we record the rankings
          if cut_name == "c" and split_name in ["test1", "test2"]:
            # example: MSRVTT-public_server_val-predictions.csv
            if split_name in ["test1"]:
              split_name = "public_server_val"
            elif split_name in ["test2"]:
              split_name = "public_server_test"
            prediction_path = pathlib.Path(self.exp_dir) / f"{dataset_basename}-{split_name}-qv-predictions.csv"
            compressed_preds = compress_predictions(
                query_masks=query_masks,
                sims=quant_vid_t2v_sims,
            )
            np.savetxt(prediction_path,
                       compressed_preds,
                       delimiter=",",
                       fmt="%d")
            logger.debug("Saved similarity matrix (quantize videos) predictions to %s", prediction_path)
            prediction_path = pathlib.Path(self.exp_dir) / f"{dataset_basename}-{split_name}-qt-predictions.csv"
            compressed_preds = compress_predictions(
                query_masks=query_masks,
                sims=quant_text_t2v_sims,
            )
            np.savetxt(prediction_path,
                       compressed_preds,
                       delimiter=",",
                       fmt="%d")
            logger.debug("Saved v2t similarity matrix (quantize texts) predictions to %s", prediction_path)
          sims_data = {"quant_vid_t2v_sims": quant_vid_t2v_sims, "query_masks": query_masks}
          sims_path = pathlib.Path(self.exp_dir) / f"{dataset_basename}-{split_name}-qv-sims.npy"
          np.save(sims_path, sims_data)
          logger.info("Saved similarity matrix (quantize videos) to %s", str(sims_path))
          sims_data = {"quant_text_t2v_sims": quant_text_t2v_sims, "query_masks": query_masks}
          sims_path = pathlib.Path(self.exp_dir) / f"{dataset_basename}-{split_name}-qt-sims.npy"
          np.save(sims_path, sims_data)
          logger.info("Saved v2t similarity matrix (quantize texts) to %s", str(sims_path))

        nested_metrics = {}
        self.timer.update("valid.conf_mat", time.time() - conf_mat_start)
        metrics_start = time.time()

        logger.debug("Computing metrics ...")
        for metric in self.metrics:
          metric_name = metric.__name__
          logger.debug("Computing %s metric ...", metric_name)
          if metric_name[:3] == 't2v':
            nested_metrics[metric_name] = metric(quant_vid_t2v_sims, query_masks=query_masks)
          elif metric_name[:3] == 'v2t': # it can automatically transpose the t2v matrix
            nested_metrics[metric_name] = metric(quant_text_t2v_sims, query_masks=query_masks)
          else:
            raise RuntimeError(f"Invalid metric name: {metric_name}!")

          # Log the metrics in Tensorboard
          logger.debug("Logging %s on Tensorboard ...", metric_name)
          self.log_metrics(metric_store=nested_metrics[metric_name],
                           epoch=epoch,
                           metric_name=metric_name,
                           dataset_name=dataset_name)

        result_dict["metrics"][dataset_name] = nested_metrics
        self.timer.update("valid.metrics", time.time() - metrics_start)

    return result_dict

  def _progress(self, batch_idx):
    base = "[{}/{} {}/{} ({:.0f}%)]"
    current = batch_idx + 1
    total = self.batches_per_epoch
    current_samples = (batch_idx + 1) * self.batch_size * self.n_pairs
    total_samples = self.samples_per_epoch
    return base.format(current, total, current_samples, total_samples,
                       100.0 * current / total)
