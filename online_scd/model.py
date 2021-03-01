from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import optim
import numpy as np
import math

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import scipy.signal
from online_scd.utils import tp_fp_fn

def weighted_focal_loss(inputs, targets, alpha=0.25, gamma=2, padding_value=-100):
    alpha = torch.tensor([alpha, 1 - alpha]).to(inputs.device)
    #can set padded values to 0 for torch.gather as they are masked out
    targets[targets == padding_value] = 0
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    at = torch.gather(alpha, 0, targets.type(torch.long).view(-1))
    pt = torch.exp(-bce_loss)
    f_loss = at * (1 - pt) ** gamma * bce_loss
    return f_loss


def cross_entropy_loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction="none")

def collar_ce_loss(log_probs, targets, collar, pos_weight):
    # sometimes lengths are slightly different due to padding in feature extraction
    targets = targets[0:len(log_probs)]
    result = log_probs[:, 0].sum()
    
    one_indexes  = (targets==1).nonzero(as_tuple=False)
    for one_index in one_indexes:
        collar_variant_logs = []
        collar_start_index = max(0, one_index - collar)
        collar_end_index = min(len(targets) - 1, one_index + collar)        
        ref_tmp = torch.eye((collar_end_index - collar_start_index + 1).item(), dtype=int, device=log_probs.device)
        time_index = range(collar_start_index, collar_end_index + 1)
        collar_variant_logs = log_probs[time_index, ref_tmp].sum(1)
        result = result - log_probs[time_index, 0].sum()
        result = result + pos_weight * torch.logsumexp(collar_variant_logs, 0)        
    return -result/len(targets)

def find_fov(model, input_dim):
  for i in range(1, 10000):
    try:
      result = model(torch.zeros(4, input_dim, i))
      if result.shape[0] == 4 and result.shape[2] == 1:
        return i
      else:
        raise Exception("Something went wrong. Couldn't derive field-of-view.")      
    except RuntimeError:
      pass
  raise Exception("Something went wrong. Couldn't derive field-of-view.")

class SCDModel(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
       
        pre_lstm_layers = []
        conv_kernel_sizes = [int(i.strip()) for i in self.hparams.conv_kernels.split(",")]
        conv_kernel_dilations = [int(i.strip()) for i in self.hparams.conv_dilations.split(",")]
        conv_kernel_strides = [int(i.strip()) for i in self.hparams.conv_strides.split(",")]
        assert len(conv_kernel_sizes) == len(conv_kernel_dilations)
        assert len(conv_kernel_sizes) == len(conv_kernel_strides)
        current_input_dim = self.hparams.num_fbanks

        if self.hparams.load_xvector_model is not None:
            from torch_xvectors2 import XVectorModel
            self.xvector_model = XVectorModel.load_from_checkpoint(self.hparams.load_xvector_model)
            for p in self.xvector_model.parameters():
                p.requires_grad = False
            # find output dim
            raise Exception("not implemented")
            #dummy_mel_spec = torch.zeros(1, self.hparams.num_fbanks, 1000)
            #dummy_mel_spec_length = torch.tensor([1000])
            #dummy_vad = self.xvector_model.vad_extractor(dummy_mel_spec, dummy_mel_spec_length)
            #dummy_output = self.xvector_model._forward_until_before_pooling(dummy_mel_spec, dummy_vad, torch.tensor([0]))
            #current_input_dim = dummy_output.shape[1]
            #self.encoder_subsampling_factor = round(1000 / dummy_output.shape[2])

        else:
            for i in range(len(conv_kernel_sizes)):
                conv_layer = nn.Conv1d(current_input_dim, self.hparams.conv_hidden_dim, conv_kernel_sizes[i], 
                                    dilation=conv_kernel_dilations[i], 
                                    padding=0,
                                    stride=conv_kernel_strides[i],
                                    bias=False)
                pre_lstm_layers.append(conv_layer)            
                pre_lstm_layers.append(nn.BatchNorm1d(self.hparams.conv_hidden_dim, affine=False))
                pre_lstm_layers.append(nn.ReLU(inplace=True))
                current_input_dim = self.hparams.conv_hidden_dim

            self.pre_lstm_layers = nn.Sequential(*pre_lstm_layers)
            self.encoder_fov = find_fov(self.pre_lstm_layers, self.hparams.num_fbanks)


        self.lstm = nn.LSTM(input_size=current_input_dim,
                    hidden_size=self.hparams.lstm_hidden_size,
                    batch_first=True,
                    num_layers=self.hparams.lstm_num_layers)

        self.post_lstm_layers = nn.Sequential(nn.Linear(self.hparams.lstm_hidden_size, 2))


        if self.hparams.loss_function == "focal":
            self.loss_fn = weighted_focal_loss
        elif self.hparams.loss_function == "cross-entropy":
            self.loss_fn = cross_entropy_loss
        else:
            raise RuntimeError("Unknown loss function")

    def _encode_features(self, x, x_lens, hidden=None):
        x_orig = x
        batch_size = x.shape[0]
        # x: (B x L x F)
        x = x.unsqueeze(1)
        # x: (B x 1 x L x F)
        x = F.unfold(x, (self.encoder_fov, self.hparams.num_fbanks), stride=self.hparams.detection_period)
        # x: (B, self.encoder_fov * self.hparams.num_fbanks, ...)
        x = x.reshape(batch_size, self.encoder_fov, self.hparams.num_fbanks, -1)
        # x: (B, self.encoder_fov, self.hparams.num_fbanks, ...)

        time_steps = x.shape[-1]
        x = x.permute(0, 3, 2, 1).reshape(batch_size * time_steps, self.hparams.num_fbanks, self.encoder_fov)
        # x: (B * time_steps, self.hparams.num_fbanks, self.encoder_fov)

        #breakpoint()
        x = self.encode_windowed_features(x)

        x = x.reshape(batch_size, time_steps, -1)
        return x

    def encode_windowed_features(self, x):
        # x: (B * time_steps, self.hparams.num_fbanks, self.encoder_fov)
        if self.hparams.load_xvector_model is not None:
            raise Exception("not implemented")
            #vad = self.xvector_model.vad_extractor(x, x_lens)
            #x = self.xvector_model._forward_until_before_pooling(x, vad, torch.tensor([0] * x.shape[0]).to(x.device))
            #x_lens = x_lens // self.encoder_subsampling_factor 
        else:
            x = self.pre_lstm_layers(x)
        return x

    def _decode(self, encoded_features, lengths):
        #targets = F.one_hot(targets, num_classes=2).float()
        #targets = self.decoder_emb(targets)
        #combined_inputs = torch.cat([encoded_features, targets], dim=-1)
        
        combined_inputs = pack_padded_sequence(encoded_features, lengths, batch_first=True, enforce_sorted=False)
        output_enc_packed, hidden = self.lstm(combined_inputs)
        output_enc_padded, output_lengths = pad_packed_sequence(output_enc_packed, batch_first=True)
        output_enc_padded = self.post_lstm_layers(output_enc_padded)
        return output_enc_padded

    def decode_single_timestep(self, encoded_feature, hidden):

        y, hidden = self.lstm(encoded_feature.permute(0, 2, 1), hidden)
        y = self.post_lstm_layers(y)
        return y, hidden


    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        mel_spec = batch["mel_spec"]
        mel_spec_length = batch["mel_spec_length"]
        y = batch["label"]
        batch_size = y.shape[0]
        batch_width = y.shape[1]
        y_folded = F.unfold(
            y.reshape(batch_size, 1, batch_width, 1).float(), 
            (self.encoder_fov, 1), 
            stride=self.hparams.detection_period).long()
        y_subsampled = y_folded[:, self.encoder_fov//2-self.hparams.detection_period//2:self.encoder_fov//2+self.hparams.detection_period//2, :].max(dim=1)[0]

        output_enc_padded = self._encode_features(mel_spec, mel_spec_length)
        logits = self._decode(
            output_enc_padded[:, 1:y_subsampled.shape[1]], 
            y_subsampled[:, :-1],  
            torch.clamp((mel_spec_length - self.encoder_fov) // self.hparams.detection_period, 1, y_subsampled[:, :-1].shape[1]))


        # We ignore loss values from frames that exceed sequence length 
        # and also from left_context leftmost frames because LSTM needs some 'warmup'
        if self.hparams.train_collar > 0:
            # FIXME: implement mask
            losses = []
            log_probs = logits.log_softmax(dim=-1)
            for i in range(len(mel_spec)):
                losses.append(collar_ce_loss(log_probs[i], y_subsampled[i], self.hparams.train_collar // self.hparams.detection_period, self.hparams.pos_weight))
                
            loss_val = torch.stack(losses).mean()

        else:
            mask = torch.arange(y_subsampled.shape[1], device=mel_spec.device)[None, :] < ((mel_spec_length - self.encoder_fov) // self.hparams.detection_period - 1)[:, None]
            loss_val = (self.loss_fn(logits.reshape(-1, 2), y_subsampled[:, 1:].reshape(-1)) * mask[:,1:].reshape(-1)).sum() / mask.sum()

        self.log('train_loss', loss_val, prog_bar=True, logger=True)
        self.log('lr', torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'], device=loss_val.device), prog_bar=True)
        return loss_val


    def validation_step(self, batch, batch_idx):
        mel_spec = batch["mel_spec"]
        mel_spec_length = batch["mel_spec_length"]
        y = batch["label"]
        # compress y in time, so that we take the max value of every subsampled period
        batch_size = y.shape[0]
        batch_width = y.shape[1]
        y_folded = F.unfold(
            y.reshape(batch_size, 1, batch_width, 1).float(), (self.encoder_fov, 1), 
            stride=self.hparams.detection_period).long()
        y_subsampled = y_folded[:, self.encoder_fov//2-self.hparams.detection_period//2:self.encoder_fov//2+self.hparams.detection_period//2, :].max(dim=1)[0]
        output_enc_padded = self._encode_features(mel_spec, mel_spec_length)

        logits = self._decode(output_enc_padded[:, 1:y_subsampled.shape[1]], y_subsampled[:, :-1],  (mel_spec_length - self.encoder_fov) // self.hparams.detection_period)

        self.last_logits = logits
        if self.hparams.train_collar > 0:
            # FIXME: implement mask
            losses = []
            log_probs = logits.log_softmax(dim=-1)
            for i in range(len(mel_spec)):
                losses.append(collar_ce_loss(log_probs[i], y_subsampled[i], self.hparams.train_collar // self.hparams.detection_period, self.hparams.pos_weight))
            loss_val = torch.stack(losses).mean()
        else:
            mask = torch.arange(y_subsampled.shape[1], device=mel_spec.device)[None, :] < ((mel_spec_length - self.encoder_fov) // self.hparams.detection_period - 1)[:, None]
            loss_val = (self.loss_fn(logits.view(-1, 2), y_subsampled[:, 1:].view(-1) * 0) * mask[:,1:].view(-1)).sum() / mask.sum()
        y_hat = logits.softmax(dim=-1)
        # We set changepoint probability during the 1st second to 0
        #y_hat[:, 0:100//self.hparams.detection_period, 1] = 0.0

        probs = y_hat[0, :, 1]
        return {"probs": probs, "y": y_subsampled, "val_loss": loss_val}



    def validation_epoch_end(self, outputs):
        best_f1 = 0
        best_recall = 0
        best_precision = 0
        best_threshold = 0.5
        for threshold in  [(i+1) * self.hparams.threshold_search_interval for i in range(int(1/self.hparams.threshold_search_interval))]:
            tp = 0
            fp = 0
            fn = 0
            for output in outputs:
                probs = output["probs"].cpu().reshape(-1)
                y = output["y"].cpu().reshape(-1)
                preds = probs > threshold
                _tp, _fp, _fn = tp_fp_fn(preds, y, tolerance=50//self.hparams.detection_period)
                tp += _tp
                fp += _fp
                fn += _fn

            if tp == 0 and fp == 0:
                precision = float("nan")
            else:
                precision = tp / (fp + tp)
                
            if tp == 0 and fn == 0:
                recall = float("nan")
            else:
                recall = tp / (fn + tp)
            if (precision + recall == 0.0):
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            if not math.isnan(f1) and best_f1 < f1:
                best_f1 = f1
                best_threshold = threshold
                best_recall = recall
                best_precision = precision

        tensorboard = self.trainer.logger.experiment
        for i, output in enumerate(outputs):
            fig = plt.figure()
            probs = output["probs"].cpu().reshape(-1)
            y = output["y"].cpu().reshape(-1)
            preds = probs > best_threshold

            plt.plot(probs)
            for x in torch.nonzero(y, as_tuple=False):
                plt.axvline(x=x, c='r', linestyle='-')
            for x in torch.nonzero(preds, as_tuple=False):
                plt.axvline(x=x, c='g', alpha=0.5, linestyle=':')
            plt.axhline(y=best_threshold, c='black')
            tensorboard.add_figure(f"val_{i}", fig, global_step=self.trainer.current_epoch)

        val_loss = torch.tensor([o["val_loss"] for o in outputs]).mean()
        self.log('precision', best_precision, prog_bar=True)
        self.log('recall', best_recall, prog_bar=True)
        self.log('f1', best_f1, prog_bar=True)
        self.log('threshold', best_threshold, prog_bar=True)
        self.log('val_loss', val_loss, prog_bar=True)


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)        
        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        parser.add_argument('--batch-size', default=32, type=int)
        parser.add_argument('--detection-period', default=10, type=int)
        parser.add_argument('--label-delay', default=100, type=int)
        parser.add_argument('--num-fbanks', default=40, type=int)
        parser.add_argument('--lstm-hidden-size', default=256, type=int)
        parser.add_argument('--lstm-num-layers', default=2, type=int)
        parser.add_argument('--conv-kernels',   default="5,1,3,1,3,1,3,1,1")
        parser.add_argument('--conv-dilations', default="1,1,1,1,2,1,2,1,1")
        parser.add_argument('--conv-strides',   default="1,2,1,2,1,2,1,1,1")
        parser.add_argument('--conv-hidden-dim', default=512, type=int)
        parser.add_argument('--learning-rate', default=0.003, type=float)
        parser.add_argument('--left-context', default=100, type=int)
        parser.add_argument('--loss-function', default="focal", type=str)
        parser.add_argument('--threshold-search-interval', default=0.05, type=float)
        parser.add_argument('--train-collar', default=0, type=int)
        parser.add_argument('--pos-weight', default=1.0, type=float)

        parser.add_argument('--load-xvector-model', type=str)

        return parser
