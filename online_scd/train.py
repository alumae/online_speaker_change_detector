"""
Runs a model on a single node across N-gpus.
"""
import os
import sys
from argparse import ArgumentParser
import multiprocessing as mp

import logging
import numpy as np
import torch
import torch.utils.data 

from online_scd.model import SCDModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from online_scd.data import SCDDataset

seed_everything(234)

def main(args):
    """
    Main training routine specific for this project
    :param hparams:
    """
    if args.train_datadir is not None and args.dev_datadir is not None:
        train_dataset = SCDDataset(args.train_datadir, extract_chunks=True, **vars(args))

        batch_size = args.batch_size

        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=train_dataset.collater,
                num_workers=8)
        
        dev_dataset = SCDDataset(args.dev_datadir, extract_chunks=False, **vars(args), no_augment=True)

        dev_loader = torch.utils.data.DataLoader(
                dataset=dev_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=dev_dataset.collater,
                num_workers=2)

        if (args.load_checkpoint):
            model = SCDModel.load_from_checkpoint(args.load_checkpoint, **vars(args))
        else:
            model = SCDModel(**vars(args))

        checkpoint_callback = ModelCheckpoint(
                save_top_k=4,
                save_last=True,
                verbose=True,
                monitor='f1',
                mode='max',
                prefix=''
        )        
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             callbacks=[
                                             ])
        trainer.fit(model, train_dataloader=train_loader,
                    val_dataloaders=dev_loader)
    elif args.test_datadir is not None:
        # Not implemented
        pass
    else:
        raise Exception("Either --train-datadir and --dev-datadir or --test-datadir and --load-checkpoint should be specified")

        


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #mp.set_start_method('fork')

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    

    # each LightningModule defines arguments relevant to it
    parser = SCDModel.add_model_specific_args(parent_parser, root_dir)
    parser = SCDDataset.add_data_specific_args(parser, root_dir)
    parser = Trainer.add_argparse_args(parser)

    # data
    parser.add_argument('--train-datadir', required=False, type=str)       
    parser.add_argument('--dev-datadir', required=False, type=str)
    
    parser.add_argument('--load-checkpoint', required=False, type=str)        
    parser.add_argument('--test-datadir', required=False, type=str)        

    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)
