from evaluator import Evaluator
from typing import Optional
from trainer import Trainer
# from setup import Config
from logger import logger
from dataclasses import asdict
import os
import random
import numpy as np
import torch
from setup import args, DEVICE

# Set path to current directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def make_evaluator(args=args, device=DEVICE, trained_model: Optional = None):
     """Creates an Evaluator object which is ready to .eval on, or .eval_on_setups in case of the automated experience. """
     return Evaluator(
          args=args,
          device=device,
          model=trained_model
     )

if __name__ == "__main__":

     set_seed(seed=args.seed)

     if args.run_mode == 'train':
          logger.info("Running training only")
          trainer = Trainer(args, DEVICE)
          trainer.train()
     elif args.run_mode == 'eval':
          logger.info("Running evaluation only")
          evaluator = make_evaluator()
          evaluator.eval_on_setups('run_mode')
     elif args.run_mode == 'perturb':
          logger.info("Running perturbation only")
          trainer = Trainer(args, DEVICE)
          trainer.perturb()
     elif args.run_mode == 'interpolate':
          logger.info("Running interpolation only")
          trainer = Trainer(args, DEVICE)
          trainer.interpolate()
     else:
          logger.info("Running training and evaluation of this model")
          trainer = Trainer(args, DEVICE)
          trainer.train()
          trainer.best_and_worst()

          evaluator = make_evaluator(trained_model=trainer.model)
          evaluator.eval_on_setups('run_mode')
