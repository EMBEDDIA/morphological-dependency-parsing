# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import supar
import torch
import torch.distributed as dist
from supar.utils import Config, Dataset
from supar.utils.field import Field
from supar.utils.logging import init_logger, logger
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev, test,
              buckets=32,
              batch_size=5000,
              lr=2e-3,
              mu=.9,
              nu=.9,
              epsilon=1e-12,
              clip=5.0,
              decay=.75,
              decay_steps=5000,
              epochs=5000,
              patience=100,
              verbose=True,
              bert_lr=2e-5,
              **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Load the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        total_examples = len(train.sentences) + len(dev.sentences) + len(test.sentences)
        train_tokens = sum(len(train.sentences[i].values[train.sentences[i].maps["words"]]) for i in range(len(train.sentences)))
        dev_tokens = sum(len(dev.sentences[i].values[dev.sentences[i].maps["words"]]) for i in range(len(dev.sentences)))
        test_tokens = sum(len(test.sentences[i].values[test.sentences[i].maps["words"]]) for i in range(len(test.sentences)))
        logger.info(f"\t- Num. test tokens: {test_tokens}")
        logger.info(f"\t- Num. sentences: {total_examples}")
        logger.info(f"\t- Num. tokens: {train_tokens + dev_tokens + test_tokens}")

        logger.info(f"{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)

        param_groups = []
        # BERT fine-tuning requires a much lower learning rate than the other layers
        for attr_name in dir(self.model):
            curr = getattr(self.model, attr_name)
            if isinstance(curr, nn.Module):
                curr_lr = args.bert_lr if attr_name == "bert_embed" else args.lr
                curr_params = list(curr.parameters())

                # Skip modules with no learnable parameters and pretrained non-contextual word embeddings
                if len(curr_params) == 0 or attr_name == "pretrained":
                    continue
                param_groups.append({"params": curr_params, "lr": curr_lr})

        self.optimizer = Adam(param_groups, args.lr, (args.mu, args.nu), args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()
        best_mean_as = -float("inf")  # mean of UAS and LAS

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            loss, dev_metric = self._evaluate(dev.loader)
            curr_mean_as = 0.5 * (dev_metric.uas + dev_metric.las)

            logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            print(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            print((f"{'test:':6} - loss: {loss:.4f} - {test_metric}"))

            t = datetime.now() - start
            # save the model if it is the best so far
            if curr_mean_as > best_mean_as:
                best_e, best_mean_as, best_metric = epoch, curr_mean_as, dev_metric
                if is_master():
                    self.save(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= args.patience:
                break
        loss, metric = self.load(args.path)._evaluate(test.loader)

        test_preds = self._predict(test.loader)
        for name, value in test_preds.items():
            setattr(test, name, value)
        if test_preds is not None:
            logger.info(f"Save predicted results to {'predictions.txt'}")
            self.transform.save('predictions.txt', test.sentences)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} - {best_metric}")
        logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

        print(f"Epoch {best_e} saved")
        print(f"{'dev:':6} - {best_metric}")
        print(f"{'test:':6} - {metric}")
        print(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Load the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Load the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None:
            logger.info(f"Save predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        r"""
        Load data fields and model parameters from a pretrained parser.

        Args:
            path (str):
                - a string with the shortcut name of a pre-trained parser defined in supar.PRETRAINED
                  to load from cache or download, e.g., `crf-dep-en`.
                - a path to a directory containing a pre-trained parser, e.g., `./<path>/model`.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loaded parser.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(path):
            state = torch.load(path)
        else:
            path = supar.PRETRAINED[path] if path in supar.PRETRAINED else path
            state = torch.hub.load_state_dict_from_url(path)
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        return cls(args, model, transform)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': self.args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path)
