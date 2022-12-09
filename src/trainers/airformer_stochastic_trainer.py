import time
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from src.base.trainer import BaseTrainer
from src.utils import graph_algo


class Trainer(BaseTrainer):
    def __init__(self, **args):
        super(Trainer, self).__init__(**args)
        self._optimizer = Adam(self.model.parameters(), self._base_lr)
        self._supports = self._calculate_supports(
            args['adj_mat'], args['filter_type'])
        self._lr_scheduler = MultiStepLR(self._optimizer,
                                         self._steps,
                                         gamma=self._lr_decay_ratio)
        self.rec_mae = nn.L1Loss()
        self.alpha = 1

    def _calculate_supports(self, adj_mat, filter_type):
        # For GNNs, not for AirFormer
        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)

        if filter_type == "scalap":
            supports = [graph_algo.calculate_scaled_laplacian(
                new_adj).todense()]
        elif filter_type == "normlap":
            supports = [graph_algo.calculate_normalized_laplacian(
                new_adj).astype(np.float32).todense()]
        elif filter_type == "symnadj":
            supports = [graph_algo.sym_adj(new_adj)]
        elif filter_type == "transition":
            supports = [graph_algo.asym_adj(new_adj)]
        elif filter_type == "doubletransition":
            supports = [graph_algo.asym_adj(new_adj),
                        graph_algo.asym_adj(np.transpose(new_adj))]
        elif filter_type == "identity":
            supports = [np.diag(np.ones(new_adj.shape[0])).astype(np.float32)]
        else:
            error = 0
            assert error, "adj type not defined"
        supports = [torch.tensor(i).cuda() for i in supports]
        return supports

    def train_batch(self, X, label, iter):
        '''
        the training process of a batch
        '''
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports
        self.optimizer.zero_grad()
        pred, X_rec, kl_loss = self.model(X, supports)
        pred, label = self._inverse_transform([pred, label])

        pred_loss = self.loss_fn(pred, label, 0.0)
        # negative elbo
        rec_loss = self.rec_mae(X_rec[..., :6], X[..., :6]) # only reconstructing air quality-related attributes
        loss = pred_loss + self.alpha * (rec_loss + kl_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item(), pred_loss.item(), rec_loss.item(), kl_loss.item()

    def train(self):
        '''
        rewrite the train process due to the stochastic stage
        '''
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()

            train_losses = []
            pred_losses = []
            rec_losses = []
            kl_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, (X, label) in enumerate(self.data['train_loader']):
                X, label = self._check_device([X, label])
                loss, pred_loss, rec_loss, kl_loss = self.train_batch(
                    X, label, iter)
                train_losses.append(loss)
                pred_losses.append(pred_loss)
                rec_losses.append(rec_loss)
                kl_losses.append(kl_loss)
                iter += 1
                if iter != None:
                    if iter % self._save_iter == 0:
                        val_loss = self.evaluate()
                        message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, \
                            pred_mae: {:.4f}, rec_loss: {:.4f}, \
                            kl_loss: {:.4f}, val_mae: {:.4f} '.format(epoch,
                                                                      self._max_epochs,
                                                                      iter,
                                                                      np.mean(
                                                                          train_losses),
                                                                      np.mean(
                                                                          pred_losses),
                                                                      np.mean(
                                                                          rec_losses),
                                                                      np.mean(
                                                                          kl_losses),
                                                                      val_loss)
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp)
                            self._logger.info(
                                'Val loss decrease from {:.4f} to {:.4f}, '
                                'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                            val_losses.append(val_loss)
                            saved_epoch = epoch

            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch,
                                 self._max_epochs,
                                 iter,
                                 np.mean(train_losses),
                                 val_loss,
                                 new_lr,
                                 (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

    def test_batch(self, X, label):
        pred, _, _ = self.model(X, self.supports)
        pred, label = self._inverse_transform([pred, label])
        return pred, label
