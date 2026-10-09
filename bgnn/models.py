"""Unary BGNN numerical core extracted from architecture.py.

Independent posterior draws follow the manuscript's mean-field sampling.
The original likelihood and regularization calculations are retained.
"""
import sys
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GIN
from torch_geometric.loader import DataLoader

from .features import output_dims
from .runtime import cprint


def select_conv_layer(name):
    if name != "GIN":
        raise ValueError("Only the paper's GIN backbone is supported.")
    return GIN


class Scaler:

    def __init__(self, prop, mode, feature_range=(0, 1), eps=1e-07, device=None):
        if mode not in ['normalization', 'standardization']:
            raise NotImplementedError('Unsupported configuration')
        self.prop = prop
        self.mode = mode
        self.feature_range = feature_range
        self.eps = eps
        self.device = device

    def fit(self, dataset):
        Y = self.extract_y(dataset)
        self.min = Y.min()
        self.max = Y.max()
        self.mean = Y.mean()
        self.std = Y.std()
        return

    def extract_y(self, dataset):
        if output_dims[self.prop] == 1:
            Y = torch.tensor(list(map(lambda data: data.y, dataset)))
            # Keep the sample axis for one-molecule validation/test splits.
            return Y.reshape(-1)
        else:
            Y = torch.vstack(list(map(lambda data: data.y, dataset)))
        return Y.squeeze()

    def transform_dataset(self, dataset):
        Y = self.extract_y(dataset)
        Y_ = self.transform(Y)
        list(map(lambda data, y_: setattr(data, 'y', y_), dataset, Y_))
        return dataset

    def transform(self, Y):
        if self.mode == 'normalization':
            Y_ = (Y - self.min) / (self.max - self.min) * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        elif self.mode == 'standardization':
            Y_ = (Y - self.mean) / (self.std + self.eps)
        else:
            raise NotImplementedError('Unsupported configuration')
        return Y_

    def inverse_transform(self, Y_):
        if self.mode == 'normalization':
            Y = (Y_ - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * (self.max - self.min) + self.min
        elif self.mode == 'standardization':
            Y = Y_ * (self.std + self.eps) + self.mean
        else:
            raise NotImplementedError('Unsupported configuration')
        return Y

class BGNN(nn.Module):

    posterior_sampling = 'independent'

    def __init__(self, prop, num_layers=5, hidden_channels=128, fingerprint_channels=64, bayes=False, scaling=True, W_rho_init=-3, b_rho_init=-3, W_prior_std=0.5, b_prior_std=0.5, std_min=0.001, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device='cuda'):
        super(BGNN, self).__init__()
        self.prop = prop
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.fingerprint_channels = fingerprint_channels
        self.device = device
        self.bayes = bayes
        self.scaling = scaling
        self.W_rho_init = W_rho_init
        self.b_rho_init = b_rho_init
        self.W_prior_std = W_prior_std
        self.b_prior_std = b_prior_std
        self.std_min = std_min
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.output_dim = output_dims[self.prop]
        if self.bayes:
            self.W_mu = nn.Parameter(torch.randn(self.fingerprint_channels, self.output_dim))
            self.W_rho = nn.Parameter(W_rho_init * torch.ones(self.fingerprint_channels, self.output_dim))
            self.b_mu = nn.Parameter(torch.randn(self.output_dim))
            self.b_rho = nn.Parameter(b_rho_init * torch.ones(self.output_dim))
        else:
            self.W = nn.Parameter(torch.randn(self.fingerprint_channels, self.output_dim))
            self.b = nn.Parameter(torch.randn(self.output_dim))

    def migrate_priors(self):
        self.W_prior = Normal(self.W_prior.loc.to(self.device), self.W_prior.scale.to(self.device))
        self.b_prior = Normal(self.b_prior.loc.to(self.device), self.b_prior.scale.to(self.device))
        return

    def load_weights(self, checkpointdir):
        checkpoint = torch.load(checkpointdir, weights_only=True, map_location='cpu')
        if checkpoint.get('schema_version') != 2 or checkpoint.get('posterior_sampling') != 'deterministic':
            raise ValueError('Warm start requires a schema-2 deterministic GNN checkpoint')
        weights = checkpoint['model']
        weights['W_mu'] = weights.pop('W')
        weights['b_mu'] = weights.pop('b')
        self.load_state_dict(weights, strict=False)
        self.eval()
        del weights
        return

    def gradient_clipping(self, bounds=(-10, 10)):
        for name, weights in self.named_parameters():
            if weights.requires_grad:
                weights.register_hook(lambda grad, bounds=bounds: torch.clamp(grad, bounds[0], bounds[1]))

    def setup_optimizer(self):
        if self.clipping:
            self.gradient_clipping(bounds=self.gradient_bounds)
        if self.no_decay:
            cprint('Set weight decay parameter to zero.')
            weight_p, bias_p = ([], [])
            for name, p in self.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            params = [{'params': weight_p, 'weight_decay': 0}, {'params': bias_p, 'weight_decay': 0}]
        else:
            params = self.parameters()
        optimizer = optim.AdamW(params, lr=self.lr)
        if self.lr_scheduling:
            scheduler = ReduceLROnPlateau(optimizer, patience=self.patience, factor=self.factor, min_lr=self.min_lr)
        else:
            scheduler = None
        return (optimizer, scheduler)

    def set_priors(self):
        zeros_W_mu = torch.zeros(self.W_mu.shape, device=self.device, requires_grad=False)
        zeros_b_mu = torch.zeros(self.b_mu.shape, device=self.device, requires_grad=False)
        ones_W_mu = torch.ones(self.W_mu.shape, device=self.device, requires_grad=False)
        ones_b_mu = torch.ones(self.b_mu.shape, device=self.device, requires_grad=False)
        if self.warmstart:
            self.W_prior = Normal(self.detach(self.W_mu, device=self.device), self.W_prior_std * ones_W_mu)
            self.b_prior = Normal(self.detach(self.b_mu, device=self.device), self.b_prior_std * ones_b_mu)
        else:
            self.W_prior = Normal(zeros_W_mu, ones_W_mu * self.scale_prior)
            self.b_prior = Normal(zeros_b_mu, ones_b_mu * self.scale_prior)
        return

    def forward(self, X):
        Z = self._forward(X)
        W, b = self.sampling()
        y_pred = torch.matmul(Z, W) + b
        if self.prop == 'sp':
            y_pred = F.softplus(y_pred)
        return y_pred

    def eval_total_error(self, dataset):
        y_pred = self(dataset)
        if self.bayes:
            y_pred = y_pred.mean(dim=-1).reshape(-1, 1)
        y_test = self.extract_y(dataset)
        total_error = self.eval_error(y_pred, y_test)
        return total_error

    def eval_error(self, y_pred, y, etype=None):
        assert y_pred.shape == y.shape
        _y_pred = self.scaler.inverse_transform(y_pred)
        _y = self.scaler.inverse_transform(y)
        if etype == None:
            etype = self.etype
        if etype == 'mae':
            return torch.sum(torch.abs(_y_pred - _y)) / len(y)
        elif etype == 'rmse':
            return torch.sqrt(torch.sum((_y_pred - _y) ** 2) / len(y))
        else:
            raise NotImplementedError('Unsupported configuration')

    def extract_y(self, dataset):
        y = dataset.y
        n_batch = self.get_batch_size(dataset)
        y = y.reshape(n_batch, -1)
        return y

    def get_batch_size(self, X):
        return int(max(X.batch)) + 1

    def eval_loss(self, dataset):
        y = self.extract_y(dataset)
        X = dataset
        if self.bayes:
            y_preds = self(X)
            y_pred = y_preds.mean(dim=-1).reshape(-1, 1)
            log_prior = self.eval_log_prior() / self.b_total
            log_post = self.eval_log_post() / self.b_total
            log_like = self.eval_log_like(y_preds, y)
            error = self.eval_error(y_pred, y)
            if self.z_regularize_opt is None:
                loss = ((log_post - log_prior) * self.scale_kl - log_like) * self.scale_loss
                return {'loss': loss, 'log_post': log_post, 'log_prior': log_prior, 'log_like': log_like, self.etype + '_train': error}
            else:
                z = self.z
                z_reg = self.eval_z_regularize(z)
                loss = ((log_post - log_prior) * self.scale_kl - log_like) * self.scale_loss + z_reg
                return {'loss': loss, 'log_post': log_post, 'log_prior': log_prior, 'log_like': log_like, 'z_regularize': z_reg, self.etype + '_train': error}
        else:
            y_pred = self(X)
            loss = F.mse_loss(y_pred, y) * self.scale_loss
            error = self.eval_error(y_pred, y)
            return {'loss': loss, self.etype + '_train': error}

    def evaluate_metric_per_epoch(self, recorder):
        losses = {}
        N = torch.tensor(recorder['n'])
        for key, values in recorder.items():
            if key == 'n':
                pass
            else:
                recorder[key] = torch.tensor(values)
        for key, values in recorder.items():
            if key == 'n':
                pass
            elif key == 'loss' and (not self.bayes) or 'mae' in key or 'mse' in key:
                total_squared_error = N * values
                losses[key] = sum(total_squared_error) / sum(N)
            elif key == 'loss' and self.bayes:
                pass
            elif key in ['log_prior', 'log_post', 'log_like']:
                losses[key] = sum(values)
            elif key in ['log_prior', 'log_post', 'log_like', 'z_regularize']:
                losses[key] = sum(values)
            else:
                raise NotImplementedError('Unsupported metric')
        if self.bayes:
            if self.z_regularize_opt is None:
                losses['loss'] = ((losses['log_post'] - losses['log_prior']) * self.scale_kl - losses['log_like']) * self.scale_loss
            else:
                losses['loss'] = ((losses['log_post'] - losses['log_prior']) * self.scale_kl - losses['log_like']) * self.scale_loss + losses['z_regularize']
        return losses

    def eval_log_prior(self):
        log_prior_W = self.W_prior.log_prob(self.W)
        log_prior_b = self.b_prior.log_prob(self.b)
        log_prior = log_prior_W.mean(dim=-1).sum(dim=-1) + log_prior_b.mean(dim=-1).sum(dim=-1)
        return log_prior

    def eval_log_post(self):
        p_post_W = Normal(self.W_mu, torch.log(1 + torch.exp(self.W_rho)))
        p_post_b = Normal(self.b_mu, torch.log(1 + torch.exp(self.b_rho)))
        log_post_W = p_post_W.log_prob(self.W)
        log_post_b = p_post_b.log_prob(self.b)
        log_post = log_post_W.mean(dim=-1).sum(dim=-1) + log_post_b.mean(dim=-1).sum(dim=-1)
        return log_post

    def eval_log_like(self, y_preds, y):
        mu = y_preds.mean(dim=-1)
        sigma = y_preds.std(dim=-1)
        sigma.data[sigma < self.std_min] = self.std_min
        p_like = Normal(mu, sigma)
        log_like = p_like.log_prob(y.flatten())
        log_like = log_like.mean(dim=-1)
        return log_like

    def eval_z_regularize(self, z):
        l1_term = z.abs().sum() / 10
        l2_term = (z ** 2).sum() / 10
        z_abs = z.abs()
        z_tilde = z_abs / (z_abs.sum(dim=1, keepdim=True) + 1e-08)
        sigma_w = torch.log(1 + torch.exp(self.W_rho)).squeeze(-1)
        sigma_w_tilde = sigma_w / (sigma_w.sum() + 1e-08)
        sigma_w_tilde = sigma_w_tilde.unsqueeze(0)
        # Keep the original offset and scale: this is 30 * (LPAS - 1).
        anti_alignment_penalty = -(z_tilde * sigma_w_tilde).sum(dim=1).mean() * 30
        if self.z_regularize_opt == 'L1':
            return l1_term
        elif self.z_regularize_opt == 'L2':
            return l2_term
        elif self.z_regularize_opt == 'anti_alignment':
            return anti_alignment_penalty
        elif self.z_regularize_opt == 'agl':
            return -anti_alignment_penalty
        else:
            raise ValueError(f'{self.z_regularize_opt} is not an option')

    def sampling(self):
        if self.bayes:
            if self.output_dim != 1:
                raise ValueError('Independent sampling currently supports scalar properties only')
            # One independent epsilon per weight and MC draw; bias is independent too.
            epsilon_w = torch.randn((self.fingerprint_channels, self.n_rand),
                                    device=self.W_mu.device, dtype=self.W_mu.dtype)
            epsilon_b = torch.randn((self.n_rand,), device=self.b_mu.device, dtype=self.b_mu.dtype)
            self.W = self.W_mu + torch.log(1 + torch.exp(self.W_rho)) * epsilon_w
            self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * epsilon_b
        return (self.W, self.b)

    def predict(self, dataset, edge_attr_dim=7):
        self.device = 'cpu'
        self.to(self.device)
        if len(dataset) == 1:
            _dataset = deepcopy(dataset[0])
            _dataset = _dataset.to(self.device)
            if _dataset.edge_index.dim() == 1:
                _dataset.edge_index = torch.empty((2, 0), dtype=torch.long)
            if _dataset.edge_attr.dim() == 1:
                _dataset.edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
        else:
            _dataset = deepcopy(dataset)
            _dataset = DataLoader(_dataset, batch_size=len(_dataset), shuffle=False, follow_batch=self.follow_batch)
            _dataset = list(_dataset)[0]
            _dataset.to(self.device)
        self.eval()
        with torch.no_grad():
            y_pred = self(_dataset)
        if self.scaling:
            return self.detach(self.scaler.inverse_transform(y_pred), to_numpy=True)
        else:
            return self.detach(y_pred, to_numpy=True)

    def get_y_span(self, datasets):
        dataset_train = datasets['train']
        Y = np.array([float(data.y) for data in dataset_train])
        return Y.max() - Y.min()

    def scale_dataset(self, datasets):
        dataset_train = datasets['train']
        dataset_validate = datasets['validate']
        dataset_test = datasets['test']
        if self.scaling:
            if output_dims[self.prop] == 1:
                self.scaler = Scaler(prop=self.prop, mode='standardization', device=self.device)
            elif self.prop == 'sp':
                self.scaler = Scaler(prop=self.prop, mode='normalization', feature_range=(0, 30), device=self.device)
            else:
                raise NotImplementedError('Unsupported configuration')
            self.scaler.fit([*dataset_train, *dataset_validate, *dataset_test])
            dataset_train = self.scaler.transform_dataset(dataset_train)
            dataset_validate = self.scaler.transform_dataset(dataset_validate)
            dataset_test = self.scaler.transform_dataset(dataset_test)
        return (dataset_train, dataset_validate, dataset_test)

    def batch_data_creation(self, dataset_train, dataset_validate, dataset_test):
        if self.device == 'cpu':
            pin_memory = False
            persistent_workers = False
            num_workers = 0
        else:
            pin_memory = True
            persistent_workers = True
            num_workers = 4
        dataset_train_loader = DataLoader(dataset_train, batch_size=self.n_batch, shuffle=True, follow_batch=self.follow_batch, persistent_workers=persistent_workers, pin_memory=pin_memory, num_workers=num_workers)
        dataset_validate_loader = DataLoader(dataset_validate, batch_size=len(dataset_validate), shuffle=True, follow_batch=self.follow_batch, persistent_workers=persistent_workers, pin_memory=pin_memory, num_workers=num_workers)
        dataset_test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True, follow_batch=self.follow_batch, persistent_workers=persistent_workers, pin_memory=pin_memory, num_workers=num_workers)
        dataset_validate = list(dataset_validate_loader)[0].to(self.device)
        dataset_test = list(dataset_test_loader)[0].to(self.device)
        return (dataset_train_loader, dataset_validate, dataset_test)

    def fit(self, datasets, checkpointdir=None, n_epoch=500, n_batch=256, n_rand=100, lr=0.001, lr_scheduling=True, min_lr=1e-06, patience=10, factor=0.8, resume=False, scale_loss=1000.0, scale_kl=1, scale_prior=1, z_regularize_opt=None, scaling=True, no_decay=False, loss_thres_rel=0.001, error_thres_rel=0.001, loss_patience=60, loss_tolerance=0.0, clipping=True, gradient_bounds=(-10, 10), warmstart=True, etype='mae', metric_per_epoch=True):
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.n_rand = n_rand
        self.lr = lr
        self.lr_scheduling = lr_scheduling
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.resume = resume
        self.scale_loss = scale_loss
        self.scale_kl = scale_kl
        self.scale_prior = scale_prior
        self.scaling = scaling
        self.z_regularize_opt = z_regularize_opt
        self.no_decay = no_decay
        self.loss_thres_rel = loss_thres_rel
        self.error_thres_rel = error_thres_rel
        self.loss_patience = loss_patience
        self.loss_tolerance = loss_tolerance
        self.clipping = clipping
        self.warmstart = warmstart
        self.gradient_bounds = gradient_bounds
        self.etype = etype
        self.metric_per_epoch = metric_per_epoch
        if self.bayes:
            if self.warmstart:
                cprint('Load GNN model for training BGNN with warm start.', color='c')
                self.load_weights(Path(checkpointdir).with_name('checkpoint_GNN.pt'))
            self.set_priors()
        if not self.resume:
            self.optimizer, self.scheduler = self.setup_optimizer()
        cprint('Unlink datasets...', color='c')
        datasets = deepcopy(datasets)
        cprint('Scaling datasets...', color='c')
        self.y_span = self.get_y_span(datasets)
        dataset_train, dataset_validate, dataset_test = self.scale_dataset(datasets)
        cprint('Creating Dataloader objects...', color='c')
        dataset_train_loader, dataset_validate, dataset_test = self.batch_data_creation(dataset_train, dataset_validate, dataset_test)
        self.n_total = len(dataset_train)
        self.b_total = len(dataset_train_loader)
        if self.n_total > 100000:
            migrate = True
        else:
            migrate = False
        cprint('Training initiated...', color='c')
        self.to(self.device)
        self.train()
        logs = {}
        for e in range(self.n_epoch):
            recorder = {}
            self.e = e
            self.n_finished = 0
            self.start_time = time()
            for b, dataset_batch in enumerate(dataset_train_loader):
                dataset_batch.to(self.device)
                self.optimizer.zero_grad()
                losses = self.eval_loss(dataset_batch)
                losses['loss'].backward(retain_graph=True)
                self.optimizer.step()
                n = len(dataset_batch)
                self.n_finished += n
                self.print_progress(losses)
                if self.metric_per_epoch:
                    recorder = self.record_metrics(recorder, losses, n, migrate)
            self.eval()
            with torch.no_grad():
                if self.metric_per_epoch:
                    losses = self.evaluate_metric_per_epoch(recorder)
                losses[self.etype + '_validate'] = self.eval_total_error(dataset_validate)
            self.train()
            if self.lr_scheduling:
                self.scheduler.step(losses[self.etype + '_validate'])
            lr = self.optimizer.param_groups[0]['lr']
            self.print_progress(losses, lr)
            logs = self.record_log(logs, losses, lr)
            if self.satisfy_termination_criteria(logs):
                break
            elif self.lr_scheduling and self.scheduler.num_bad_epochs > self.scheduler.patience:
                cprint('Training has been early stopped at epoch', e, 'by scheduler.', color='g')
                break
        self.eval()
        with torch.no_grad():
            logs[self.etype + '_test'] = self.detach(self.eval_total_error(dataset_test))
        for key, value in logs.items():
            logs[key] = np.array(value)
        error_train = logs[self.etype + '_train'][-1]
        error_validate = logs[self.etype + '_validate'][-1]
        error_test = logs[self.etype + '_test']
        cprint('Final metrics:', self.etype.upper(), '| training:', error_train, '| validation:', error_validate, '| test:', error_test, color='c')
        cprint('Training finished.', color='g')
        return logs

    def satisfy_termination_criteria(self, logs):
        initial_loss = float(logs['loss'][0])
        loss = float(logs['loss'][-1])
        # A negative/zero initial NLL has no meaningful fractional-reduction ratio.
        if len(logs['loss']) > 1 and initial_loss > 0:
            if loss / initial_loss < self.loss_thres_rel:
                cprint('Training converged by relative loss.', color='g')
                return True
        error = float(logs[self.etype + '_validate'][-1])
        if self.y_span > 0 and error / self.y_span < self.error_thres_rel:
            cprint('Training converged by validation error / training target span.', color='g')
            return True
        if len(logs['loss']) > self.loss_patience:
            recent_losses = logs['loss'][-self.loss_patience:]
            best_recent = min(recent_losses)
            best_past = min(logs['loss'][:-self.loss_patience])
            if best_past - best_recent <= self.loss_tolerance:
                cprint(f'Training stopped: no loss improvement for {self.loss_patience} epochs.', color='g')
                return True
        return False

    def detach(self, x, clone=True, device='cpu', to_numpy=False):
        if clone:
            if to_numpy:
                return x.clone().to(device).detach().numpy()
            else:
                return x.clone().to(device).detach()
        elif to_numpy:
            return x.to(device).detach().numpy()
        else:
            return x.to(device).detach()

    def record_metrics(self, recorder, losses, n, migrate):
        if 'n' not in recorder.keys():
            recorder['n'] = [n]
        else:
            recorder['n'].append(n)
        for key, value in losses.items():
            if migrate:
                value = self.detach(value, device='cpu')
            if key not in recorder.keys():
                recorder[key] = [value]
            else:
                recorder[key].append(value)
        return recorder

    def record_log(self, logs, losses, lr):
        if self.e == 0:
            logs['lr'] = []
            for name, value in losses.items():
                logs[name] = []
        logs['lr'].append(lr)
        for name, value in losses.items():
            logs[name].append(self.detach(value))
        return logs

    def reorder_dict(self, data, lastkey):
        data = {key: value for key, value in sorted(data.items())}
        if lastkey in data:
            value = data.pop(lastkey)
            data_ = {**data, lastkey: value}
            return data_
        else:
            return data

    def print_progress(self, losses, lr=None):
        losses = self.reorder_dict(losses, lastkey='lr')
        ratio = self.n_finished / self.n_total * 100.0
        print('\rEpoch #%d | %d / %d (%6.1f %%) ' % (self.e + 1, self.n_finished, self.n_total, ratio), end='')
        for name, value in losses.items():
            print('| %s = %8.1f ' % (name, value), end='')
        elapsed_time = time() - self.start_time
        eta = elapsed_time / self.n_finished * (self.n_total - self.n_finished)
        if lr:
            print('| lr: %1.1e' % lr, end='')
            print('')
        else:
            print('| ETA: %s ' % self.time_format(eta), end='')
        sys.stdout.flush()
        return

    def time_format(self, t):
        m, s = divmod(t, 60)
        m = int(m)
        s = int(s)
        if m == 0:
            return '%d sec' % s
        else:
            return '%d min %d sec' % (m, s)

class BGNN_unary(BGNN):

    def __init__(self, prop, conv_model, num_node_features, num_edge_features, num_layers=5, hidden_channels=128, fingerprint_channels=64, dropout=0.3, bayes=False, device='cuda'):
        super().__init__(prop=prop, num_layers=num_layers, hidden_channels=hidden_channels, fingerprint_channels=fingerprint_channels, device=device, bayes=bayes)
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.dropout = dropout
        self.follow_batch = ['x']
        self.conv_model = conv_model
        self.conv_layer = select_conv_layer(self.conv_model)
        self.GNN = self.conv_layer(in_channels=num_node_features, hidden_channels=hidden_channels, out_channels=fingerprint_channels, num_layers=num_layers, dropout=dropout)

    def _forward(self, data):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch
        z_conv = self.GNN.forward(x=x, edge_attr=edge_attr, edge_index=edge_index)
        self.z = global_mean_pool(z_conv, batch)
        return self.z
