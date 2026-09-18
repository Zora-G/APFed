import copy
import contextlib

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import get_model_update


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, device=None, training_seed=None):
        self.args = args
        self.device = device if device is not None else _device(args)
        self.fixed_batches_per_epoch = int(getattr(
            args, 'fixed_batches_per_epoch', 0))
        if self.fixed_batches_per_epoch < 0:
            raise ValueError('fixed_batches_per_epoch must be non-negative')
        if (self.fixed_batches_per_epoch > 0 and
                len(idxs) < self.fixed_batches_per_epoch * int(args.local_bs)):
            raise ValueError(
                'fixed local exposure requires at least {} samples; client has {}'.format(
                    self.fixed_batches_per_epoch * int(args.local_bs), len(idxs)))
        generator = None
        if training_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(training_seed))
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size=args.local_bs, shuffle=True,
                                      generator=generator)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, model, global_model=None, trace_out=None,
                       return_post_state=False):
        model = copy.deepcopy(model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=getattr(self.args, 'weight_decay', 0.0))
        epoch_loss = []
        optimizer_steps = 0
        samples_consumed = 0
        steps_per_epoch = []
        samples_per_epoch = []
        for _ in range(self.args.local_ep):
            batch_loss = []
            epoch_steps = 0
            epoch_samples = 0
            for batch_index, (images, labels) in enumerate(self.trainloader):
                if (self.fixed_batches_per_epoch > 0 and
                        batch_index >= self.fixed_batches_per_epoch):
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                optimizer_steps += 1
                epoch_steps += 1
                samples = int(labels.size(0))
                samples_consumed += samples
                epoch_samples += samples
            if (self.fixed_batches_per_epoch > 0 and
                    epoch_steps != self.fixed_batches_per_epoch):
                raise RuntimeError(
                    'fixed local exposure produced {} of {} requested batches'.format(
                        epoch_steps, self.fixed_batches_per_epoch))
            steps_per_epoch.append(epoch_steps)
            samples_per_epoch.append(epoch_samples)
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        loss = sum(epoch_loss) / max(len(epoch_loss), 1)
        if trace_out is not None:
            trace_out.clear()
            trace_out.update({
                'optimizer_steps': int(optimizer_steps),
                'samples_consumed': int(samples_consumed),
                'steps_per_epoch': [int(value) for value in steps_per_epoch],
                'samples_per_epoch': [int(value) for value in samples_per_epoch],
                'fixed_batches_per_epoch': int(self.fixed_batches_per_epoch),
                'batch_size': int(self.args.local_bs),
                'local_epochs': int(self.args.local_ep),
            })
        post_state = None
        if return_post_state:
            post_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if global_model is None:
            result = model.state_dict()
        else:
            result = get_model_update(model, global_model)
        if return_post_state:
            return result, loss, post_state
        return result, loss

    def inference(self, model):
        model.eval()
        total, correct, losses = 0, 0, []
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                losses.append(self.criterion(output, labels).item())
                correct += int(torch.sum(torch.argmax(output, dim=1) == labels).item())
                total += int(labels.size(0))
        return correct / max(total, 1), sum(losses) / max(len(losses), 1)

@contextlib.contextmanager
def _local_rng(training_seed, device):
    if training_seed is None:
        yield
        return
    devices = []
    if getattr(device, 'type', None) == 'cuda':
        devices = [0 if device.index is None else int(device.index)]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(training_seed))
        if devices:
            torch.cuda.manual_seed_all(int(training_seed))
        yield

def LocalTraining(args, model, dataset, idxs, global_model=None, device=None,
                  training_seed=None, trace_out=None,
                  return_post_state=False):
    """One edge node's SGD phase; returns g_i=theta_global-theta_local."""
    resolved_device = device if device is not None else _device(args)
    with _local_rng(training_seed, resolved_device):
        local = LocalUpdate(args, dataset, idxs, resolved_device,
                            training_seed=training_seed)
        return local.update_weights(
            model, global_model, trace_out=trace_out,
            return_post_state=return_post_state)

def _device(args):
    if getattr(args, 'gpu', -1) >= 0 and torch.cuda.is_available():
        return torch.device('cuda:' + str(args.gpu))
    return torch.device('cpu')
