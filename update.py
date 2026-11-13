"""Local SGD from the global model or a client's own model, including label/sign attacks."""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from utils import model_to_vector, vector_to_model


def local_gradient(args, model, dataset, indices, device, training_seed, attack='none',
                   start=None, return_state=False):
    """Return g_i = W_global - W_local, training from start (a parameter vector) if given.

    With return_state the local model after SGD is returned as well; empty clients
    keep their start model.
    """
    initial = model_to_vector(model)
    local = copy.deepcopy(model).to(device)
    if start is not None:
        vector_to_model(start, local)
    if len(indices) == 0:
        state = model_to_vector(local)
        gradient = initial - state
        gradient = -gradient if attack == 'sign' else gradient
        return (gradient, state) if return_state else gradient
    local.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.local_bs, shuffle=True,
                        generator=torch.Generator().manual_seed(training_seed), num_workers=0)
    optimizer = torch.optim.SGD(local.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=getattr(args, 'weight_decay', 0.0))
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.local_ep):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if attack == 'label':
                labels = labels.clone()
                labels[labels == args.source_label] = args.target_label
            optimizer.zero_grad()
            loss = criterion(local(images), labels)
            loss.backward()
            optimizer.step()
    state = model_to_vector(local)
    gradient = initial - state
    gradient = -gradient if attack == 'sign' else gradient
    return (gradient, state) if return_state else gradient


def collect_updates(args, model, dataset, groups, malicious, attack_active,
                    device, round_index, leaders=None, client_states=None):
    """Train every client once and return the uploads and the leaders' g_0.

    With client_states (a per-client list of parameter vectors) each client trains
    from its own model and the result replaces that entry before any filtering;
    a sign-flipping client keeps its honestly trained model and negates only the
    upload. A leader's g_0 is its own update of this round before the sign flip.
    """
    updates, leader_updates = {}, {}
    leader_ids = set(leaders or [])
    for client in range(args.num_users):
        attack = args.attack if attack_active and client in malicious else 'none'
        training_seed = args.seed + (round_index + 1) * args.num_users + client
        if client_states is None:
            gradient = local_gradient(args, model, dataset, groups[client], device,
                                      training_seed, attack)
        else:
            gradient, client_states[client] = local_gradient(
                args, model, dataset, groups[client], device, training_seed, attack,
                start=client_states[client], return_state=True)
        updates[client] = gradient
        if client in leader_ids:
            leader_updates[client] = -gradient if attack == 'sign' else gradient
    return updates, {i: leader_updates[leader] for i, leader in enumerate(leaders or [])}
