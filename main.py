"""Train APFed or a plain baseline from seeded random initialization."""

import json
import os
import sys
import warnings
import tempfile
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from aggregation import apfed_round, baseline_round, clustering_round
from clustering import cfl_clusters
from datasets import load_data
from models import CNNMnist, CNNCifar
from options import args_parser
from paillier import keygen
from test import evaluate
from update import collect_updates
from utils import model_to_vector, setup_seed, vector_to_model


def main(argv=None):
    args = args_parser(argv)
    setup_seed(args.seed)
    torch.set_num_threads(4)
    # Deterministic implementations are used where available; unsupported ops warn.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')
        if args.gpu >= 0:
            warnings.warn('CUDA is unavailable; using the permitted CPU fallback')
    print('Training device: ' + str(device), file=sys.stderr, flush=True)
    if args.crypto:
        warnings.warn('Per-coordinate Paillier is slow for full CNNs; use tiny instances for crypto checks')
    train, evaluation, groups = load_data(args)
    model = (CNNCifar() if args.dataset == 'cifar10'
             else CNNMnist(fc_width=args.mnist_fc_width)).to(device)
    vector = model_to_vector(model)
    # Persistent clients all start from W0; the CFL pass below does not change them.
    client_states = ([vector.clone() for _ in range(args.num_users)]
                     if args.client_state == 'persistent' else None)
    malicious = set(np.random.default_rng(args.seed).permutation(args.num_users)[
        :int(args.ratio * args.num_users)].tolist()) if args.attack != 'none' else set()
    clusters, leaders = [], []
    if args.defense == 'apfed':
        # All nodes are honest during the one-off CFL clustering at W0.
        initial, _ = collect_updates(args, model, train, groups, malicious,
                                     False, device, round_index=0)
        # CFL reports W_local - W0, the negative of the training upload convention.
        deltas = np.stack([-initial[i].numpy() for i in range(args.num_users)])
        clusters = cfl_clusters(deltas, args.cfl_threshold, args.cfl_min_size,
                                args.cfl_max_clusters)
        leader_rng = np.random.default_rng(args.seed)
        leaders = [int(leader_rng.choice(cluster)) for cluster in clusters]
    pk, sk = keygen(args.key_bits) if args.crypto and args.defense == 'apfed' else (None, None)
    run_name = args.run_name or '{}_{}_{}_seed{}'.format(
        args.dataset, args.defense, args.attack, args.seed)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        out = save_dir / run_name
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(tempfile.mkdtemp(prefix=run_name + '_', dir=save_dir))
        run_name = out.name
    config = dict(vars(args), device=str(device))
    config['run_name'] = run_name
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    with (out / 'metrics.jsonl').open('w') as metrics:
        def emit(row):
            line = json.dumps(row, sort_keys=True)
            metrics.write(line + '\n')
            metrics.flush()
            print(line, flush=True)

        emit({'round': 0, 'phase': 'initialization', 'accuracy': None, 'test_loss': None,
              'attack_success_rate': None, 'number_of_clusters': len(clusters),
              'cluster_sizes': [len(cluster) for cluster in clusters], 'clusters': clusters,
              'leaders': leaders, 'malicious_ids': sorted(malicious),
              'attack_active': args.attack != 'none' and args.warmup_rounds == 0,
              'detected_malicious_ids': [], 'false_positives': 0, 'false_negatives': 0,
              'cra_weights': [], 'cra_fallback': False})
        previous_update = None
        for round_index in range(1, args.warmup_rounds + args.rounds + 1):
            attack_active = args.attack != 'none' and round_index > args.warmup_rounds
            uploads, leader_gradients = collect_updates(
                args, model, train, groups, malicious, attack_active, device, round_index, leaders,
                client_states)
            if args.defense == 'apfed':
                new_vector, details = apfed_round(uploads, clusters, leader_gradients,
                                                  vector, args, pk, sk, previous_update)
            else:
                if args.defense == 'clustering':
                    new_vector, rejected = clustering_round(list(uploads.values()), vector,
                                                            args.server_eta)
                else:
                    new_vector, rejected = baseline_round(list(uploads.values()), vector,
                                                          args.defense, args.trim_ratio,
                                                          args.server_eta), []
                details = {'detected_malicious_ids': rejected, 'weights': [], 'cra_fallback': False,
                           'contributing_cluster_ids': [], 'skipped_round': False}
            previous_update = vector - new_vector
            vector = new_vector
            vector_to_model(vector, model)
            performance = evaluate(model, evaluation, device, args.source_label, args.target_label)
            if args.attack != 'label':
                performance['attack_success_rate'] = None
            detected = set(details['detected_malicious_ids'])
            active_malicious = malicious if attack_active else set()
            row = dict(round=round_index, attack_round=max(0, round_index - args.warmup_rounds),
                       attack_active=attack_active, number_of_clusters=len(clusters),
                       false_positives=len(detected - active_malicious),
                       false_negatives=len(active_malicious - detected), **performance)
            row.update(details)
            row['cra_weights'] = row.pop('weights')
            emit(row)
    (out / 'result.json').write_text(json.dumps(
        {'status': 'COMPLETED', 'total_rounds': args.warmup_rounds + args.rounds,
         'final': row}, indent=2) + '\n')
    return out


if __name__ == '__main__':
    main()
