import os

from options import args_parser


if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

    import json
    from pathlib import Path

    import numpy as np
    import torch

    from aggregation import apfed_round
    from datasets import load_data
    from defense import TrustedAnchorGuard
    from models import CNNMnist
    from test import test_img
    from update import LocalTraining
    from utils import setup_seed, vector_to_model

    if not torch.cuda.is_available():
        raise RuntimeError('An NVIDIA CUDA device is required')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode('error')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    setup_seed(args.seed)
    out = Path(args.save) / ('sign50' if args.attack == 'sign' else 'clean')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    data, test, groups, seeds = load_data(args.data_dir, args.inputs)
    snap = torch.load(args.checkpoint, weights_only=False)
    vector = snap['model_vector'].clone()
    clusters = snap['clusters']
    guard = TrustedAnchorGuard(snap['cra_reference'], clusters)
    malicious = set(np.random.RandomState(0).permutation(100)[:50].tolist()) if args.attack == 'sign' else set()
    device = torch.device('cuda:0')
    (out / 'initial_state.json').write_text(json.dumps({
        'start_round': 0,
        'malicious_clients': sorted(malicious),
        'max_step_norm': guard.limit,
    }, indent=2))
    try:
        for t in range(1, 1 + args.epochs):
            model = CNNMnist(10, (16, 32, 32, 64), 256, 'relu', 'max')
            vector_to_model(vector, model)
            model.to(device)
            raw = {}
            for client in range(100):
                gradient, _ = LocalTraining(
                    args, model, data, groups[client], global_model=model,
                    device=device, training_seed=seeds[str(t)][client],
                )
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError('round{} client{}'.format(t, client))
                raw[client] = gradient.detach().cpu().float()
            uploaded = {i: -g if i in malicious else g for i, g in raw.items()}
            leaders = {i: raw[cluster[0]] for i, cluster in enumerate(clusters)}
            guard.last = {}
            result = apfed_round(uploaded, clusters, leaders, vector, args, guard, round_index=t)
            delta = float((result[0] - vector).double().norm())
            vector = result[0]
            vector_to_model(vector, model)
            accuracy, loss = test_img(model, test, args, device)
            detected = set(result[2])
            row = dict(
                round=t, attack_round=t, accuracy=accuracy, test_loss=loss,
                false_positives=len(detected-malicious),
                false_negatives=len(malicious-detected), actual_step_norm=delta,
                skipped_round=result[6], weights=result[4]['normalized_weights'],
                **guard.last,
            )
            with (out / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            torch.save(dict(round=t, model=vector, anchor=guard.anchor), out / 'latest.pt')
            print(json.dumps(row), flush=True)
        (out / 'result.json').write_text(json.dumps(dict(status='COMPLETED', final=row), indent=2))
    except Exception:
        import traceback
        (out / 'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=2))
        raise
