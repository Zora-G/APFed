import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default='sign', choices=['sign', 'none'])
    parser.add_argument('--epochs', type=int, default=30, help='number of federated training rounds')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--inputs', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save', type=str, default='./save')
    parser.set_defaults(
        seed=0, local_ep=2, local_bs=16, lr=0.03,
        momentum=0.0, weight_decay=0.0001, fixed_batches_per_epoch=0,
        root_norm=0.0, crypto=0, packing=0, quant_bits=16, quant_clip=0.0,
        pack_bits=48, pad_bits=20, slots_per_ciphertext=3,
        cos_threshold=0.0, cra_eta=1.0, cra_lr=1.0, cra_mode='paper_eq')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30:
        parser.error('epochs must be in [1,30]')
    return args
