import argparse


def args_parser(argv=None):
    parser = argparse.ArgumentParser(description='APFed')
    parser.add_argument('--dataset', choices=['mnist', 'fmnist', 'cifar10'], default='mnist')
    partition = parser.add_mutually_exclusive_group()
    partition.add_argument('--iid', action='store_true')
    partition.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_users', type=int, default=100)
    parser.add_argument('--attack', choices=['none', 'label', 'sign'], default='none')
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--source_label', type=int, default=1)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--defense', choices=['apfed', 'fedavg', 'median', 'trimmed', 'clustering'],
                        default='apfed')
    parser.add_argument('--rounds', type=int, default=30, help='rounds following the warm-up')
    parser.add_argument('--warmup_rounds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='CUDA index; -1 selects CPU')
    parser.add_argument('--crypto', type=int, choices=[0, 1], default=0)
    parser.add_argument('--key_bits', type=int, default=1024)
    parser.add_argument('--quant_bits', type=int, default=16)
    parser.add_argument('--cfl_threshold', type=float, default=0.5)
    parser.add_argument('--cfl_min_size', type=int, default=1)
    parser.add_argument('--cfl_max_clusters', type=int, default=0)
    parser.add_argument('--cos_threshold', type=float, default=0.0)
    parser.add_argument('--server_eta', type=float, default=1.0)
    parser.add_argument('--local_ep', type=int, default=2)
    parser.add_argument('--local_bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--client_state', choices=['global', 'persistent'], default='global',
                        help='model each client trains from in round t > 1: global = the current '
                             'global model, as in step 6 of APFed Fig. 2; persistent = its own '
                             'model after the previous round')
    parser.add_argument('--mnist_fc_width', type=int, default=256)
    parser.add_argument('--trim_ratio', type=float, default=0.1,
                        help='fraction of values Trimmed Mean removes from each side')
    parser.add_argument('--dirichlet_min_size', type=int, default=1,
                        help='redraw the Dirichlet partition until every client holds this many samples')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--save', default='./save')
    parser.add_argument('--run_name', default=None, help='output directory name; default: dataset/defense/attack/seed with a unique suffix')
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.warmup_rounds < 0:
        parser.error('rounds must be positive and warmup_rounds must be nonnegative')
    if args.num_users < 1 or args.local_ep < 1 or args.local_bs < 1:
        parser.error('num_users, local_ep and local_bs must be positive')
    if args.alpha <= 0 or not 0 <= args.ratio <= 1:
        parser.error('alpha must be positive and ratio must be in [0, 1]')
    if args.cfl_min_size < 1 or args.cfl_max_clusters < 0:
        parser.error('cfl_min_size must be positive and cfl_max_clusters must be nonnegative')
    if args.quant_bits < 0 or args.key_bits < 32:
        parser.error('quant_bits must be nonnegative and key_bits must be at least 32')
    if not 0 <= args.source_label < 10 or not 0 <= args.target_label < 10:
        parser.error('source_label and target_label must be class ids in [0, 9]')
    if args.defense == 'trimmed' and 2 * int(args.trim_ratio * args.num_users) >= args.num_users:
        parser.error('trimmed mean requires 2 * floor(trim_ratio * num_users) < num_users')
    if args.weight_decay < 0 or args.mnist_fc_width < 1 or args.dirichlet_min_size < 0:
        parser.error('weight_decay must be nonnegative, mnist_fc_width positive, '
                     'dirichlet_min_size nonnegative')
    return args
