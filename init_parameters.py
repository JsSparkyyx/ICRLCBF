import argparse
import torch

def init_parameters():
    parser = argparse.ArgumentParser(description="Learning Control Barrier Functions with ICRL.")
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--env_name', type=str, default="OfflineCarGoal2-v0")
    parser.add_argument('--epoch_dynamics', type=int, default=1)
    parser.add_argument('--epoch_cbf', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--regularizer_coeff', type=float, default=0.1)
    parser.add_argument('--name', type=str, default="ICRLCBF")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()._get_kwargs()
    args = {k:v for (k,v) in args}
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['name'] = f"{args['name']}_{str(args['seed'])}_{str(args['lr'])}_{str(args['weight_decay'])}"
    return args