import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-d', '--dataset', type=str, default='mr')
    arg.add_argument('-f', '--finetune', type=bool, default=False)
    arg.add_argument('-b', '--batch_size', type=int, default=32)
    arg.add_argument('--data_dir', type=str, default='/remote-home/dqwang/PJTask/SLSTM/data/cls')
    arg.add_argument('--glove_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt')
    arg.add_argument('--dropout', type=float, default=0.2)
    arg.add_argument('--max-len', type=int, default=100)
    arg.add_argument('--use-cls', type=bool, default=False)
    arg.add_argument('--use_lstm', type=bool, default=False)
    arg.add_argument('-l', '--layer', type=int, default=0)
    return arg.parse_args()
