import argparse


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train or test')
    parser.add_argument('--n_exp', type=int, default=0,
                        help='experiment index')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to run')
    parser.add_argument('--seed', type=int, default=0)

    # data
    parser.add_argument('--dataset', type=str, default='AIR_TINY')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--aug', type=float, default=1.0)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--input_dim', type=int, default=27)
    parser.add_argument('--output_dim', type=int, default=1)

    # training
    parser.add_argument('--max_epochs', type=int, default=250) 
    parser.add_argument('--save_iter', type=int, default=400)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--patience', type=int, default=5)

    # test
    parser.add_argument('--save_preds', type=bool, default=False)
    return parser