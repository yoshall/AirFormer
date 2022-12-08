from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import Tensor
import numpy as np
import os
from src.utils.scaler import StandardScaler

def get_dataloader(datapath, batch_size, output_dim, mode='train'):
    '''
    get data loader from preprocessed data
    '''
    data = {}
    processed = {}
    results = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scalers = []
    for i in range(output_dim):
        scalers.append(StandardScaler(mean=data['x_train'][..., i].mean(),
                                      std=data['x_train'][..., i].std()))

    # Data format
    for category in ['train', 'val', 'test']:
        # normalize the target series (generally, one kind of series)
        for i in range(output_dim):
            data['x_' + category][..., i] = scalers[i].transform(data['x_' + category][..., i])
            data['y_' + category][..., i] = scalers[i].transform(data['y_' + category][..., i])

        new_x = Tensor(data['x_' + category])
        new_y = Tensor(data['y_' + category])
        processed[category] = TensorDataset(new_x, new_y)

    results['train_loader'] = DataLoader(processed['train'], batch_size, shuffle=True)
    results['val_loader'] = DataLoader(processed['val'], batch_size, shuffle=False)
    results['test_loader'] = DataLoader(processed['test'], batch_size, shuffle=False)

    print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
                                                   len(results['val_loader'].dataset),
                                                   len(results['test_loader'].dataset)))
    results['scalers'] = scalers
    return results

def check_device(device=None):
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)

def get_num_nodes(dataset):
    d = {'AIR_TINY': 1085}
    assert dataset in d.keys()
    return d[dataset]
