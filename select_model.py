import os

import torch


def select_best_model(mode):
    model_folder_path = r'./models/' + mode + '/'
    min_loss = None
    best_model = None

    for liqu_model_name in os.listdir(model_folder_path):
        model = torch.load(model_folder_path + liqu_model_name)
        if min_loss is None or model['loss'] < min_loss:
            min_loss = model['loss']
            best_model = model
    print_model_information(best_model)
    torch.save(best_model, './models/best_' + mode + '.pth')


def print_model_information(model):
    print(f'\nbest model:')
    print(f'Epoch:{model["epoch"]}')
    print(f'loss:{model["loss"]}')
    print(f'mem_days:{model["config"].mem_days}')
    print(f'hidden_size:{model["config"].hidden_size}')
    print(f'num_layers:{model["config"].num_layers}')


if __name__ == '__main__':
    mode = 'liqu_model'

    select_best_model(mode)
    # select_best_model(pres_model_path)

    print_model_information(torch.load('./models/best_liqu_model.pth'))

