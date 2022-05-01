import argparse
import copy
import pickle
from pathlib import Path

import torch
import yaml

from model.lossFunction import inverseHuberLoss
from model.model_architecture import Resnet50BasedModel
from utils.datasetutil import get_nyuv2_test_train_dataloaders, load_test_train_ids, save_test_train_ids
from utils.trainutil import adjust_learning_rate, train, validate


def argparser():
    parser = argparse.ArgumentParser(description='DeepSense_DL_Project')
    parser.add_argument('-c', '--config', default='./config_resnet.yaml')
    return parser


def save_pickle(data, file_path):
    if not Path(file_path).parent.is_dir():
        Path(file_path).parent.mkdir(exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def main():
    args = argparser().parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.Loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    model = Resnet50BasedModel(device=device)
    if torch.cuda.is_available():
        model = model.cuda()
    # optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.reg)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    best = float('inf')
    # best_cm = None
    best_model = None
    train_ids, val_ids, test_ids = load_test_train_ids('datasets/Nyu_v2/train_val_test_ids.json')
    train_loader, val_loader, test_loader = get_nyuv2_test_train_dataloaders('datasets/Nyu_v2/nyu_depth_v2_labeled.mat', train_ids, val_ids, test_ids, batch_size=args.batch_size)

    criterion = inverseHuberLoss
    train_losses = []
    val_losses = []
    learning_rates = []
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f'Learning Rate for EPOCH {epoch} => {lr}')
        learning_rates.append(lr)

        # train loop
        print(f'********* Training Started for epoch {epoch} *********')
        train_loss = train(epoch, train_loader, model, optimizer, criterion)
        train_losses.append(train_loss)
        print(f'Training loss for the EPOCH {epoch} ==> {train_loss:.4f}')

        # validation loop
        print(f'********* Validation Started for epoch {epoch} *********')
        loss = validate(epoch, val_loader, model, criterion)
        val_losses.append(loss)
        print(f'Validation loss for the EPOCH {epoch} ==> {loss:.4f}')

        if loss < best:
            best = loss
            # best_cm = cm
            best_model = copy.deepcopy(model)
            if args.save_best:
                torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + f'-{epoch}.pth')
        save_pickle(train_losses, f'metrics/train_{args.model.lower()}_berHu_default.pkl')
        save_pickle(val_losses, f'metrics/val_{args.model.lower()}_berHu_default.pkl')
        save_pickle(learning_rates, f'metrics/learning_rate_{args.model.lower()}_default.pkl')

    print('Best Loss: {:.4f}'.format(best))


if __name__ == '__main__':
    main()
