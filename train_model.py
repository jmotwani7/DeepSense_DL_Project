import argparse
import copy

import torch
import yaml

from model.lossFunction import inverseHuberLoss
from model.model_architecture import Resnet50BasedModel
from utils.trainutil import adjust_learning_rate, train, validate
from utils.datasetutil import get_nyuv2_test_train_dataloaders, load_test_train_ids
import pickle
from pathlib import Path
import os


def argparser():
    parser = argparse.ArgumentParser(description='DeepSense_DL_Project')
    parser.add_argument('-c', '--config', default='./config_resnet.yaml')
    return parser


def save_pickle(data, file_path):
    if not Path(file_path).parent.is_dir():
        Path(file_path).parent.mkdir(exist_ok=True)
    with open(file_path, 'w') as f:
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
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.reg)
    best = float('inf')
    # best_cm = None
    best_model = None
    train_ids, test_ids = load_test_train_ids('datasets/Nyu_v2/test_train_ids.json')
    train_loader, test_loader = get_nyuv2_test_train_dataloaders('datasets/Nyu_v2/nyu_depth_v2_labeled.mat', train_ids, test_ids)

    criterion = inverseHuberLoss
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        print(f'********* Training Started for epoch {epoch} *********')
        train_loss = train(epoch, train_loader, model, optimizer, criterion)
        train_losses.append(train_loss)

        # validation loop
        print(f'********* Validation Started for epoch {epoch} *********')
        loss = validate(epoch, test_loader, model, criterion)
        val_losses.append(loss)

        if loss < best:
            best = loss
            # best_cm = cm
            best_model = copy.deepcopy(model)
            if args.save_best:
                torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + f'-{epoch}.pth')
        save_pickle(train_losses, 'metrics/train_berHu_default.pkl')
        save_pickle(val_losses, 'metrics/val_berHu_default.pkl')

    print('Best Prec @1 Loss: {:.4f}'.format(best))
    # per_cls_acc = best_cm.diag().detach().numpy().tolist()
    # for i, acc_i in enumerate(per_cls_acc):
    #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))


if __name__ == '__main__':
    main()
