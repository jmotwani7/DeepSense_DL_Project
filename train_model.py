import argparse
import copy

import torch
import yaml

from model.lossFunction import inverseHuberLoss
from model.model_architecture import Resnet50BasedModel
from utils.trainutil import adjust_learning_rate, train, validate
from utils.datasetutil import get_nyuv2_test_train_dataloaders


def argparser():
    parser = argparse.ArgumentParser(description='DeepSense_DL_Project')
    parser.add_argument('-c', '--config', default='./config_resnet.yaml')
    return parser


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
    train_loader, test_loader = get_nyuv2_test_train_dataloaders('datasets/Nyu_v2/nyu_depth_v2_labeled.mat')

    criterion = inverseHuberLoss
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        loss = validate(epoch, test_loader, model, criterion)

        if loss < best:
            best = loss
            # best_cm = cm
            best_model = copy.deepcopy(model)

    print('Best Prec @1 Loss: {:.4f}'.format(best))
    # per_cls_acc = best_cm.diag().detach().numpy().tolist()
    # for i, acc_i in enumerate(per_cls_acc):
    #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')


if __name__ == '__main__':
    main()