import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import os

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument(
        '--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--save_models', help='Path to location saving models')

    parser.add_argument('--num_workers', help="Num workers",
                        type=int, default=3)

    parser.add_argument('--batch_size', help="Batch size", type=int, default=2)

    parser.add_argument(
        '--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs',
                        type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders

    if parser.csv_train is None:
        raise ValueError('Must provide --csv_train,')

    if parser.csv_classes is None:
        raise ValueError(
            'Must provide --csv_classes,')

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    print(
        f"[INFO] batch_size: {parser.batch_size}, num_workers: {parser.num_workers}")
    sampler = AspectRatioBasedSampler(
        dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(
        dataset_train, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            dataset_val, batch_size=parser.batch_size, drop_last=False)
        dataloader_val = DataLoader(
            dataset_val, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(
            num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = False

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet(
                        [data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet(
                        [data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            print("[INFO] mAP: ", mAP)

        scheduler.step(np.mean(epoch_loss))
        # if os.path.exists("../drive/My\ Drive/Colab\ Notebooks/models/facenet"):
        if parser.save_models is not None:
            if os.path.exists(parser.save_models):
                path = os.path.join(parser.save_models,
                                    f'retinanet_{epoch_num}.pt')
                print(f"[INFO] Saving model at: {path}")
                path_last = os.path.join(
                    parser.save_models, 'retinanet_last.pt')
                print(f"[INFO] Saving last model at: {path_last}")
                torch.save(
                    retinanet.module, path)
                torch.save(
                    retinanet.module, path_last)
            else:
                print(f"[INFO] Not found location: {parser.save_models}")
                print("[INFO] Auto saving model in: models/")
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(retinanet.module,
                           './models/retinanet_{}.pt'.format(epoch_num))
                torch.save(retinanet.module,
                           './models/retinanet_last.pt')
        else:
            print("[INFO] Auto saving model in: models/")
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(retinanet.module,
                       './models/retinanet_{}.pt'.format(epoch_num))
            torch.save(retinanet.module,
                       './models/retinanet_last.pt')

    retinanet.eval()

    # torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
