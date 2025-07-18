import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import Option
from data_utils.dataset import load_data
from encoders.model import Model
from utils.util import build_optimizer, save_checkpoint, setup_seed, show_grad
from utils.loss import triplet_loss, rn_loss
from utils.valid import valid_cls

from data_utils.vis import vis_pil_tensor, vis_s5_data
from utils.logger import create_logger


def train():
    train_data, sk_valid_data, im_valid_data = load_data(args)

    model = Model(args)
    model = model.cuda()

    # batch=15, lr=1e-5 / batch=30, lr=2e-5
    optimizer = build_optimizer(args, model)

    train_data_loader = DataLoader(train_data, args.batch, num_workers=2, drop_last=True)

    start_epoch = 0
    accuracy = 0

    for i in range(start_epoch, args.epoch):
        logger.info('------------------------train------------------------')
        epoch = i + 1
        model.train()
        torch.set_grad_enabled(True)

        start_time = time.time()
        num_total_steps = args.datasetLen // args.batch

        for index, (sk, im, sk_neg, im_neg, sk_label, im_label, _, _) in enumerate(train_data_loader):
            
            """
            vis_s5_data(sk[0], title="sk", coor_mode="REL")
            vis_s5_data(sk_neg[0], title="sk_neg", coor_mode="REL")
            vis_pil_tensor(im[0], title="im")
            vis_pil_tensor(im_neg[0], title="im_neg")
            """
            # prepare data
            sk = torch.cat((sk, sk_neg))
            im = torch.cat((im, im_neg))
            sk, im = sk.cuda(), im.cuda()

            # calculate feature
            cls_fea = model(sk, im)

            # loss
            losstri = triplet_loss(cls_fea, args)
            loss = losstri

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            step = index + 1
            if step % 30 == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logger.info(f'epoch_{epoch} step_{step} eta {remaining_time}: loss:{loss.item():.3f}')

        if epoch >= 1:
            logger.info('------------------------valid------------------------')
            # log
            map_all, map_200, precision_100, precision_200 = valid_cls(args, model, sk_valid_data, im_valid_data)
            logger.info(f'map_all:{map_all:.4f} map_200:{map_200:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')
            # save
            if map_all > accuracy:
                accuracy = map_all
                precision = precision_100
                logger.info("Save the BEST {}th model......".format(epoch))
                save_checkpoint(
                    {'model': model.state_dict(), 'epoch': epoch, 'map_all': accuracy, 'precision_100': precision},
                    args.save, f'best_checkpoint')


if __name__ == '__main__':
    args = Option().parse()
    logger = create_logger(args)
    args.logger = logger
    logger.info("train args:" + str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    logger.info("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    train()
