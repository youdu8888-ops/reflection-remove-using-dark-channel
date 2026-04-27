import os
from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data


def _build_eval_ceil(opt, p):
    if not (os.path.isdir(p) and os.path.isdir(join(p, 'blended')) and os.path.isdir(join(p, 'transmission_layer'))):
        return None
    ds = datasets.CEILTestDataset(p)
    return datasets.DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)


def _build_eval_real20(opt, p, size, max_edge):
    if not (os.path.isdir(p) and os.path.isdir(join(p, 'blended')) and os.path.isdir(join(p, 'transmission_layer'))):
        return None
    ds = datasets.CEILTestDataset(p, size=size, max_long_edge=max_edge)
    return datasets.DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)


def main():
    opt = TrainOptions().parse()

    cudnn.benchmark = True

    opt.display_freq = 10

    if opt.debug:
        opt.display_id = 1
        opt.display_freq = 20
        opt.print_freq = 20
        opt.nEpochs = 40
        opt.max_dataset_size = 100
        opt.no_log = False
        opt.nThreads = 0
        opt.decay_iter = 0
        opt.serial_batches = True
        opt.no_flip = True

    # processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
    datadir = './datasets/processed_data'
    datadir_real = join(datadir, 'real_train')
    train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

    if opt.real_data_only:
        train_dataset_fusion = train_dataset_real
        train_dataloader_fusion = datasets.DataLoader(
            train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
            num_workers=opt.nThreads, pin_memory=True)
    else:
        datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
        train_dataset = datasets.CEILDataset(
            datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True,
            low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
            low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)
        train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])
        train_dataloader_fusion = datasets.DataLoader(
            train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
            num_workers=opt.nThreads, pin_memory=True)

    eval_dataloader_ceilnet = _build_eval_ceil(opt, join(datadir, 'testdata_CEILNET_table2'))
    eval_dataloader_real = _build_eval_real20(opt, join(datadir, 'real20'), 20, 512)

    """Main Loop"""
    engine = Engine(opt)

    def set_learning_rate(lr):
        for optimizer in engine.model.optimizers:
            print('[i] set learning rate to {}'.format(lr))
            util.set_opt_param(optimizer, 'lr', lr)

    if opt.resume and eval_dataloader_ceilnet is not None:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

    # define training strategy
    engine.model.opt.lambda_gan = 0
    set_learning_rate(1e-4)
    while engine.epoch < 60:
        if engine.epoch == 20:
            engine.model.opt.lambda_gan = 0.01
        if engine.epoch == 30:
            set_learning_rate(5e-5)
        if engine.epoch == 40:
            set_learning_rate(1e-5)
        if (not opt.real_data_only) and engine.epoch == 45:
            ratio = [0.5, 0.5]
            print('[i] adjust fusion ratio to {}'.format(ratio))
            train_dataset_fusion.fusion_ratios = ratio
            set_learning_rate(5e-5)
        if engine.epoch == 50:
            set_learning_rate(1e-5)

        engine.train(train_dataloader_fusion)

        if engine.epoch % 5 == 0:
            if eval_dataloader_ceilnet is not None:
                engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')
            if eval_dataloader_real is not None:
                engine.eval(eval_dataloader_real, dataset_name='testdata_real20')


if __name__ == '__main__':
    main()
