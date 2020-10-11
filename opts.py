import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str,
                        help='path to weight file')
    parser.add_argument('--image_path', type=str,
                        help='input image path')
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')
    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=20, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)
    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    args = parser.parse_args()

    return args
