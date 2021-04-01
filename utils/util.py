import csv
import tensorboardX


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr()
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self):
        fmt = '{}'
        return '[' + fmt + ']'


class Metrics():
    def __init__(self, log_dir):
        self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    def log_metrics(self, train_losses, val_losses, epoch, lr):
        train_tot_loss = train_losses['total_loss']
        train_sbj_loss = train_losses['sbj_loss']
        train_obj_loss = train_losses['obj_loss']
        train_rel_loss = train_losses['rel_loss']

        val_tot_loss = val_losses['total_loss']
        val_sbj_loss = val_losses['sbj_loss']
        val_obj_loss = val_losses['obj_loss']
        val_rel_loss = val_losses['rel_loss']

        # val_tot_loss, val_sbj_loss, val_obj_loss, val_rel_loss = val_metrics

        # write summary
        self. summary_writer.add_scalar(
            'train_losses/train_tot_loss', train_tot_loss, global_step=epoch)
        self.summary_writer.add_scalar(
            'train_losses/train_sbj_loss', train_sbj_loss, global_step=epoch)
        self.summary_writer.add_scalar(
            'train_losses/train_obj_loss', train_obj_loss, global_step=epoch)
        self.summary_writer.add_scalar(
            'train_losses/train_rel_loss', train_rel_loss, global_step=epoch)

        # write summary
        self. summary_writer.add_scalar(
            'val_losses/val_tot_loss', val_tot_loss, global_step=epoch)
        self.summary_writer.add_scalar(
            'val_losses/val_sbj_loss', val_sbj_loss, global_step=epoch)
        self.summary_writer.add_scalar(
            'val_losses/val_obj_loss', val_obj_loss, global_step=epoch)
        self.summary_writer.add_scalar(
            'val_losses/val_rel_loss', val_rel_loss, global_step=epoch)

        self.summary_writer.add_scalar(
            'lr_rate', lr, global_step=epoch)

        # self.summary_writer.add_scalar(
        #     'losses/val_tot_loss', val_tot_loss, global_step=epoch)
        # self.summary_writer.add_scalar(
        #     'losses/val_sbj_loss', val_sbj_loss, global_step=epoch)
        # self.summary_writer.add_scalar(
        #     'losses/val_obj_loss', val_obj_loss, global_step=epoch)
        # self.summary_writer.add_scalar(
        #     'losses/val_rel_loss', val_rel_loss, global_step=epoch)
