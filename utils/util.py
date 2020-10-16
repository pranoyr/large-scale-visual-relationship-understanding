import csv
import tensorboardX


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val 
        self.count += 1
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


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

