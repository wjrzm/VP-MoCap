from tensorboardX import SummaryWriter
import torch
import numpy as np
import trimesh


class ContRecorder():
    def __init__(self, opt):
        self.iter = 0
        self.logdir = opt.logdir
        self.logger = SummaryWriter(self.logdir)

        self.save_freq = opt.record.save_freq
        self.show_freq = opt.record.show_freq
        self.print_freq = opt.record.print_freq

        self.checkpoint_path = opt.checkpoint_path
        self.result_path = self.logdir
        self.name = opt.name
        self.m_smpl = trimesh.load('../../bodyModels/smpl/smpl_uv/smpl_uv.obj',process=False)

    def init(self):
        self.iter = 0

    def logTensorBoard(self, l_data):
        self.logger.add_scalar('loss', l_data['loss'].item(), self.iter)
        # if l_data.__contains__('weight_loss'):
        #     self.logger.add_scalar('weight_loss', l_data['weight_loss'].item(), self.iter)
        self.iter += 1

    def logPressNetTensorBoard(self, l_data):
        self.logger.add_scalar('loss', l_data['loss'].item(), self.iter)
        self.logger.add_scalar('loss_press', l_data['loss_press'].item(), self.iter)
        self.logger.add_scalar('loss_cont', l_data['loss_cont'].item(), self.iter)
        self.iter += 1

    def log(self, l_data):
        # if (l_data['epoch']+1)% self.print_freq == 0:
        #     print('Epoch:%d loss:%f' % (l_data['epoch'], l_data['loss'].item()))
        if (l_data['epoch']+1) % self.save_freq == 0:
            print('Save checkpoint to %s/%s/latest.'% (self.checkpoint_path, self.name))
            torch.save(l_data['net'].state_dict(), '%s/%s/latest' % (self.checkpoint_path, self.name))
            torch.save(l_data['net'].state_dict(),
                    '%s/%s/net_epoch_%d' % (self.checkpoint_path, self.name, l_data['epoch']))

        if (l_data['epoch']+1) % self.show_freq == 0:
            print('Save visual.')
            labels_gt = l_data['data']['contact_labels'].detach().cpu().numpy()
            labels_pred = l_data['data']['pred'].detach().cpu().numpy()
            save_name = '%s/epoch%d_iter%d.npy' % (self.result_path, l_data['epoch'], self.iter)

            results={
                'img_path':l_data['img_path'],
                'labels_gt':labels_gt,
                'labels_pred':labels_pred
            }
            np.save(save_name,results)

