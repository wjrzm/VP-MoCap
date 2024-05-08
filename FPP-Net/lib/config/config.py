import os
from yacs.config import CfgNode as CN


class config_cont():

    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.logdir = ''
        self.cfg.load_net_checkpoint = ''
        self.cfg.checkpoint_path = ''
        self.cfg.result_path = ''

        self.cfg.trainer = CN()
        self.cfg.trainer.lr = 1e-4
        self.cfg.trainer.epochs = 400
        self.cfg.trainer.module = ''
        self.cfg.trainer.path = ''
        self.cfg.trainer.num_train_epochs = 100
        self.cfg.trainer.w_press = 0.4
        self.cfg.trainer.w_cont = 0.6

        self.cfg.dataset = CN()
        self.cfg.dataset.module = ''
        self.cfg.dataset.path = ''
        self.cfg.dataset.datadir = ''
        self.cfg.dataset.seq_name = ''
        self.cfg.dataset.tv_fn = ''
        self.cfg.dataset.w_sc = 1.0
        # self.cfg.dataset.w_bsc = 1.0
        self.cfg.dataset.w_nc = 1.0
        self.cfg.dataset.img_res = 224
        self.cfg.dataset.serial_batches = False
        self.cfg.dataset.pin_memory = False
        self.cfg.dataset.aug = CN()
        self.cfg.dataset.aug.is_aug = True
        self.cfg.dataset.aug.is_sam = False
        self.cfg.dataset.aug.noise_factor = 0.4
        self.cfg.dataset.aug.scale_factor = 0.15  # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.cfg.dataset.aug.rot_factor = 30  # Random rotation in the range [-rot_factor, rot_factor]
        self.cfg.dataset.g2g_num = 100  # Random rotation in the range [-rot_factor, rot_factor]

        self.cfg.networks = CN()
        self.cfg.networks.module = ''
        self.cfg.networks.path = ''
        self.cfg.networks.seqlen = 5

        self.cfg.record = CN()
        self.cfg.record.save_freq = 1
        self.cfg.record.show_freq = 1
        self.cfg.record.print_freq = 1

        self.cfg.gpus = ''
        self.cfg.batch_size = 0

    def get_cfg(self):
        return self.cfg.clone()

    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()