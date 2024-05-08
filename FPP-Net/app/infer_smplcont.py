from lib.config.config import config_cont as config
import argparse
import torch
import os,cv2
from tqdm import tqdm,trange
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from icecream import ic
import numpy as np

from lib.Dataset import make_dataset
from lib.Networks import make_network
from lib.Dataset.InsoleModule import InsoleModule
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--batch_size', default=1,type=int)
    parser.add_argument('--gpus', type=str, default='cpu', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.num_threads = arg.num_threads
    cfg.gpus = arg.gpus
    cfg.batch_size = arg.batch_size
    cfg.freeze()
    ic(cfg)

    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs('%s/%s' % (cfg.result_path, cfg.name), exist_ok=True)

    dataset_bsc = make_dataset.make_dataset(cfg.dataset, phase='test')
    dataloader = DataLoader(dataset_bsc, batch_size=cfg.batch_size, shuffle=False,
               num_workers=cfg.num_threads, pin_memory=cfg.dataset.pin_memory)

    gc_net = make_network.make_network(cfg.networks)

    if cfg.gpus != 'cpu':
        gpu_ids = [int(i) for i in cfg.gpus.split(',')]
        device = torch.device('cuda:%d' % gpu_ids[0])
        gc_net = gc_net.to(device)
        gc_net = DataParallel(gc_net, gpu_ids)

    else:
        gpu_ids = None
        device = torch.device('cpu')

    if os.path.exists(cfg.load_net_checkpoint):
        print('gc_net : loading from %s' % cfg.load_net_checkpoint)
        gc_net.load_state_dict(torch.load(cfg.load_net_checkpoint))
        gc_net.eval()
    else:
        print('can not find checkpoint %s'% cfg.load_net_checkpoint)

    mse_press_ls,bce_cont_ls,mse_cont_ls = [],[],[]
    insole_ratio = 10
    m_insole = InsoleModule('/data/PressureDataset')
    for data in tqdm(dataloader):
        for data_item in ['keypoints']:
            data[data_item] = data[data_item].to(device=device)
        pred_press, pred_cont = gc_net(keypoints=data['keypoints'])
        pred_press = pred_press.detach().cpu()
        pred_cont = pred_cont.detach().cpu()

        mse_press = F.mse_loss(pred_press, data['insole'])
        mse_cont = F.mse_loss(pred_cont, data['contact_smpl'])
        pred_cont[pred_cont>0.5] = 1
        bce_cont = F.binary_cross_entropy(pred_cont, data['contact_smpl'])
        mse_press_ls.append(mse_press.numpy())
        bce_cont_ls.append(bce_cont.numpy())
        mse_cont_ls.append(mse_cont.numpy())

        pred_press = pred_press.numpy().reshape([-1])
        press_gt = data['insole'].detach().cpu().numpy().reshape([-1])

        pred_cont = pred_cont.numpy().reshape([-1]).reshape([2,-1])
        cont_gt = data['contact_smpl'].detach().cpu().numpy().reshape([-1]).reshape([2,-1])

        # vis iamge
        press_gt_img, press_gt_data = m_insole.visMaskedPressure(press_gt)
        press_pred_img, press_pred_data = m_insole.visMaskedPressure(pred_press)
        press_gt_img = cv2.resize(press_gt_img,(press_gt_img.shape[1]*insole_ratio,press_gt_img.shape[0]*insole_ratio))
        press_pred_img = cv2.resize(press_pred_img,(press_pred_img.shape[1]*insole_ratio,press_pred_img.shape[0]*insole_ratio))

        image_path = data['case_name'][0]
        data_id,sub_ids,seq_name,frame_ids = (image_path).split('/')
        img_fn = os.path.join(cfg.dataset.datadir,'%s/%s/%s/color/%s.png'%(data_id,sub_ids,seq_name,frame_ids))
        img = cv2.imread(img_fn)

        insole_img = np.concatenate([press_gt_img,press_pred_img],axis=1)
        # insole_img = cv2.resize(insole_img,(insole_img.shape[1]*insole_ratio,insole_img.shape[0]*insole_ratio))
        img = cv2.resize(img, (insole_img.shape[1], int(img.shape[0] / (img.shape[1] / insole_img.shape[1]))))
        save_img = np.concatenate([img,insole_img],axis=0)
        save_fn = os.path.join(cfg.result_path, cfg.name,'%s_%s_%s_%s.png'%(data_id,sub_ids,seq_name,frame_ids))
        cv2.imwrite(save_fn,save_img)

        results={
            'pressure':{
                'gt':press_gt_data,
                'pred':press_pred_data,},
            'contact_smpl':{
                'gt':cont_gt,
                'pred':pred_cont,}}
        np.save(save_fn[:-3]+'npy',results)

    print("pressure mse:",np.mean(mse_press_ls),'cont mse:',np.mean(mse_cont_ls),'cont bce:',np.mean(bce_cont_ls))