import hydra
import torch
import numpy as np
import copy
import logging
import os
import os.path as osp
from tqdm import tqdm

from lib.dataset.dataset_mmvp import Dataset
from lib.optimize_util import perspective_projection, encode, decode
from lib.loss.losses import OPENPOSE, HALPE, loss_2d, loss_3d_t
from lib.initial_trans.initial_trans import initial_trans
from lib.utils.visualize_utils import Camera, Visualizer
from lib.utils.render_utils import Renderer
from lib.utils.filter_utils import fil_pose, fil_trans
from models.smpl import SMPL
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    device = torch.device('cuda:{}'.format(cfg['gpu'])) if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    task_cfg = cfg['task']
    method_cfg = cfg['method']

    opt_result_path = osp.join(task_cfg['input_path_base'], 'opt_results')
    if not os.path.isdir(opt_result_path):
        os.makedirs(opt_result_path)

    smpl_model = SMPL(method_cfg['smpl_file'], scale=task_cfg['scale']).to(device)

    camera = Camera(cfg, dtype=dtype, device=device)
    dataset = Dataset(cfg)
    visualizer = Visualizer(cfg, opt_result_path)
    renderer = Renderer(focal_length=camera.focal_length, img_w=camera.img_w, img_h=camera.img_h,
                            faces=smpl_model.faces, same_mesh_color=False,
                            rotation=camera.rotation, translation=camera.translation)

    pose = torch.tensor(dataset.pose,dtype=dtype,device=device)
    betas = torch.tensor(dataset.betas,dtype=dtype,device=device)
    trans = initial_trans(cfg, dataset).to(device)

    kp_2d = torch.tensor(dataset.target_halpe_fil,dtype=dtype,device=device)
    kp_2d_score = torch.tensor(dataset.target_halpe_score,dtype=dtype,device=device)
    contact = dataset.contact

    point_cloud = torch.tensor(dataset.point_cloud, dtype=dtype, device=device)

    vp, ps = load_model('models/V02_05', model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vp = vp.to(device)

    mse = torch.nn.MSELoss()
    pose_arr = []
    beta_arr = []
    trans_arr = []

    last_output_joints = None
    for frame in range(pose.shape[0]):
        log.info(f'frame {frame}')

        pose_single = pose[frame].unsqueeze(0)
        beta_single = betas[frame].unsqueeze(0)
        trans_single = trans[frame].unsqueeze(0)
        kp_2d_single = kp_2d[frame].unsqueeze(0)
        kp_2d_score_single = kp_2d_score[frame].unsqueeze(0)
        contact_single = contact[frame]

        pred_body_poZ = encode(pose_single,vp,device)
        pred_body_poZ.requires_grad = True

        trans_root = copy.deepcopy(trans_single)
        trans_root.requires_grad = True

        params = [pred_body_poZ,trans_root]
        optimizer = torch.optim.Rprop(params,lr=method_cfg['lr'])
        # optimizer = torch.optim.Adam(params,lr=method_cfg['lr'])

        last_loss = 0

        for i in range(cfg['method']['max_iter']):
            
            optimizer.zero_grad()
            pose_single_rec = decode(pred_body_poZ,vp,device,pose_single)

            output_opt = smpl_model(betas=beta_single,
                                    body_pose=pose_single_rec[:,1:],
                                    global_orient=pose_single_rec[:,[0]],
                                    pose2rot=False,
                                    transl=trans_root)
                        
            kp_2d_opt = perspective_projection(output_opt.joints[:,:25],camera.rotation,camera.translation,camera.focal_length,camera.camera_center)
            source = kp_2d_opt[:,torch.tensor(OPENPOSE),:]
            target = kp_2d_single[:,torch.tensor(HALPE),:]

            loss1 = loss_2d(source, target, kp_2d_score_single, method_cfg['lambda_2d_limbs'], method_cfg['lambda_2d_feet'], method_cfg['lambda_2d_others'])
            loss2 = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
            loss3 = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)

            loss2, loss3 = loss_3d_t(contact_single, point_cloud, target, output_opt, camera, mse, loss2, loss3, last_output_joints)
            
            loss4 = mse(pose_single,pose_single_rec) 

            loss = loss1 + loss2 * method_cfg['lambda_3d'] + loss3 * method_cfg['lambda_timing'] + loss4 * method_cfg['lambda_pose']
            loss.requires_grad_(True)        
            loss.backward(retain_graph=True)

            optimizer.step()
            if abs(loss - last_loss) < 1e-3:
                log.info('loss_delta almost equal to ZERO!')
                log.info(f'end at {i}, final loss {loss}')
                log.info(f'{loss1}, {loss2}, {loss3}, {loss4}')
                break
            last_loss = loss


        pose_arr.append(pose_single_rec.detach().cpu())
        beta_arr.append(beta_single.detach().cpu())
        trans_arr.append(trans_root.detach().cpu())

        last_output_joints = output_opt.joints
    
    pose_result = torch.from_numpy(fil_pose(np.stack(pose_arr)))
    # pose_result = torch.from_numpy(np.stack(pose_arr))

    beta_result = torch.cat(beta_arr)
    trans_result = torch.from_numpy(fil_trans(np.stack(trans_arr)))
    # trans_result = torch.from_numpy(np.stack(trans_arr))


    save_result = {
        'pose': pose_result,
        'beta': beta_result,
        'trans': trans_result,
    }

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    torch.save(save_result, osp.join(opt_result_path, "opt_result.pth"))
    torch.save(save_result, osp.join(opt_result_path, "opt_result_" + hydra_path.split('\\')[-1].split('.')[0] + ".pth"))

    for frame in tqdm(range(pose_result.shape[0])):
        result = smpl_model(betas=beta_result[frame].unsqueeze(0).type(dtype).to(device),
                            body_pose=pose_result[frame,1:].unsqueeze(0).type(dtype).to(device),
                            global_orient=pose_result[frame,[0]].unsqueeze(0).type(dtype).to(device),
                            pose2rot=False,
                            transl=trans_result[frame].unsqueeze(0).type(dtype).to(device))
    
        visualizer.visual_smpl_2d_single(result.vertices, renderer, frame)
        visualizer.save_o3d_mesh(result.vertices, smpl_model.faces, frame)


if __name__ == "__main__":
    main()