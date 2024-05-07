import torch

OPENPOSE =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
HALPE =   [0,18,6,8,10,5,7,9,19,12,14,16,11,13,15,2,1,4,3,20,22,24,21,23,25]
HALPE_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle',
    'head','neck','hip','left_big_toe','right_big_toe',
    'left_small_toe','right_small_toe','left_heel','right_heel'
]

LIMBS = [2,3,4,5,6,7,9,10,11,12,13,14]
FEET = [19,20,21,22,23,24]
OTHERS = [0,1,8,15,16,17,18]


def loss_2d(source, target, score, lambda_limbs, lambda_feet, lambda_others):
    loss_2d_limbs = torch.mean(torch.sqrt(score[:,LIMBS]) * torch.norm(source[:,LIMBS,:] - target[:,LIMBS,:], dim=-1))
    loss_2d_feet = torch.mean(torch.sqrt(score[:,FEET]) * torch.norm(source[:,FEET,:] - target[:,FEET,:], dim=-1))
    loss_2d_others = torch.mean(torch.sqrt(score[:,OTHERS]) * torch.norm(source[:,OTHERS,:] - target[:,OTHERS,:], dim=-1))

    loss_2d = lambda_limbs * loss_2d_limbs + lambda_feet * loss_2d_feet + lambda_others * loss_2d_others
    return loss_2d

def loss_3d_t(contact_single, point_cloud, target, output_opt, camera, mse, loss2, loss3, last_output_joints):
    if contact_single[0] == 1:
        ground_LBT = point_cloud[int(target[:,19,1]) if int(target[:,19,1]) < camera.img_h else camera.img_h-1,int(target[:,19,0]) if int(target[:,19,0]) < camera.img_w else camera.img_w-1].unsqueeze(0)
        ground_LST = point_cloud[int(target[:,20,1]) if int(target[:,20,1]) < camera.img_h else camera.img_h-1,int(target[:,20,0]) if int(target[:,20,0]) < camera.img_w else camera.img_w-1].unsqueeze(0)
        loss2 = loss2.clone() + mse(output_opt.joints[:,19], ground_LBT)
        loss2 = loss2.clone() + mse(output_opt.joints[:,20], ground_LST)
        if last_output_joints is not None:
            loss3 = loss3.clone() + mse(output_opt.joints[:,19], last_output_joints[:,19])
            loss3 = loss3.clone() + mse(output_opt.joints[:,20], last_output_joints[:,20])
    if contact_single[1] == 1:
        ground_LH = point_cloud[int(target[:,21,1]) if int(target[:,21,1]) < camera.img_h else camera.img_h-1,int(target[:,21,0]) if int(target[:,21,0]) < camera.img_w else camera.img_w-1].unsqueeze(0)
        loss2 = loss2.clone() + mse(output_opt.joints[:,21], ground_LH)
        if last_output_joints is not None:
            loss3 = loss3.clone() + mse(output_opt.joints[:,21], last_output_joints[:,21])
    if contact_single[2] == 1:
        ground_RBT = point_cloud[int(target[:,22,1]) if int(target[:,22,1]) < camera.img_h else camera.img_h-1,int(target[:,22,0]) if int(target[:,22,0]) < camera.img_w else camera.img_w-1].unsqueeze(0)
        ground_RST = point_cloud[int(target[:,23,1]) if int(target[:,23,1]) < camera.img_h else camera.img_h-1,int(target[:,23,0]) if int(target[:,23,0]) < camera.img_w else camera.img_w-1].unsqueeze(0)
        loss2 = loss2.clone() + mse(output_opt.joints[:,22], ground_RBT)
        loss2 = loss2.clone() + mse(output_opt.joints[:,23], ground_RST)
        if last_output_joints is not None:
            loss3 = loss3.clone() + mse(output_opt.joints[:,22], last_output_joints[:,22])
            loss3 = loss3.clone() + mse(output_opt.joints[:,23], last_output_joints[:,23])
    if contact_single[3] == 1:
        ground_RH = point_cloud[int(target[:,24,1]) if int(target[:,24,1]) < camera.img_h else camera.img_h-1,int(target[:,24,0]) if int(target[:,24,0]) < camera.img_w else camera.img_w-1].unsqueeze(0)
        loss2 = loss2.clone() + mse(output_opt.joints[:,24], ground_RH)
        if last_output_joints is not None:
            loss3 = loss3.clone() + mse(output_opt.joints[:,24], last_output_joints[:,24])\
            
    return loss2, loss3