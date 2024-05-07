import torch
import torchgeometry as tgm


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    rotation = rotation.repeat(batch_size,1,1)
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)

    # ATTENTION: the line shoule be commented out as the points have been aligned
    # points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def encode(pred_rotmat,vp,device):
    rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
    rot_pad = rot_pad.expand(pred_rotmat.shape[0] * 24, -1, -1)
    rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad), dim=-1)
    pred_rotvec = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)  # N*72
    pred_rotvec_body = pred_rotvec[:,3:66]

    pred_body_poZ = vp.encode(pred_rotvec_body).mean

    return pred_body_poZ

def decode(pred_body_poZ,vp,device,pred_rotmat):
    pred_rotmat_0 = pred_rotmat[:,[0]]
    
    pred_rotmat_wrist = torch.repeat_interleave(torch.unsqueeze(torch.stack([torch.eye(3,device=device),torch.eye(3,device=device)],dim=0),dim=0),pred_rotmat.shape[0],dim=0)
    pred_rotmat_hand = torch.repeat_interleave(torch.unsqueeze(torch.stack([torch.eye(3,device=device),torch.eye(3,device=device)],dim=0),dim=0),pred_rotmat.shape[0],dim=0)
    
    pred_rotmat_body_pose_rec = vp.decode(pred_body_poZ)['pose_body_matrot'].reshape(-1,21,3,3)

    # wrist rotmat equal to 3*3 eye matrix
    pred_rotmat_body_pose_rec[:,19:21] = pred_rotmat_wrist
    
    pred_rotmat_rec = torch.cat((pred_rotmat_0,pred_rotmat_body_pose_rec,pred_rotmat_hand),dim=1)

    return pred_rotmat_rec