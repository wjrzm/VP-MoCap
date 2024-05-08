import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random
import copy

class ImageModule():
    # Upper left corner (0, 0)
    # option: aug & input_size
    def __init__(self, opt):
        self.opt = opt

    def get_transform(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def transform(self, pt, center, scale, res, invert=0, rot=0):
        """Transform pixel location to different reference."""
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            # t = np.linalg.inv(t)
            t_torch = torch.from_numpy(t)
            t_torch = torch.inverse(t_torch)
            t = t_torch.numpy()
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int) + 1

    def myimrotate(self, img, angle, center=None, scale=1.0, border_value=0, auto_bound=False):
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
        return rotated

    def myimresize(self, img, size, return_scale=False, interpolation='bilinear'):

        h, w = img.shape[:2]
        resized_img = cv2.resize(
            img, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale

    def crop(self, img, center, scale, res, rot=0):
        """Crop image according to the supplied bounding box."""
        # Upper left point
        ul = np.array(self.transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(self.transform([res[0] + 1,
                                      res[1] + 1], center, scale, res, invert=1)) - 1
        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        if not rot == 0:
            ul -= pad
            br += pad
        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])

        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]
        if not rot == 0:
            # Remove padding
            # new_img = scipy.misc.imrotate(new_img, rot)
            new_img = self.myimrotate(new_img, rot)
            new_img = new_img[pad:-pad, pad:-pad]

        # new_img = scipy.misc.imresize(new_img, res)
        new_img = self.myimresize(new_img, [res[0], res[1]])
        return new_img

    def flip_img(self, img):
        """Flip rgb images or masks.
        channels come last, e.g. (256,256,3).
        """
        img = np.fliplr(img)
        return img

    def pad_image(self, img, sc, res):
        H, W, _ = img.shape

        new_img = np.ones([res, res, 3]) * 128

        _h = int(res * sc)
        _w = int(res * sc)
        if H > W:
            _w = int((W / (H / res)) * sc)
        elif H < W:
            _h = int((H / (W / res)) * sc)

        img = cv2.resize(img, (_w, _h))

        sx = max(int((res - _w) / 2) - 1, 0)
        sy = max(int((res - _h) / 2) - 1, 0)

        new_img[sy:(sy + _h), sx:(sx + _w), :] = img

        return new_img
