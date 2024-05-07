import numpy as np
import cv2
import os.path as osp
import os

from tqdm import tqdm
from icecream import ic

def is_contact(_contact):
    """Whether the foot is in contact with the ground? if yes, return True, else return False.
    """
    return any([_contact[0], _contact[1], _contact[2], _contact[3]])

def find_contact_change_idx(contact):
    contact_change_idx_all = [[0,is_contact(contact[0])]]
    for i in range(2,contact.shape[0]-1):
        last_contact_flag = is_contact(contact[i-1])
        now_contact_flag = is_contact(contact[i])
        if last_contact_flag != now_contact_flag:
            contact_change_idx_all.append([i-1,last_contact_flag])
            contact_change_idx_all.append([i,now_contact_flag])
    contact_change_idx_all.append([contact.shape[0]-1,is_contact(contact[contact.shape[0]-1])])
    return contact_change_idx_all
