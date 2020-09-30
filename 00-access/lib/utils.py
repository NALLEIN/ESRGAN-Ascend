import logging
import os
import glob
import moxing as mox

import cv2
import numpy as np


def log(logflag, message, level='info'):
    """logging to stdout and logfile if flag is true"""
    print(message, flush=True)

    if logflag:
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        elif level == 'critical':
            logging.critical(message)


def create_dirs(target_dirs):
    """create necessary directories to save output files"""
    for dir_path in target_dirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


def normalize_images(*arrays):
    """normalize input image arrays"""
    return [arr / 127.5 - 1 for arr in arrays]


def de_normalize_image(image):
    """de-normalize input image array"""
    return (image + 1) * 127.5


def save_image(FLAGS, images, phase, global_iter, save_max_num=5):
    """save images in specified directory"""
    if phase == 'train' or phase == 'pre-train':
        save_dir = FLAGS.train_url
    elif phase == 'inference':
        save_dir = FLAGS.inference_result_dir
        save_max_num = len(images)
    else:
        print('specified phase is invalid')

    for i, img in enumerate(images):
        if i >= save_max_num:
            break

        cv2.imwrite(save_dir + '/{0}_HR_{1}_{2}.jpg'.format(phase, global_iter, i), de_normalize_image(img))


def crop(img, FLAGS):
    """crop patch from an image with specified size"""
    img_h, img_w, _ = img.shape

    rand_h = np.random.randint(img_h - FLAGS.crop_size)
    rand_w = np.random.randint(img_w - FLAGS.crop_size)

    return img[rand_h:rand_h + FLAGS.crop_size, rand_w:rand_w + FLAGS.crop_size, :]


def data_augmentation(LR_images, HR_images, aug_type='horizontal_flip'):
    """data augmentation. input arrays should be [N, H, W, C]"""

    if aug_type == 'horizontal_flip':
        return LR_images[:, :, ::-1, :], HR_images[:, :, ::-1, :]
    elif aug_type == 'rotation_90':
        return np.rot90(LR_images, k=1, axes=(1, 2)), np.rot90(HR_images, k=1, axes=(1, 2))


def load_npz_data(FLAGS):
    """load array data from data_path"""
    print('obs path :', FLAGS.data_url)
    mox.file.copy_parallel(FLAGS.data_url, FLAGS.native_data)
    print("mox copy files finished")
    return np.load(FLAGS.native_data + '/' + FLAGS.HR_npz_filename)['images'], \
           np.load(FLAGS.native_data + '/' + FLAGS.LR_npz_filename)['images']
