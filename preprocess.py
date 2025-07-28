# preprocess.py
import os
import cv2
import numpy as np

def grabcut_preprocess(image_path, save_path=None):

    try:
        img = cv2.imread(image_path)
        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (5, 5, img.shape[1] - 10, img.shape[0] - 10)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        white_background = np.full_like(img, 255, dtype=np.uint8)
        output_img = np.where(mask2[:, :, np.newaxis] == 1, img, white_background)

        if save_path:
            cv2.imwrite(save_path, output_img)

        return output_img

    except Exception as e:
        print(f"[GrabCut Error] {image_path}: {e}")
        return None
