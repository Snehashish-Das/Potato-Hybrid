# ai_agent.py

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

class PotatoLeafAIAgent:
    def __init__(self, scaler=None, class_map=None, img_size=(224, 224)):
        self.scaler = scaler
        self.class_map = class_map
        self.img_size = img_size

    def edge_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea) if contours else None
        area = cv2.contourArea(cnt) if cnt is not None else 0
        perimeter = cv2.arcLength(cnt, True) if cnt is not None else 0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])
        mean_val = np.mean(hsv[:, :, 2])

        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, 'uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]

        features = [area, perimeter, mean_hue, mean_sat, mean_val, contrast, homogeneity, energy]
        features = np.concatenate([features, hist])
        return features

    def preprocess_image(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, self.img_size)
        edge = self.edge_map(image)
        features = self.extract_features(image)
        img_norm = image.astype('float32') / 255.0
        edge_norm = np.expand_dims(edge, axis=-1) / 255.0
        return img_norm, features, edge_norm

    def prepare_single_sample(self, path):
        img, feat, edg = self.preprocess_image(path)
        combined_img = np.concatenate([img, edg], axis=-1)
        if self.scaler is not None:
            feat_scaled = self.scaler.transform([feat])
        else:
            feat_scaled = np.array([feat])
        return np.expand_dims(combined_img, axis=0), feat_scaled
