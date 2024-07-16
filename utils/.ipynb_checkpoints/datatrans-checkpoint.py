#   Required transforms:
#       resize
#       random horizon flip
#       color jitter
#       random affine
#       to tensor

import torch 
from torchvision.transforms import v2   
import numpy as np
import math
#from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
import random
import torchvision.transforms.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)

        return img, bboxes
    
class Resize(object):
    def __init__(self, size=(416, 416)):
        self.size = size

    def __call__(self, img, bboxes):
        img = v2.Resize(self.size)(img)
        return img, bboxes

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = v2.RandomHorizontalFlip(p=1)(img)
            bboxes_rotate = []
            for bbox in bboxes:
                bbox[0] = 1 - bbox[0]
                bbox[1] = bbox[1]
                bbox[2] = bbox[2]
                bbox[3] = bbox[3]
                bboxes_rotate.append(bbox)
            bboxes = bboxes_rotate

        return img, bboxes
    
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = v2.RandomVerticalFlip(p=1)(img)
            bboxes_rotate = []
            for bbox in bboxes:
                bbox[0] = bbox[0]
                bbox[1] = 1 - bbox[1]
                bbox[2] = bbox[2]
                bbox[3] = bbox[3]
                bboxes_rotate.append(bbox)
            bboxes = bboxes_rotate

        return img, bboxes


    
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = v2.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)(img)
        return img, bboxes

class GaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = v2.GaussianBlur(kernel_size=5, sigma=5)(img)
        return img, bboxes

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, img, bboxes):
        img = v2.Normalize(mean=self.mean, std=self.std)(img)

        return img, bboxes


class toTensor(object):
    def __call__(self, img, bboxes):
        # use v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
        img = v2.ToImage()(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        return img, bboxes
    

class RandomRotation(object):
    def __init__(self, degrees=0, p=0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = F.rotate(img, angle, expand=False, center=None, fill=0)
            angle = -45
            w, h = img.size
            print(angle)
            image_center = (w / 2, h / 2)
            bboxes_rotated = []
            for bbox in bboxes:
                x_center = bbox[0]
                y_center = bbox[1]
                width = bbox[2]
                height = bbox[3]
                label = bbox[4]
                
                # Convert center coordinates to corner coordinates
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2

                # Rotate bounding box
                x_min_rot, y_min_rot, x_max_rot, y_max_rot = calculate_rotated_bounding_box(
                    x_min, y_min, x_max, y_max, np.radians(angle), image_center, w, h)

                # Convert back to center coordinates
                x_center_rot = (x_min_rot + x_max_rot) / 2
                y_center_rot = (y_min_rot + y_max_rot) / 2
                width_rot = x_max_rot - x_min_rot
                height_rot = y_max_rot - y_min_rot

                bboxes_rotated.append([x_center_rot, y_center_rot, width_rot, height_rot, label])
            
            bboxes = bboxes_rotated
            print("Rotated bounding boxes:", bboxes)
        return img, bboxes

def calculate_rotated_bounding_box(x_min, y_min, x_max, y_max, angle, image_center, img_width, img_height):
    """
    Rotate the bounding box coordinates.
    Args:
    - x_min, y_min, x_max, y_max: Denormalized coordinates of the bounding box.
    - angle: Rotation angle in radians.
    - image_center: The center of the image (cx, cy) in pixel coordinates.
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - x_min_rot, y_min_rot, x_max_rot, y_max_rot: Rotated bounding box coordinates in normalized form.
    """
    # Convert to corner points
    corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])

    # Calculate rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Translate corners to origin
    translated_corners = corners - image_center

    # Rotate corners
    rotated_corners = np.dot(translated_corners, rotation_matrix.T)

    # Translate corners back
    rotated_corners += image_center

    # Get new bounding box coordinates
    x_min_rot = np.clip(np.min(rotated_corners[:, 0]) / img_width, 0, 1)
    y_min_rot = np.clip(np.min(rotated_corners[:, 1]) / img_height, 0, 1)
    x_max_rot = np.clip(np.max(rotated_corners[:, 0]) / img_width, 0, 1)
    y_max_rot = np.clip(np.max(rotated_corners[:, 1]) / img_height, 0, 1)

    return x_min_rot, y_min_rot, x_max_rot, y_max_rot