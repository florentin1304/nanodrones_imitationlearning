import cv2 as cv
import numpy as np
import torch

class PencilFilter:
    def __init__(self,
                    dilatation_size = 2,
                    dilation_shape = cv.MORPH_ELLIPSE):
        self.dilatation_size = dilatation_size
        self.dilation_shape = dilation_shape #cv.MORPH_ELLIPSE cv.MORPH_RECT cv.MORPH_CROSS

    def apply(self, img):
        if (img.shape[0] != 3 and img.shape[2] != 3):
            raise ValueError(f"Invalid image format. Expected 3 channels, instead image shape is {img.shape}")

        # If channels are not in the last dimension, transpose the image
        if len(img.shape) == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)
        img = img*255

        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        dialted = self.dilatation(gray_img.copy())


        gray_img_32_bit = (gray_img.copy()).astype(np.uint32)
        dialted_my = ((gray_img_32_bit * 255)/dialted).astype(np.uint8)
        
        penciled = np.where(np.isnan(dialted_my), 255, dialted_my).astype(np.uint8)
        penciled = np.expand_dims(penciled, axis=0)
        
        
        return torch.Tensor(penciled) / 255

    def dilatation(self, img):
        element = cv.getStructuringElement(self.dilation_shape, (2 * self.dilatation_size + 1, 2 * self.dilatation_size + 1),
                                        (self.dilatation_size, self.dilatation_size))
        dilatation_dst = cv.dilate(img, element)
        return dilatation_dst