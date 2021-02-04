import cv2
import numpy as np
import matplotlib.pyplot as plt

class Mask:

    def __init__(self, im_path, min_percent_mask=0.2, max_percent_mask=0.3):
        self.im_path = im_path
        self.img = cv2.imread(im_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.MIN_PERCENT_MASK = 0.2
        self.MAX_PERCENT_MASK = 0.3

    def _rnd_mul(self, a, b):
        return int(round(a * b))

    def _random_mask_image(self):
        height = self.img.shape[0]
        width = self.img.shape[1]
        
        mask_height = np.random.randint(self._rnd_mul(self.MIN_PERCENT_MASK, height), 
                                        self._rnd_mul(self.MAX_PERCENT_MASK, width))
        mask_width = np.random.randint(self._rnd_mul(self.MIN_PERCENT_MASK, width), 
                                       self._rnd_mul(self.MAX_PERCENT_MASK, width))
        
        mask = np.zeros((mask_height, mask_width, 3))
        
        start_position_height = np.random.randint(0, height - mask_height)
        start_position_width = np.random.randint(0, width - mask_width)
        
        end_position_height = start_position_height + mask_height
        end_position_width = start_position_width + mask_width
        
        masked_img = np.copy(self.img)
        masked_img[start_position_height:end_position_height, start_position_width:end_position_width, :] = mask
        
        return masked_img, mask, start_position_height, start_position_width

    def generate_masked_images(self, num_images=10):
        masked_images = []
        mask_shapes = []
        start_position_heights = []
        start_position_widths = []
        i = 0
        j = 0
        TOTAL_MASKED_IMGS = num_images

        while i < TOTAL_MASKED_IMGS:
            masked_img, mask, start_position_height, start_position_width = self._random_mask_image()
            
            mask_shape = mask.shape
            
            j += 1
            
            if mask_shape in mask_shapes and start_position_height in start_position_heights and start_position_width in start_position_widths:
                continue
            else:
                masked_images.append(masked_img)
                mask_shapes.append(mask_shape)
                start_position_heights.append(start_position_height)
                start_position_widths.append(start_position_width)
                
                i += 1

        return masked_images

    def show_image(self, img, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

    def show_list_images(self, img_list, figsize=(10, 8)):
        for img in img_list:
            self.show_image(img, figsize)