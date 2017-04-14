import os
import numpy as np
from PIL import Image
from keras import backend as K


class RastorGenerator(object):
    """Generator for rastoring images for the datasciencechallenge.org
    safe passage contest.

    Paramters:
        img_dir   (str): Directory containing training images.
        label_dir (str): Directory containing images of labels for each class.
            Image filenames are of the form <image>_<class>.
        batch_size(int): Size of each batch to return.
        crop_size (int): Size of the crop for each rastored image.
        stride    (int): Stride to rastor the full image.
    """
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    img_size = 2000

    def __init__(self, img_dir, label=None, label_dir=None, batch_size=55,
                 crop_size=240, stride=32):
        assert ((RastorGenerator.img_size - crop_size) % stride) == 0
        assert ((((RastorGenerator.img_size - crop_size) / stride) ** 2) % batch_size) == 0
        assert os.path.exists(img_dir)
        assert label in RastorGenerator.classes
        if label_dir:
            assert os.path.exists(label_dir)
            assert label is not None

        self.img_dir     = img_dir
        self.image_names = [x for x in os.listdir(self.img_dir) if not x.endswith('.jpg')]
        self.label       = label
        self.label_dir   = label_dir
        self.batch_size  = batch_size
        self.crop_size   = crop_size
        self.stride      = stride
        self.image_idx   = 0
        self.x           = 0
        self.y           = 0

    def __len__(self):
        crops_per_img = (RastorGenerator.img_size - crop_size) / stride
        crops_per_img *= crops_per_img
        return len(self.image_names) * (crops_per_img / self.batch_size)

    def next(self):
        img_name = self.image_names[self.image_idx]
        img_name = os.path.join(self.img_dir, img_name)
        img = Image.open(img_name)
        img = np.assarray(img, K.floatx())

        if self.label:
            img_basename = os.path.splitext(os.path.basename(img_name))
            label_name = img_basename + '_' + self.label + '.npy'
            label_name = os.path.join(self.label_dir, label_name)
            label = np.load(label_name)

        batch_x = np.zeros((self.batch_size,) + img.shape)
        if self.label:
            batch_y = np.zeros((self.batch_size,) + label.shape)

        count = 0
        for x in xrange(self.x, img.shape[0], stride):
            for y in xrange(self.y, img.shape[1], stride):
                batch_x[i] = img[x:x+stride, y:y+stride]
                if self.label:
                    batch_y[i] = label[x:x+stride, y:y+stride]
                count += 1
                if count == self.batch_size:
                    break

        # keep track of where we are in rastoring the image
        self.x = (x + stride) % img.shape[0]
        self.y = (y + stride) % img.shape[1]

        # if both x and y are done then move to next image
        if self.x == 0 and self.y == 0:
            self.image_idx = (self.image_idx + 1) % len(self.image_names)

        if self.label:
            return batch_x, batch_y
        else:
            return batch_x
