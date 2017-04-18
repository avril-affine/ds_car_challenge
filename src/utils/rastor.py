import os
import numpy as np
import pandas as pd
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

    def __init__(self, img_dir, label=None, label_file=None, batch_size=55,
                 crop_size=240, stride=32):
        assert ((RastorGenerator.img_size - crop_size) % stride) == 0
        assert ((((RastorGenerator.img_size - crop_size) / stride) ** 2) % batch_size) == 0
        assert os.path.exists(img_dir)
        assert label in RastorGenerator.classes
        if label_file:
            assert os.path.exists(label_file)
            assert label is not None

        self.img_dir     = img_dir
        self.image_names = [x for x in os.listdir(self.img_dir) if x.endswith('.jpg')]
        self.label       = label
        self.label_df    = pd.read_json(label_file)
        self.batch_size  = batch_size
        self.crop_size   = crop_size
        self.stride      = stride
        self.image_idx   = 0
        self.x           = 0
        self.y           = 0
        self.img         = None
        self.label_img   = None

    def __len__(self):
        crops_per_img = (RastorGenerator.img_size - self.crop_size) / self.stride
        crops_per_img *= crops_per_img
        return len(self.image_names) * crops_per_img / self.batch_size

    def next(self):
        if self.x == 0 and self.y == 0:
            img_name = self.image_names[self.image_idx]
            img_path = os.path.join(self.img_dir, img_name)
            self.img = Image.open(img_path)
            self.img = np.asarray(self.img, K.floatx())

            if self.label:
                self.label_img = np.zeros((RastorGenerator.img_size, RastorGenerator.img_size),
                                          dtype=np.float32)
                mask = ((self.label_df['class'] == self.label)
                        & (self.label_df['image'] == img_name))
                points = self.label_df[mask]['points'].values[0]
                for point in points:
                    # images in numpy array rows are y's
                    self.label_img[point[1], point[0]] = 1

        batch_x = np.zeros((self.batch_size, self.crop_size, self.crop_size, 3))
        if self.label:
            batch_y = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1))

        count = 0
        for x in xrange(self.x, self.img.shape[0] - self.crop_size + 1, self.stride):
            for y in xrange(self.y, self.img.shape[1] - self.crop_size + 1, self.stride):
                batch_x[count] = self.img[x:x+self.crop_size, y:y+self.crop_size]
                if self.label:
                    batch_y[count,:,:,0] = self.label_img[x:x+self.crop_size, y:y+self.crop_size]
                count += 1
                if count == self.batch_size:
                    # normalize x by channels
                    mu = np.zeros(3)
                    sd = np.zeros(3)
                    for i in xrange(3):
                        mu[i] = np.mean(batch_x[:,:,:,i])
                        sd[i] = np.std(batch_x[:,:,:,i])
                    batch_x = (batch_x - mu) / sd

                    # keep track of where we are in rastoring the image
                    self.x = (x + self.stride) % (self.img.shape[0] - self.crop_size)
                    self.y = (y + self.stride) % (self.img.shape[1] - self.crop_size)

                    # if both x and y are done then move to next image
                    if self.x == 0 and self.y == 0:
                        self.image_idx = (self.image_idx + 1) % len(self.image_names)

                    if self.label:
                        return batch_x, batch_y
                    else:
                        return batch_x
