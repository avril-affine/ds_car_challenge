import os
import constants
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K


def normalize_image(img):
    """Normalize image to scale [-1, 1]

    Parameters:
        img     (ndarray): 3-dimensional numpy array
    """
    return (img - constants.mu) / constants.sd


class RandomRastorGenerator(object):
    """Generator for randomly selecting images for datasciencechallenge.org
    safe passage contest.

    Paramters:
        img_dir     (str): Directory containing training images.
        label       (str): Class label.
        label_dir   (str): Directory containing images of labels for each class.
            Image filenames are of the form <image>_<class>.
        batch_size  (int): Size of each batch to return.
        crop_size   (int): Size of the crop for each rastored image.
        n_pos       (int): Number of positive samples per negative samples.
        transformer (obj): ImageDataGenerator object that implements
            random_transform().
    """

    def __init__(self,
                 img_dir,
                 label=None,
                 label_file=None,
                 batch_size=32,
                 crop_size=240,
                 n_pos=4,
                 transformer=None):
        assert os.path.exists(img_dir)
        if label_file:
            assert os.path.exists(label_file)
            assert label is not None
            assert label in constants.classes
        if transformer:
            assert hasattr(transformer, 'random_transform')

        self.img_dir      = img_dir
        self.image_names  = [x for x in os.listdir(self.img_dir) if x.endswith('.jpg')]
        self.label        = label
        self.label_df     = pd.read_json(label_file)
        self.batch_size   = batch_size
        self.crop_size    = crop_size
        self.n_pos        = n_pos
        self.transformer  = transformer
        # self.pos_label_df = self.label_df[self.label_df['points'].apply(lambda x: len(x) > 0)]

    def __len__(self):
        return self.batch_size

    def crop_transform_image(self, img, x, y, do_transform=True):
        new_img = img[x:x+self.crop_size, y:y+self.crop_size]
        if do_transform and self.transformer:
            new_img = self.transformer.random_transform(new_img)
        return new_img

    def next(self):
        batch_x = np.zeros((self.batch_size, self.crop_size, self.crop_size, 3))
        if self.label:
            batch_y = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1))

        count = 0
        while count < self.batch_size:
            image_idx = np.random.randint(0, len(self.image_names))
            img_name = self.image_names[image_idx]
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path)
            img = normalize_image(np.asarray(img, K.floatx()))

            mask = ((self.label_df['class'] == self.label)
                    & (self.label_df['image'] == img_name))
            points = self.label_df[mask]['points'].values[0]
            if self.label:
                label_img = np.zeros((constants.img_size, constants.img_size),
                                          dtype=np.float32)
                for point in points:
                    # images in numpy array rows are y's
                    label_img[point[1], point[0]] = 1

            # get random point
            for _ in xrange(2):
                x = np.random.randint(0, constants.img_size - self.crop_size)
                y = np.random.randint(0, constants.img_size - self.crop_size)
                batch_x[count] = self.crop_transform_image(img, x, y)
                if self.label:
                    batch_y[count,:,:,0] = self.crop_transform_image(label_img, x, y, False)
                count += 1
                if count == self.batch_size:
                    break
            if count == self.batch_size:
                break

            # get 2 random positive label
            if len(points) > 0:
                for _ in xrange(self.n_pos):
                    rand_idx = np.random.randint(0, len(points))
                    x_, y_ = points[rand_idx]
                    radius = constants.class_radius[self.label] / 2
                    x = max(0, x_ - radius)
                    y = max(0, y_ - radius)
                    # if radius does not fit on img subtract from start point
                    x -= max(0, (x_ + radius) - (constants.img_size - self.crop_size))
                    y -= max(0, (y_ + radius) - (constants.img_size - self.crop_size))
                    batch_x[count] = self.crop_transform_image(img, x, y)
                    if self.label:
                        batch_y[count,:,:,0] = self.crop_transform_image(label_img, x, y, False)
                    count += 1
                    if count == self.batch_size:
                        break

        if self.label:
            return batch_x, batch_y
        else:
            return batch_x
