import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from utils import constants


def make_labels(label_file):
    df = pd.read_csv(label_file)

    new_column = []
    for i, row in df.iterrows():
        detections = row['detections']
        radius = constants.class_radius[row['class']]
        labels = set()
        if detections == 'None':
            new_column.append(list(labels))
            continue

        for detection in detections.split('|'):
            x, y = detection.split(':')
            x = int(x)
            y = int(y)

            for x_ in xrange(max(0, x - radius), min(constants.img_size, x + 1)):
                for y_ in xrange(max(0, y - radius), min(constants.img_size, y + 1)):
                    if ((x_ - x)*(x_ - x) + (y - y_)*(y - y_)) <= radius*radius:
                        xSym = x - (x_ - x)
                        ySym = y - (y_ - y)

                        if x_ >= 0 and x_ < constants.img_size and y_ >= 0 and y_ < constants.img_size:
                            labels.add((x_, y_))
                        if x_ >= 0 and x_ < constants.img_size and ySym >= 0 and ySym < constants.img_size:
                            labels.add((x_, ySym))
                        if xSym >= 0 and xSym < constants.img_size and y_ >= 0 and y_ < constants.img_size:
                            labels.add((xSym, y_))
                        if xSym >= 0 and xSym < constants.img_size and ySym >= 0 and ySym < constants.img_size:
                            labels.add((xSym, ySym))
        new_column.append(list(labels))

    df['points'] = new_column
    data_dir = os.path.dirname(label_file)
    df.to_json(os.path.join(data_dir, 'new_train.json'))


def calc_mean_std(data_dir):
    img_dir = os.path.join(data_dir, 'train')

    # calc mean
    acc = np.zeros((constants.img_size, constants.img_size, 3))
    count = 0
    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        acc += np.asarray(img, np.float)
        count += 1
    mu = np.mean(acc / np.array([count] * 3), axis=(0,1))

    # calc std
    acc = np.zeros((constants.img_size, constants.img_size, 3))
    count = 0
    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = np.asarray(img, np.float)
        img = (img - mu) ** 2
        acc += img
        count += 1
    sd = np.sqrt(np.mean(acc / np.array([count] * 3), axis=(0,1)))

    print 'mu:', mu
    print 'std:', sd


def calc_area_ratio(home_dir):
    df = pd.read_json(os.path.join(home_dir, 'data/new_train.json'))
    df['num_points'] = df['detections'].map(lambda x: 0 if x == 'None' else len(x.split('|')))
    df['area_per_point'] = df['class'].map(lambda x: constants.class_radius[x] ** 2)
    df['area_ratio'] = df['area_per_point'] * df['num_points'] / 4000000.
    grouped_df = df.groupby('class')
    print grouped_df['area_ratio'].mean()


def train_val_split(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    else:
        print 'Validation directory exists... not splitting'
        return

    train_imgs = [x for x in os.listdir(train_dir) if x.endswith('.jpg')]

    indices = range(len(train_imgs))
    np.random.shuffle(indices)

    val_indices = indices[:100]
    for i in val_indices:
        img_name = train_imgs[i]
        shutil.move(os.path.join(train_dir, img_name),
                    os.path.join(val_dir, img_name))


def main():
    filename = os.path.abspath(__file__)
    home_dir = os.path.dirname(os.path.dirname(filename))

    label_file = os.path.join(home_dir, 'data/trainingObservations.csv')

    print 'Calculating acceptable pixels...'
    make_labels(label_file)

    print 'Calculating area ratios...'
    calc_area_ratio(home_dir)

    print 'Calculating mean/std...'
    data_dir = os.path.join(home_dir, 'data')
    calc_mean_std(data_dir)

    print 'Splitting train/val set...'
    train_val_split(data_dir)


if __name__ == '__main__':
    main()
