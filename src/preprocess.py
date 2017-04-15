import os
import shutil
import numpy as np
import pandas as pd


IMG_SIZE = 2000
CLASS_RADIUS = {'A': 12,
                'B': 30,
                'C': 30,
                'D': 30,
                'E': 30,
                'F': 30,
                'G': 30,
                'H': 40,
                'I': 45}


def make_labels(label_file):
    df = pd.read_csv(label_file)

    new_column = []
    for i, row in df.iterrows():
        detections = row['detections']
        radius = CLASS_RADIUS[row['class']]
        labels = set()
        if detections == 'None':
            new_column.append(list(labels))
            continue

        for detection in detections.split('|'):
            x, y = detection.split(':')
            x = int(x)
            y = int(y)

            for x_ in xrange(max(0, x - radius), x + 1):
                for y_ in xrange(max(0, y - radius), y + 1):
                    if ((x_ - x)*(x_ - x) + (y - y_)*(y - y_)) <= radius*radius:
                        xSym = x - (x_ - x)
                        ySym = y - (y_ - y)

                        labels.add((x_, y_))
                        labels.add((x_, ySym))
                        labels.add((xSym, y_))
                        labels.add((xSym, ySym))
        new_column.append(list(labels))

    df['points'] = new_column
    data_dir = os.path.dirname(label_file)
    df.to_json(os.path.join(data_dir, 'new_train.json'))


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

    print 'Splitting train/val set...'
    data_dir = os.path.join(home_dir, 'data')
    train_val_split(data_dir)


if __name__ == '__main__':
    main()