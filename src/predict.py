import os
import sys
import cv2
import tarfile
import argparse
from PIL import Image
from keras.models import load_model
from utils.rastor import RastorGenerator


def predict(weights_file, test_dir, label_file, crop_size, stride, threshold, archive=None):
    mdl = load_model(weights_file)

    test_generator = RastorGenerator(test_dir,
                                     batch_size=1,
                                     crop_size=crop_size,
                                     stride=stride)

    test_files = [x for x in os.listdir(test_dir) if x.endswith('.jpg')]
    rastors_per_img = len(test_generator) / test_size

    mask = np.ones((crop_size, crop_size))
    all_boxes = []
    for test_file in test_files:
        preds = np.zeros((2000, 2000))
        count = np.zeros((2000, 2000))
        for _ in xrange(rastors_per_img):
            x = test_generator.x
            y = test_generator.y

            batch_x = test_generator.next()
            preds_ = model.predict_on_batch(batch_x)[0]

            preds[x:x+crop_size, y:y+crop_size] += preds_
            count[x:x+crop_size, y:y+crop_size] += mask

        preds = (preds / count) > threshold   # average and threshold predictions
        countours, _ = cv2.findContours(preds, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in countours:
            box = cv2.boundingRect(cnt)
            boxes.append(box)
        all_boxes.append(boxes)

        test_path = os.path.join(test_dir, test_file)
        test_img = Image.open(test_path)  # testing
        plot_preds(test_img, boxes)
        cv2.imshow('img', test_img) # testing
        if archive:
            preds_dir = os.path.join('/tmp/preds')
            pred_file = os.path.join(preds_dir, os.listdir(test_dir)[i])
            cv2.imwrite(pred_file, test_img)
            archive.add(pred_file)
        break   # testing
    return convert_boxes(all_boxes)


def plot_preds(img, boxes):
    for box in boxes:
        x,y,w,h = box
        cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), 2)


def convert_boxes(all_boxes):
    """Converts a list of boxes to string for submission

    Parameters:
        all_boxes: list of list of boxes with tuple (x,y,w,h)
    Returns:
        list of strings.
    """
    res = []
    for boxes in all_boxes:
        centers = []
        for box in boxes:
            x,y,w,h = box
            centers.append('{}:{}'.format(x + w / 2, y + h / 2))
        centers = centers if len(centers) > 0 else ['None']
        res.append('|'.join(centers))
    return res


def main(args):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    model_dir = args.model_dir

    # check if everything exists
    if not os.path.exists(model_dir):
        print 'Model directory: {} does not exist.'.format(model_dir)
        sys.exit(1)
    for cls in classes:
        class_dir = os.path.join(model_dir, cls)
        if not os.path.exists(class_dir):
            print 'Missing class directory: ' + class_dir
            sys.exit(1)
        weights_file = os.path.join(class_dir, 'weights.h5')
        if not os.path.exists(weights_file):
            print 'Missing weights file for class: ' + class_dir
            sys.exit(1)

    if args.archive_file:
        archive = tarfile.open(args.archive_file, 'w:gz')
    else:
        archive = None
    for cls in classes:
        weights_file = os.path.join(model_dir, cls, 'weights.h5')
        predict(weights_file, args.test_dir, args.label_file, args.crop_size,
                args.stride, args.threshold, archive=archive)
    archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get predictions for a unet model and return center points.')
    parser.add_argument('--model_dir', required=True,
        help='Directory containing folders for each of 9 models (A-I).')
    parser.add_argument('--test_dir', required=True,
        help='Directory containing images to predict.')
    parser.add_argument('--label_file', default=None,
        help='Specify to also score predictions.')
    parser.add_argument('--archive_file', default=None,
        help='If specified will write predictions to file and gzip.')
    parser.add_argument('--crop_size', type=int, default=240,
        help='Size of rastor image.')
    parser.add_argument('--stride', type=int, default=110,
        help='Stride to rastor image.')
    parser.add_argument('--threshold', type=float, default=0.5,
        help='Threshold to predict 0 or 1.')
    args = parser.parse_args()

    main(args)
