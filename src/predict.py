import os
import sys
import cv2
import tarfile
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from skimage import img_as_ubyte
from keras import backend as K
from keras.models import load_model
from utils.image_processing import normalize_image
from utils import constants


def predict(cls, model_dir, test_dir, crop_size, stride, threshold):
    weights_file = os.path.join(model_dir, cls, 'weights.h5')
    model = load_model(weights_file)

    min_area = (constants.class_radius[cls] / 2) * (constants.class_radius[cls] / 2)

    test_files = [x for x in os.listdir(test_dir) if x.endswith('.jpg')]

    mask = np.ones((crop_size, crop_size, 1))
    all_boxes = []
    for test_file in test_files:
        print test_file
        preds = np.zeros((constants.img_size, constants.img_size, 1))
        count = np.zeros((constants.img_size, constants.img_size, 1))

        test_path = os.path.join(test_dir, test_file)
        test_img = Image.open(test_path)
        test_img = np.asarray(test_img, np.float32)
        test_img = normalize_image(test_img)

        # rastor test image
        for x in xrange(0, constants.img_size - crop_size + 1, stride):
            for y in xrange(0, constants.img_size - crop_size + 1, stride):
                batch_x = np.array([test_img[x:x+crop_size, y:y+crop_size]])
                preds_ = model.predict_on_batch(batch_x)[0]
                print np.max(preds_)

                preds[x:x+crop_size, y:y+crop_size] += preds_
                count[x:x+crop_size, y:y+crop_size] += mask

        # average predictions and take threshold
        preds = (preds / count) > threshold
        preds = img_as_ubyte(preds)
        _, contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find boxes from thresholded prediction
        test_img = Image.open(test_path)
        test_img = np.asarray(test_img, np.float32)
        boxes = []
        for cnt in contours:
            box = cv2.boundingRect(cnt)
            if box[2] * box[3] > min_area:
                boxes.append(box)
                x,y,w,h = box
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow(test_img)
        cv2.waitKey(0)

        all_boxes.append(boxes)
    K.clear_session()
    return all_boxes

def plot_preds(test_dir, all_boxes, archive):
    for test_file, boxes in zip(os.listdir(test_dir), all_boxes):
        # load test image
        test_path = os.path.join(test_dir, test_file)
        test_img = Image.open(test_path)
        test_img = np.asarray(test_img, np.float32)

        # draw predicted boxes
        for box in boxes:
            x,y,w,h = box
            cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), 2)
        preds_dir = os.path.join('/tmp/preds')
        pred_file = os.path.join(preds_dir, test_file)

        # write to file and add to archive
        cv2.imwrite(pred_file, test_img)
        archive.add(pred_file)


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
    # classes = constants.classes
    classes = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
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

    test_files = [x for x in os.listdir(args.test_dir) if x.endswith('.jpg')]
    test_ids = [os.path.splitext(x)[0] for x in test_files]
    ids = []
    detections = []
    for cls in classes:
        print 'predicting class {}...'.format(cls)
        all_boxes = predict(cls, model_dir, args.test_dir, args.crop_size,
                            args.stride, args.threshold)
        # if archive:
        #     plot_preds(args.test_dir, all_boxes, archive)
        preds = convert_boxes(all_boxes)
        test_ids = [x + '_' + cls for x in test_ids]
        ids.extend(test_ids)
        detections.extend(preds)
    archive.close()

    print 'writing submission csv'
    df = pd.DataFrame({'id': ids, 'detections': detections})
    df.to_csv(os.path.join(model_dir, 'submission.csv'), index=False)


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
