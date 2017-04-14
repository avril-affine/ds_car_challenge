import os
import argparse
import tensorflow as tf
from models import unet
from losses import jaccard
from keras.optimizers import Adam
from keras import backend as K
from utils.rastor import RastorGenerator


def get_summary(name, mdl, generator):
    loss = []
    for _ in xrange(len(generator)):
        batch_x, batch_y = generator.next()
        loss.append(mdl.test_on_batch(batch_x, batch_y))

    loss = 1. * sum(loss) / len(loss)
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    return loss, summary


def main(args):
    optimizer = Adam(LR)
    mdl = unet()
    mdl.compile(optimizer, jaccard)

    print mdl.metrics_names

    train_generator = RastorGenerator(args.train_dir,
                                      label=args.label,
                                      label_dir=args.label_dir,
                                      batch_size=args.batch_size,
                                      crop_size=args.crop_size,
                                      stride=args.stride)
    train_generator = RastorGenerator(args.val_dir,
                                      label=args.label,
                                      label_dir=args.label_dir,
                                      batch_size=args.batch_size,
                                      crop_size=args.crop_size,
                                      stride=args.stride)

    sess = K.get_session()
    writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'logs'), sess.graph)

    best_loss = None
    for epoch in xrange(NUM_EPOCHS):
        # train
        for _ in xrange(len(train_generator)):
            batch_x, batch_y = train_generator.next()
            mdl.train_on_batch(batch_x, batch_y)

        # write train/val loss summary
        train_loss, train_summary = get_summary('train_loss', mdl, train_generator)
        val_loss, val_summary     = get_summary('val_loss', mdl, val_generator)
        writer.add_summary(train_summary, epoch)
        writer.add_summary(val_summary, epoch)

        if best_loss and (val_loss < best_loss):
            best_loss = val_loss
            mdl.save(os.path.join(args.model_dir, 'weights.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a keras unet model for safe passage contest'.)
    parser.add_argument('--model_dir', required=True,
        help='Directory to save model weights and logs to.')
    parser.add_argument('--train_dir', required=True,
        help='Directory for images to train on.')
    parser.add_argument('--val_dir', required=True,
        help='Directory for images to validate on.')
    parser.add_argument('--label_dir', required=True,
        help='Directory for labels of each image.')
    parser.add_argument('--label', required=True,
        help='Which label to train on.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
        help='Learning rate for gradient descent.')
    parser.add_argument('--batch_size', type=int, default=55,
        help='Size of each batch for training.')
    parser.add_argument('--num_epochs', type=int, default=100,
        help='Number of epochs to run model.')
    parser.add_argument('--crop_size', type=int, default=240,
        help='Size for each rastored image.')
    parser.add_argument('--rastor_stride', type=int, default=32,
        help='Stride to rastor image.')
    args = parser.parse_args()

    main(args)
