import os
import argparse
import models
import tensorflow as tf
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    optimizer = Adam(args.learning_rate)
    mdl = models.small_unet()
    mdl.compile(optimizer, jaccard)

    train_generator = RastorGenerator(args.train_dir,
                                      label=args.label,
                                      label_file=args.label_file,
                                      batch_size=args.batch_size,
                                      crop_size=args.crop_size,
                                      stride=args.rastor_stride)
    val_generator = RastorGenerator(args.val_dir,
                                    label=args.label,
                                    label_file=args.label_file,
                                    batch_size=args.batch_size,
                                    crop_size=args.crop_size,
                                    stride=args.rastor_stride)
    print 'train_gen size: {}, val_gen size: {}'.format(len(train_generator), len(val_generator))

    sess = K.get_session()
    writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'logs'), sess.graph)

    best_loss = None
    for epoch in xrange(args.num_epochs):
        # train
        for i in xrange(len(train_generator)):
            batch_x, batch_y = train_generator.next()
            loss = mdl.train_on_batch(batch_x, batch_y)
            if i % 25:
                print 'Loss = {}'.format(loss)

        # write train/val loss summary
        train_loss, train_summary = get_summary('train_loss', mdl, train_generator)
        val_loss, val_summary     = get_summary('val_loss', mdl, val_generator)
        writer.add_summary(train_summary, epoch)
        writer.add_summary(val_summary, epoch)

        print 'Train_loss: {}, Val_loss: {}'.format(train_loss, val_loss)

        if best_loss and (val_loss < best_loss):
            print 'New best validation loss. Saving...'
            best_loss = val_loss
            mdl.save(os.path.join(args.model_dir, 'weights.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a keras unet model for safe passage contest.')
    parser.add_argument('--model_dir', required=True,
        help='Directory to save model weights and logs to.')
    parser.add_argument('--train_dir', required=True,
        help='Directory for images to train on.')
    parser.add_argument('--val_dir', required=True,
        help='Directory for images to validate on.')
    parser.add_argument('--label_file', required=True,
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
