# coding: utf-8
import os
import caffe
import logging
import numpy as np
import pandas as pd
import time


class ImageClassifier(object):
    default_args = {
        'model_def_file': 'model/deploy.prototxt',
        'pretrained_model_file': 'model/bvlc_reference_caffenet.caffemodel',
        'mean_file': 'model/mean.npy',
        'class_labels_file': 'model/labels.txt',
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:])
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort_values(by='synset_id')['name'].values

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (nn, p.replace('(', '').replace(')', ''), float(scores[i]))
                for nn, (i, p) in enumerate(zip(indices, predictions))
            ]
            logging.info('result: %s', str(meta))
            return max(meta, key=lambda x: x[2])

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
