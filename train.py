import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR

import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model import Model
from hparams import hparams as hp

if not os.path.exists('logdir'): os.mkdir('logdir')
logdir = 'logdir/{}'.format(time.strftime("%y-%m-%dT%H-%M-%S"))
os.mkdir(logdir)
print('Saving to {}'.format(logdir))

train_model = Model('train', logdir)
test_model = Model('test', logdir)
writer = tf.summary.FileWriter(logdir, train_model.g)

def weighted_accuracy(confusion):
    accs = confusion * np.eye(hp.num_classes) / confusion.sum(axis=1, keepdims=True)
    weighted_accuracy = accs.sum(axis=0).mean()
    return accs, weighted_accuracy

for e in range(hp.num_epochs):
    train_model.sess.run(train_model.helper.initializer)
    train_loss = 0
    num_batches = train_model.helper.num_batches
    for b in tqdm(range(num_batches), unit='batch'):
        loss, loss_sum, acc, acc_sum, conf, conf_sum, gs = train_model.update()
        writer.add_summary(loss_sum, gs)
        train_loss += loss
    writer.add_summary(acc_sum, gs)
    writer.add_summary(conf_sum, gs)
    train_model.reset_metrics()
    print('Train - Epoch: {}, Avg Loss: {}'.format(e, train_loss/num_batches))
    accs, wa = weighted_accuracy(conf)
    print(hp.classes)
    print(conf)
    print(accs)
    print('Unweighted Accuracy: {}'.format(acc))
    print('Weighted Accuracy: {}'.format(wa))
    save_path = train_model.save_model()

    test_model.load_model(save_path)
    test_model.sess.run(test_model.helper.initializer)
    test_loss = 0
    num_batches = test_model.helper.num_batches
    for b in tqdm(range(num_batches), unit='batch'):
        loss, acc, acc_sum, conf, conf_sum, gs = test_model.test()
        test_loss += loss
    writer.add_summary(acc_sum, e)
    writer.add_summary(conf_sum, gs)
    test_model.reset_metrics()
    print('Test - Epoch: {}, Avg Loss: {}'.format(e, test_loss/num_batches))
    accs, wa = weighted_accuracy(conf)
    print(hp.classes)
    print(conf)
    print(accs)
    print('Unweighted Accuracy: {}'.format(acc))
    print('Weighted Accuracy: {}'.format(wa))


