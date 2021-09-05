import argparse
import datetime
import gc
import os
import shutil
import time
import numpy as np
import tensorflow as tf
from custom_metric import AUC_multitag as AUC
from data_input import generate_datasets_from_dir
from orpheus_model import build_model
from orpheus_model import parse_config_json


''' Contains tools to train our model using the mirrored distribution strategy.

Notes
-----
This module can be run as a script. To do so, just type 'python train.py' in the terminal. The help 
page should contain all the options you might possibly need.

This module contains the actual training function used to train a model
using the built-in Keras model.fit API, or alternatively a custom training loop. 
It combines the data input pipeline defined in data_input.py (which makes use of 
the .tfrecord files previously generated by preprocessing.py), 
the CNN architecture proposed by Pon et. al (2018) defined in orpheus_model.py, 
and a standard training loop with mirrored strategy 
integrated for multiple-GPU training.

The training function tries to optimise BinaryCrossentropy for each tag
prediction (batch loss is summed over the batch size, instead of being averaged), and 
displays the area under the ROC and PR curves as metrics. The optimizer can
be fully specified in the config.json file.


Logs and checkpoints are automatically saved in subdirectories named after the 
frontend adopted and a timestamp. 
Logs can be accessed and viewed using TensorBoard. 

The config.json file is automatically copied in the directory for future reference.

IMPORTANT: if trying to profile a batch in TensorBoard, make sure the environment
variable LD_LIBRARY_PATH is specified.
(e.g. 'export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:
                               /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"')
'''


def train(train_dataset, valid_dataset, frontend, strategy, config, epochs, n_clusters, steps_per_epoch=None,
          timestamp_to_resume=None, update_freq=1, analyse_trace=False, multi_tag=True, save_ROC=None, save_PR=None,
          save_confusion_matrix=None):
    ''' Creates a compiled instance of the training model and trains it for 'epochs' epochs using a custom training loop.

    Parameters
    ----------
    train_dataset: tf.data.Dataset
        The training dataset.
        
    valid_dataset: tf.data.Dataset
        The validation dataset. If None, validation will be disabled. Tfe callbacks might not work properly.

    frontend: {'waveform', 'log-mel-spectrogram'}
        The frontend to adopt.
        
    strategy: tf.distribute.Strategy
        Strategy for multi-GPU distribution.

    config: argparse.Namespace
        Instance of the config namespace. It is generated when parsing the config.json file.
        
    epochs: int
        Specifies the number of epochs to train for.
        
    n_clusters: int
        Specifies the number of output neurons
    
    timestamp_to_resume: str
        Specifies the timestamp of the checkpoint to restore. Should be a timestamp in the 'YYMMDD-hhmm' format.

    update_freq: int
        Specifies the number of batches to wait before writing to logs. Note that writing too frequently can slow down training.
    
    analyse_trace: bool
        Specifies whether to enable profiling.
                
    multi_tag: bool
        If true, we compute the macro-averaged metrics, otherwise, we compute the micro-averaged ones
        
    save_ROC
        Path in which we save a npy file containing an array representing the AUC ROC per label

    save_PR
        Path in which we save a npy file containing an array representing the AUC PR per label

    save_confusion_matrix
        Path in which we save a npy file containing an array representing the confusion matrix per label        

    '''

    timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M")

    record_ROC = []
    record_PR = []
    record_confusion_matrix = []

    with strategy.scope():

        num_replica = strategy.num_replicas_in_sync
        tf.print('num_replica:', num_replica)

        # build model
        model = build_model(frontend, num_output_neurons=n_clusters, num_units=config.n_dense_units,
                            num_filts=config.n_filters, y_input=config.n_mels)

        # initialise loss, optimizer and metrics
        optimizer = tf.keras.optimizers.get({"class_name": config.optimizer_name, "config": config.optimizer})
        train_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        train_mean_loss = tf.keras.metrics.Mean(name='train_mean_loss', dtype=tf.float32)
        if multi_tag:
            train_metrics_1 = AUC(curve='ROC', name='train_AUC-ROC')
            train_metrics_2 = AUC(curve='PR', name='train_AUC-PR')
        else:
            train_metrics_1 = tf.keras.metrics.AUC(curve='ROC', name='train_AUC-ROC', dtype=tf.float32)
            train_metrics_2 = tf.keras.metrics.AUC(curve='PR', name='train_AUC-PR', dtype=tf.float32)

        # set up checkpoint
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        prev_epoch = -1

        # set up logs and checkpoints directories
        if timestamp_to_resume is None:
            log_dir = os.path.join(os.path.expanduser(config.log_dir), 'custom_' + frontend[:13] + '_' + timestamp)
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            shutil.copy(config.path, log_dir)  # copy config file in the same folder where the models will be saved
        else:
            log_dir = os.path.join(os.path.expanduser(config.log_dir), 'custom_' + frontend[
                                                                                   :13] + '_' + timestamp_to_resume)  # keep saving logs and checkpoints in the 'old' folder

            # try to load checkpoint
            chkp = tf.train.latest_checkpoint(log_dir)
            if chkp:
                tf.print("Checkpoint file {} found. Restoring...".format(chkp))
                checkpoint.restore(chkp)
                tf.print("Checkpoint restored.")
                prev_epoch = int(chkp.split('-')[-1]) - 1  # last completed epoch number (from 0)
            else:
                tf.print("Checkpoint file not found!")
                return

        tf.summary.trace_off()  # in case of previous keyboard interrupt

        # setting up summary writers
        train_log_dir = os.path.join(log_dir, 'train/')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if valid_dataset:
            val_log_dir = os.path.join(log_dir, 'validation/')
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            if multi_tag:
                val_metrics_1 = AUC(curve='ROC', name='val_AUC-ROC')
                val_metrics_2 = AUC(curve='PR', name='val_AUC-PR')
            else:
                val_metrics_1 = tf.keras.metrics.AUC(curve='ROC', name='val_AUC-ROC', dtype=tf.float32)
                val_metrics_2 = tf.keras.metrics.AUC(curve='PR', name='val_AUC-PR', dtype=tf.float32)
            val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)

        if analyse_trace:  # make sure the variable LD_LIBRARY_PATH is properly set up
            print('TIPS: To ensure the profiler works correctly, make sure the LD_LIBRARY_PATH is set correctly. \
                  For Boden, set--- export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64" before Python is initialised.')
            prof_log_dir = os.path.join(log_dir, 'profile/')
            prof_summary_writer = tf.summary.create_file_writer(prof_log_dir)

        # rescale loss
        def compute_loss(labels, predictions):
            per_example_loss = train_loss(labels, predictions)
            return per_example_loss / config.batch_size

        def train_step(batch):
            audio_batch, label_batch = batch
            with tf.GradientTape() as tape:
                logits = model(audio_batch)
                loss = compute_loss(label_batch, logits)
            variables = model.trainable_variables
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            train_metrics_1.update_state(label_batch, logits)
            train_metrics_2.update_state(label_batch, logits)
            train_mean_loss.update_state(loss)
            return loss

        def valid_step(batch):
            audio_batch, label_batch = batch
            logits = model(audio_batch, training=False)
            loss = compute_loss(label_batch, logits)

            val_metrics_1.update_state(label_batch, logits)
            val_metrics_2.update_state(label_batch, logits)
            val_loss.update_state(loss)
            return loss

        @tf.function
        def distributed_train_body(batch, epoch, num_replica):
            num_batches = 0
            if multi_tag:
                for batch in train_dataset:
                    strategy.run(train_step, args=(batch,))
                    num_batches += 1
                    # print metrics after each iteration
                    if tf.equal(num_batches % update_freq, 0):
                        tf.print('Epoch', epoch, '; Step', num_batches, '; loss',
                                 tf.multiply(train_mean_loss.result(), num_replica), '; ROC_AUC',
                                 tf.math.reduce_mean(train_metrics_1.result()), ';PR_AUC',
                                 tf.math.reduce_mean(train_metrics_2.result()))

                        with train_summary_writer.as_default():
                            tf.summary.scalar('ROC_AUC_itr', tf.math.reduce_mean(train_metrics_1.result()),
                                              step=optimizer.iterations)
                            tf.summary.scalar('PR_AUC_itr', tf.math.reduce_mean(train_metrics_2.result()),
                                              step=optimizer.iterations)
                            tf.summary.scalar('Loss_itr', tf.multiply(train_mean_loss.result(), num_replica),
                                              step=optimizer.iterations)
                            train_summary_writer.flush()
                    gc.collect()
            else:
                for batch in train_dataset:
                    # strategy is the strategy for paralelising the GPU computation
                    strategy.run(train_step, args=(batch,))
                    num_batches += 1
                    # print metrics after each iteration
                    if tf.equal(num_batches % update_freq, 0):
                        tf.print('Epoch', epoch, '; Step', num_batches, '; loss',
                                 tf.multiply(train_mean_loss.result(), num_replica), '; ROC_AUC',
                                 train_metrics_1.result(), ';PR_AUC', train_metrics_2.result())
                        with train_summary_writer.as_default():
                            # Sumary is a method for the tensorboard
                            tf.summary.scalar('ROC_AUC_itr', train_metrics_1.result(), step=optimizer.iterations)
                            tf.summary.scalar('PR_AUC_itr', train_metrics_2.result(), step=optimizer.iterations)
                            tf.summary.scalar('Loss_itr', tf.multiply(train_mean_loss.result(), num_replica),
                                              step=optimizer.iterations)
                            train_summary_writer.flush()
                    gc.collect()

        @tf.function
        def distributed_val_body(batch):
            for batch in valid_dataset:
                strategy.run(valid_step, args=(batch,))
                gc.collect()

        # loop
        for epoch in tf.range(prev_epoch + 1, epochs, dtype=tf.int64):
            start_time = time.time()
            tf.print()
            tf.print()
            tf.print('Epoch {}/{}'.format(epoch, epochs - 1))

            if analyse_trace and tf.equal(epoch, 1):
                tf.summary.trace_off()
                tf.summary.trace_on(graph=False, profiler=True)

            distributed_train_body(train_dataset, epoch, num_replica)
            gc.collect()

            # write metrics on tensorboard after each epoch
            with train_summary_writer.as_default():
                if multi_tag:
                    tf.summary.scalar('ROC_AUC_epoch', tf.math.reduce_mean(train_metrics_1.result()), step=epoch)
                    tf.summary.scalar('PR_AUC_epoch', tf.math.reduce_mean(train_metrics_2.result()), step=epoch)
                    tf.summary.scalar('mean_loss_epoch', tf.multiply(train_mean_loss.result(), num_replica), step=epoch)
                    train_summary_writer.flush()
                else:
                    tf.summary.scalar('ROC_AUC_epoch', train_metrics_1.result(), step=epoch)
                    tf.summary.scalar('PR_AUC_epoch', train_metrics_2.result(), step=epoch)
                    tf.summary.scalar('mean_loss_epoch', tf.multiply(train_mean_loss.result(), num_replica), step=epoch)
                    train_summary_writer.flush()

            # print progress
            if multi_tag:
                tf.print('Epoch', epoch, ': loss', tf.multiply(train_mean_loss.result(), num_replica), '; ROC_AUC',
                         tf.math.reduce_mean(train_metrics_1.result()), '; PR_AUC',
                         tf.math.reduce_mean(train_metrics_2.result()))
            else:
                tf.print('Epoch', epoch, ': loss', tf.multiply(train_mean_loss.result(), num_replica), '; ROC_AUC',
                         train_metrics_1.result(), '; PR_AUC', train_metrics_2.result())

            train_metrics_1.reset_states()
            train_metrics_2.reset_states()
            train_mean_loss.reset_states()

            # write training profile
            if analyse_trace:
                with prof_summary_writer.as_default():
                    tf.summary.trace_export(name="trace",
                                            step=epoch,
                                            profiler_outdir=os.path.normpath(prof_log_dir))

            if valid_dataset:
                distributed_val_body(valid_dataset)
                gc.collect()

                # write metris on tensorboard after each epoch
                if multi_tag:
                    with val_summary_writer.as_default():
                        tf.summary.scalar('ROC_AUC_epoch', tf.math.reduce_mean(val_metrics_1.result()), step=epoch)
                        tf.summary.scalar('PR_AUC_epoch', tf.math.reduce_mean(val_metrics_2.result()), step=epoch)
                        tf.summary.scalar('mean_loss_epoch', tf.multiply(val_loss.result(), num_replica), step=epoch)
                        val_summary_writer.flush()
                else:
                    with val_summary_writer.as_default():
                        tf.summary.scalar('ROC_AUC_epoch', val_metrics_1.result(), step=epoch)
                        tf.summary.scalar('PR_AUC_epoch', val_metrics_2.result(), step=epoch)
                        tf.summary.scalar('mean_loss_epoch', tf.multiply(val_loss.result(), num_replica), step=epoch)
                        val_summary_writer.flush()

                if multi_tag:
                    tf.print('Val- Epoch', epoch, ': loss', tf.multiply(val_loss.result(), num_replica), ';ROC_AUC',
                             val_metrics_1.result(), '; PR_AUC', val_metrics_2.result())
                    record_ROC.append(val_metrics_1.result().numpy())
                    record_PR.append(val_metrics_2.result().numpy())
                    if epoch == epochs - 1:
                        record_confusion_matrix.append(val_metrics_1.true_positives.numpy())
                        record_confusion_matrix.append(val_metrics_1.true_negatives.numpy())
                        record_confusion_matrix.append(val_metrics_1.false_positives.numpy())
                        record_confusion_matrix.append(val_metrics_1.false_negatives.numpy())
                else:
                    tf.print('Val- Epoch', epoch, ': loss', tf.multiply(val_loss.result(), num_replica), ';ROC_AUC',
                             tf.math.reduce_mean(val_metrics_1.result()), '; PR_AUC',
                             tf.math.reduce_mean(val_metrics_2.result()))

                    # reset validation metrics after each epoch
                val_metrics_1.reset_states()
                val_metrics_2.reset_states()
                val_loss.reset_states()

            checkpoint_path = os.path.join(log_dir, 'epoch' + str(epoch.numpy()))
            saved_path = checkpoint.save(checkpoint_path)
            tf.print('Saving model as TF checkpoint: {}'.format(saved_path))

            # report time
            time_taken = time.time() - start_time
            tf.print('Epoch {}: {} s'.format(epoch, time_taken))

    if multi_tag:
        if save_ROC:
            with open(os.path.join(save_ROC, 'record_ROC_' + frontend + '_' + timestamp + '.npy'), 'wb') as f:
                np.save(f, np.array(record_ROC))
        if save_PR:
            with open(os.path.join(save_PR, 'record_PR_' + frontend + '_' + timestamp + '.npy'), 'wb') as f:
                np.save(f, np.array(record_PR))
        if save_confusion_matrix:
            with open(os.path.join(save_confusion_matrix,
                                   'record_confusion_matrix_' + frontend + '_' + timestamp + '.npy'), 'wb') as f:
                np.save(f, np.array(record_confusion_matrix))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('frontend', choices=['waveform', 'log-mel-spectrogram'])
    parser.add_argument('--tfrecords-dir', dest='tfrecords_dir',
                        help='directory to read the .tfrecord files from')
    parser.add_argument('--dictionary-dir', dest='dictionary_dir',
                        help='directory to read the dictionary that links the words and the clusterings')
    parser.add_argument('--config', help='path to config.json (default to path on Boden)', default='~/config.json')
    parser.add_argument('--multi-db', help='specify the number of different tags features in the .tfrecord files',
                        type=int, default=1)
    parser.add_argument('--multi-db-default',
                        help='specify the index of the default tags database, when there are more than one tags features in the .tfrecord files',
                        type=int)
    parser.add_argument('--epochs', help='specify the number of epochs to train on', type=int, required=True)
    parser.add_argument('--steps-per-epoch',
                        help='specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset)',
                        type=int)
    parser.add_argument('--no-shuffle', action='store_true', help='force no shuffle, override config setting')
    parser.add_argument('--resume',
                        help='load a previously saved model with the time in the format ddmmyy-hhmm, e.g. if the folder which the model is saved is custom_log-mel-spect_160919-0539, resume should take the argument 160919-0539')
    parser.add_argument('--update-freq', help='specify the frequency (in steps) to record metrics and losses', type=int,
                        default=10)
    parser.add_argument('--cuda', help='set cuda visible devices', type=int, nargs='+')
    parser.add_argument("--save-PR", dest='save_PR',
                        help='path in which we save a npy file containing an array representing the AUC PR per label if multi-tags is True')
    parser.add_argument("--save-ROC", dest='save_ROC',
                        help='path in which we save a npy file containing an array representing the AUC ROC per label if multi-tags is True')
    parser.add_argument("--save-confusion-matrix", dest='save_confusion_matrix',
                        help='path in which we save a npy file containing an array representing the confusion matrix per label if multi-tags is True')
    parser.add_argument('--multi-tag', action='store_true',
                        help='If true, we compute the macro-averaged metrics, otherwise, we compute the micro-averaged ones')
    parser.add_argument('--built-in', action='store_true', help='train using the built-in model.fit training loop')
    parser.add_argument('--genre-path', dest='genre_path',
                        help='path to the directory which containts genres.txt, a file containing the clusters with genres and the clusters to merge them, the txt is composed by two lines, the first one is a list of clusters [x,..,z] and the second one is a list of pairs of clusters to be merged [[x,y],...]')
    parser.add_argument('--save-coocurrence-matrix-path', dest='save_coocurrence_matrix_path',
                        help='path to the directory in which we save the co-occuernce matrix')
    parser.add_argument('-v', '--verbose', choices=['0', '1', '2', '3'], help='verbose mode', default='0')

    args = parser.parse_args()

    # specify number of visible gpu's
    if args.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.cuda])

    # specify verbose mode
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose

    # parse config
    config = parse_config_json(args.config)

    # override config setting
    if args.no_shuffle:
        config.shuffle = False

    # generate train_dataset and valid_dataset (valid_dataset will be None if config.split is None)
    train_dataset, valid_dataset, n_output_neurons = generate_datasets_from_dir(args.tfrecords_dir, args.frontend,
                                                                                args.dictionary_dir, split=config.split,
                                                                                which_split=(True, True,) + (False,) * (
                                                                                            len(config.split) - 2),
                                                                                sample_rate=config.sample_rate,
                                                                                batch_size=config.batch_size,
                                                                                block_length=config.interleave_block_length,
                                                                                cycle_length=config.interleave_cycle_length,
                                                                                shuffle=config.shuffle,
                                                                                shuffle_buffer_size=config.shuffle_buffer_size,
                                                                                window_length=config.window_length,
                                                                                window_random=config.window_random,
                                                                                num_mels=config.n_mels, as_tuple=True,
                                                                                filter_genres_path=args.genre_path,
                                                                                coocurrence_matrix=args.save_coocurrence_matrix_path)

    # set up training strategy
    strategy = tf.distribute.MirroredStrategy()

    # datasets need to be manually 'distributed'
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    if valid_dataset is not None:
        valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    # train model using custom training loop (default choice)
    train(train_dataset, valid_dataset, frontend=args.frontend,
          strategy=strategy, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
          config=config,
          update_freq=args.update_freq, n_clusters=n_output_neurons, timestamp_to_resume=args.resume,
          multi_tag=args.multi_tag, save_ROC=args.save_ROC, save_PR=args.save_PR,
          save_confusion_matrix=args.save_confusion_matrix)
