''' Contains tools to read the serialized .tfrecord files and generate a tf.data.Dataset.


Notes
-----
This module is meant to be imported in the training pipeline. Just run
the function generate_datasets() (see function docs below for details on the right 
parameters...) to produce the desired tf.data.Dataset. If the TFRecords are produced 
independently, the convention we are adopting for filenames 
is audioformat_num.tfrecord (e.g. waveform_74.tfrecord).

This module makes use of the performance optimization 
highlighted here: https://www.tensorflow.org/beta/guide/data_performance. You can 
substitute tf.data.experimental.AUTOTUNE with the right parameter 
if you feel it proper or necessary.


Functions
---------

- create_hash
    Creates the hash table from the clusters dictionary

- _parse_features
    Parse the serialized tf.Example.

- _reshape
    Reshape flattened audio tensors into the original ones.

- _merge
    Merge similar tags together.

- _tag_filter
    Remove (tracks with) unwanted tags from the dataset.

- _tid_filter
    Remove (tracks with) unwanted tids from the dataset.

- _tag_filter_hotenc_mask
    Change the shape of tag hot-encoded vector to suit the output of _tag_filter.

- _window_1
    Extract a sample of n seconds from each audio tensor within a batch.

- _window_2
    Extract a sample of n seconds from each audio tensor within a batch.

- _window
    Return either _window_1 if audio-format is waveform, or _window_2 if audio-format is log-mel-spectrogram.

- _spect_normalization
    Ensure zero mean and unit variance within a batch of log-mel-spectrograms.

- _batch_normalization
    Ensure zero mean and unit variance within a batch.

- _tuplify
    Transform features from dict to tuple.

- generate_datasets
    Combine all previous functions to produce a list of train/valid/test datasets.

- generate_datasets_from_dir
    Combine all previous functions to produce a list of train/valid/test datasets, fetch all .tfrecords files from a root directory.
'''

import json
import os
import pickle

import numpy as np
import tensorflow as tf


def create_hash(path_dict, filter_genres_path=None):
    ''' Creates the hash table from the clusters dictionary
    
    Parameters
    ----------
    path_dict: str
        Full path to the dictionary that links the tags with the clusters.
         
    path_dict: str
        Path of the directory that contains genres.txt. The content of this file is explained in train.py
        
    '''

    with open(path_dict, "rb") as f:
        dictionary_clusters = pickle.load(f)

    keys_tensor = tf.constant(list(dictionary_clusters.keys()))

    if filter_genres_path is None:
        clusters = np.unique(list(dictionary_clusters.values()))  # clusters is a sorted list
        length_encoding = len(clusters)
        converter = {}

        for i in range(len(clusters)):
            converter[clusters[i]] = i

        new_values = np.zeros((len(dictionary_clusters), length_encoding), dtype=np.int64)

        for i, j in enumerate(dictionary_clusters.values()):
            new_values[i, converter[j]] = 1

    else:

        with open(os.path.join(filter_genres_path, 'genres.txt')) as f:
            genres, merge_list = f.readlines()
            genres = np.sort(json.loads(genres))
            merge_list = json.loads(merge_list)

        length_encoding = len(genres)
        converter_genres = {}
        new_values = np.zeros((len(dictionary_clusters), length_encoding), dtype=np.int64)
        clusters = np.unique(list(dictionary_clusters.values()))  # clusters is a sorted list
        converter = {}

        for i in range(len(clusters)):
            converter[int(clusters[i])] = i

        for i, j in enumerate(genres):
            converter_genres[int(j)] = i

        merged_genres = []
        for i, j in merge_list:
            if i in genres and j in genres:
                merged_genres.append(converter_genres[j])
                converter_genres[j] = converter_genres[i]
                length_encoding -= 1

        for i, j in enumerate(dictionary_clusters.values()):
            if converter[j] in genres:
                new_values[i, converter_genres[converter[j]]] = 1
        new_values = np.delete(new_values, merged_genres, axis=1)

    table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string,
                                                  value_dtype=tf.int64, empty_key="<EMPTY_SENTINEL>",
                                                  deleted_key="<DELETE_SENTINEL>",
                                                  default_value=tf.zeros(length_encoding, dtype=tf.int64))
    table.insert(keys_tensor, new_values)
    return table, length_encoding


def _parse_features(example, features_dict, shape):
    ''' Parses the serialized tf.Example. '''
    features_dict = tf.io.parse_single_example(example, features_dict)
    features_dict['audio'] = tf.reshape(tf.sparse.to_dense(features_dict['audio']), shape)
    features_dict['tags'] = tf.reshape(tf.sparse.to_dense(features_dict['tags']), (-1,))

    return features_dict


def _merge(features_dict, tags):
    ''' Removes unwanted tids from the dataset (use with tf.data.Dataset.map).
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    tags: list or list-like
        List of lists of tags to be merged. Writes 1 for all tags in the hot-encoded vector whenever at least one tag of the list is present.

    Examples
    --------
    >>> features['tags'] = [0, 1, 1, 0, 0]
    >>> _merge(features, merge_tags=[[0, 1], [2, 3]])
    features['tags']: [1, 1, 1, 1, 0]
    >>> _merge(features, merge_tags=[[0, 1], [3, 4]])
    features['tags']: [1, 1, 1, 0, 0] 
    '''

    tags_databases = len(features_dict) - 2  # check if multiple databases have been provided
    num_tags = tf.cast(tf.shape(features_dict['tags']), tf.int64)

    tags = tf.dtypes.cast(tags, tf.int64)
    idxs = tf.subtract(tf.reshape(tf.sort(tags), [-1, 1]), tf.constant(1, dtype=tf.int64))
    vals = tf.constant(1, dtype=tf.int64, shape=tags.get_shape())
    tags = tf.SparseTensor(indices=idxs, values=vals, dense_shape=num_tags)
    tags = tf.sparse.to_dense(tags)
    tags = tf.dtypes.cast(tags, tf.bool)

    def _fn(tag_str, num_tags=num_tags):  # avoid repetitions of code by defining a handy function
        feature_tags = tf.dtypes.cast(features_dict[tag_str], tf.bool)
        # if at least one of the feature tags is in the current 'tags' list, write True in the bool-hot-encoded vector for all tags in 'tags'; otherwise, leave feature tags as they are
        features_dict[tag_str] = tf.where(tf.math.reduce_any(tags & feature_tags), tags | feature_tags, feature_tags)
        features_dict[tag_str] = tf.cast(features_dict[tag_str], tf.int64)

    if tags_databases > 1:
        for i in range(tags_databases):
            _fn('tags_' + str(i))
    else:
        _fn('tags')

    return features_dict


@tf.autograph.experimental.do_not_convert
def _tag_filter(features_dict, hash_table):
    ''' Removes the tids from the dataset that maps into tags without embedding.
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).

    hash_table: hash
        Table that links the tags with the encoding of the clusters.

    '''

    tags = features_dict['tags']
    tags = tf.reduce_sum(hash_table[tags], 0)
    compare = tf.constant(0, dtype=tf.int64)
    return tf.math.logical_not(tf.reduce_all(tf.equal(tags, compare)))


def _tid_filter(features_dict, tids):
    ''' Removes unwanted tids from the dataset based on given tids (use with tf.data.Dataset.filter).
        
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).

    tids: list or list-like
        List containing tids (as strings) to be "allowed" in the output dataset.
    '''

    tids = tf.constant(tids, tf.string)
    return tf.math.reduce_any(tf.math.equal(tids, features_dict['tid']))


@tf.autograph.experimental.do_not_convert
def _tag_filter_hotenc_mask(features_dict, hash_table):
    ''' Transforms the string representation of the tags into a hot-encoded vector with the same lenght as the list 'tags'.
    
    Parameters
    ----------
    features: dict
        Dict of features (as provided by .map).

    hash_table: hash
        Table that links the tags with the encoding of the clusters.
        
    
    '''

    tags = features_dict['tags']
    tags = tf.reduce_sum(hash_table[tags], 0)
    tags = tf.cast(tf.cast(tags, tf.bool), tf.int64)  # transform the components with value greater than one in 1
    features_dict['tags'] = tags

    return features_dict


def _window_1(features_dict, sample_rate, window_length=15, window_random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    sample_rate: int
        Specifies the sample rate of the audio track.
    
    window_length: int
        Length (in seconds) of the desired output window.
    
    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    slice_length = tf.math.multiply(tf.constant(window_length, dtype=tf.int32),
                                    tf.constant(sample_rate, dtype=tf.int32))  # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(window_random, dtype=tf.bool)

    def fn1a(audio, slice_length=slice_length):
        maxval = tf.subtract(tf.shape(audio, out_type=tf.int32)[0], slice_length)
        x = tf.cond(tf.equal(maxval, tf.constant(0)), lambda: tf.constant(0, dtype=tf.int32),
                    lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
        y = tf.add(x, slice_length)
        audio = audio[x:y]
        return audio

    def fn1b(audio):
        mid = tf.math.floordiv(tf.shape(audio, out_type=tf.int32)[0],
                               tf.constant(2, dtype=tf.int32))  # find midpoint of audio tensors
        x = tf.subtract(mid, tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32)))
        y = tf.add(mid, tf.math.floordiv(tf.add(slice_length, tf.constant(1)), tf.constant(2,
                                                                                           dtype=tf.int32)))  # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
        audio = audio[x:y]
        return audio

    features_dict['audio'] = tf.cond(random, lambda: fn1a(features_dict['audio']), lambda: fn1b(features_dict['audio']))
    return features_dict


def _window_2(features_dict, sample_rate, window_length=15, window_random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    sample_rate: int
        Specifies the sample rate of the audio track.
    
    window_length: int
        Length (in seconds) of the desired output window.
    
    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    slice_length = tf.math.floordiv(
        tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)),
        tf.constant(512, dtype=tf.int32))  # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(window_random, dtype=tf.bool)

    def fn2a(audio, slice_length=slice_length):
        maxval = tf.subtract(tf.shape(audio, out_type=tf.int32)[1], slice_length)
        x = tf.cond(tf.equal(maxval, tf.constant(0)), lambda: tf.constant(0, dtype=tf.int32),
                    lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
        x = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
        y = tf.add(x, slice_length)
        audio = audio[:, x:y]
        return audio

    def fn2b(audio):
        mid = tf.math.floordiv(tf.shape(audio, out_type=tf.int32)[1],
                               tf.constant(2, dtype=tf.int32))  # find midpoint of audio tensors
        x = tf.subtract(mid, tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32)))
        y = tf.add(mid, tf.math.floordiv(tf.add(slice_length, tf.constant(1)), tf.constant(2,
                                                                                           dtype=tf.int32)))  # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
        audio = audio[:, x:y]
        return audio

    features_dict['audio'] = tf.cond(random, lambda: fn2a(features_dict['audio']), lambda: fn2b(features_dict['audio']))
    return features_dict


def _window(audio_format):
    ''' Returns the right window function, depending to the specified audio-format. '''

    return {'waveform': _window_1, 'log-mel-spectrogram': _window_2}[audio_format]


def _spect_normalization(features_dict):
    ''' Normalizes the log-mel-spectrograms within a batch. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[1, 2], keepdims=True)
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset=0, scale=1,
                                                       variance_epsilon=.000001)
    return features_dict


def _batch_normalization(features_dict):
    ''' Normalizes a batch. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[0])
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset=0, scale=1,
                                                       variance_epsilon=.000001)
    return features_dict


def _tuplify(features_dict, which_tags=None):
    ''' Transforms a batch into (audio, tags) tuples, ready for training or evaluation with Keras. 
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).
    '''

    return (features_dict['audio'], features_dict['tags'])


def generate_datasets(tfrecords, audio_format, path_dict, split=None, which_split=None, sample_rate=16000, num_mels=96,
                      batch_size=32, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000,
                      window_length=15, window_random=False, with_tids=None, repeat=None, as_tuple=True,
                      filter_genres_path=None, coocurrence_matrix=None):
    ''' Reads the TFRecords and produces a list tf.data.Dataset objects ready for training/evaluation.
    
    Parameters:
    ----------
    tfrecords: str, list
        List of .tfrecord files paths.

    audio_format: {'waveform', 'log-mel-spectrogram'}
        Specifies the feature audio format.
        
    path_dict: str
        Full path to the dictionary that links the tags with the clusters.

    split: tuple
        Specifies the number of train/validation/test files to use when reading the .tfrecord files.
        If values add up to 100, they will be treated as percentages; otherwise, they will be treated as actual number of files to parse.

    which_split: tuple
        Applies boolean mask to the datasets obtained with split. Specifies which datasets are actually returned.

    sample_rate: int
        Specifies the sample rate used to process audio tracks.

    batch_size: int
        Specifies the dataset batch_size.

    block_length: int
        Controls the number of input elements that are processed concurrently.

    cycle_length: int
        Controls the number of input elements that are processed concurrently.

    shuffle: bool
        If True, shuffles the dataset with buffer size = shuffle_buffer_size.

    shuffle_buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_length: int
        Specifies the desired window length (in seconds).

    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).

    num_mels: int
        The number of mels in the mel-spectrogram.        
        
    with_tids: list
        If not None, contains the tids to be trained on.

    repeat: int
        If not None, repeats the dataset only for a given number of epochs (default is repeat indefinitely).

    as_tuple: bool
        If True, discards tid's and transforms features into (audio, tags) tuples.
        
    filter_genres_path: string
        path to genres.txt, explained in the argument parser  
    '''

    AUDIO_SHAPE = {'waveform': (-1,), 'log-mel-spectrogram': (num_mels, -1)}  # set audio tensors dense shape

    AUDIO_FEATURES_DESCRIPTION = {'audio': tf.io.VarLenFeature(tf.float32), 'tid': tf.io.FixedLenFeature((), tf.string),
                                  'tags': tf.io.VarLenFeature(tf.string)}  # tags will be added just below

    hash_table, n_output_neurons = create_hash(path_dict, filter_genres_path)

    if coocurrence_matrix:
        correlation_matrix = tf.zeros((n_output_neurons, n_output_neurons), dtype=tf.int32)

    assert audio_format in ('waveform', 'log-mel-spectrogram'), 'please provide a valid audio format'

    tfrecords = np.array(tfrecords, dtype=np.unicode)  # allow for single str as input
    tfrecords = np.vectorize(lambda x: os.path.abspath(os.path.expanduser(x)))(
        tfrecords)  # fix issues with relative paths in input list

    if split:
        if np.sum(split) == 100:
            np_split = np.cumsum(split) * len(tfrecords) // 100
        else:
            assert np.sum(split) <= len(tfrecords), 'split exceeds the number of available .tfrecord files'
            np_split = np.cumsum(split)
        tfrecords_split = np.split(tfrecords, np_split)
        tfrecords_split = tfrecords_split[:-1]  # discard last empty split
    else:
        tfrecords_split = [tfrecords]

    datasets = []

    for files_list in tfrecords_split:
        if files_list.size > 1:  # read files in parallel (number of parallel threads specified by cycle_length)
            files = tf.data.Dataset.from_tensor_slices(files_list)
            print(files)
            dataset = files.interleave(tf.data.TFRecordDataset, block_length=block_length, cycle_length=cycle_length,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.TFRecordDataset(files_list)

        # parse serialized features
        dataset = dataset.map(lambda x: _parse_features(x, AUDIO_FEATURES_DESCRIPTION, AUDIO_SHAPE[audio_format]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffle
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)

        # map the tags into the corresponding clusters
        dataset = dataset.filter(lambda y: _tag_filter(y, hash_table))
        dataset = dataset.map(lambda y: _tag_filter_hotenc_mask(y, hash_table))

        if coocurrence_matrix:
            for example in dataset:  # We use this loop of code to compute the coocurrence between the labels
                example1 = tf.cast(example['tags'], tf.bool)
                example2 = tf.cast(tf.expand_dims(example['tags'], axis=-1), tf.bool)
                correlation_matrix += tf.cast(tf.logical_and(example1, example2), tf.int32)

        if with_tids is not None:
            dataset = dataset.filter(lambda x: _tid_filter(x, tids=with_tids))

        # slice into audio windows
        dataset = dataset.map(lambda x: _window(audio_format)(x, sample_rate, window_length, window_random),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # batch
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # normalize data
        if audio_format == 'log-mel-spectrogram':
            dataset = dataset.map(_spect_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(_batch_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # convert features from dict into tuple
        if as_tuple:
            dataset = dataset.map(lambda x: _tuplify(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        datasets.append(dataset)
        print(datasets)

    if split:
        datasets = np.where(np.array(split) != 0, datasets, None)  # useful when split contains zeros

    if which_split is not None:
        if split is not None:
            assert len(which_split) == len(split), 'split and which_split must have the same length'
            datasets = np.array(datasets)[np.array(which_split, dtype=np.bool)].tolist()
        else:
            datasets = datasets + [None] * (which_split.count(
                1) - 1)  # useful when trying to unpack datasets (if you need a fixed number of datasets), but split has not been provided

    if coocurrence_matrix:
        with open(os.path.join(coocurrence_matrix, 'coocurrence_matrix_' + str(n_output_neurons) + '.npy'), 'wb') as f:
            np.save(f, np.array(correlation_matrix))

    if len(datasets) == 1:
        return datasets[0], n_output_neurons
    else:
        return datasets[0], datasets[1], n_output_neurons


def generate_datasets_from_dir(tfrecords_dir, audio_format, path_dict, split=None, which_split=None, sample_rate=16000,
                               num_mels=96, batch_size=32, block_length=1, cycle_length=1, shuffle=True,
                               shuffle_buffer_size=10000, window_length=15, window_random=False, with_tids=None,
                               repeat=1, as_tuple=True, filter_genres_path=None, coocurrence_matrix=None):
    ''' Reads the TFRecords from the input directory and produces a list tf.data.Dataset objects ready for training/evaluation.
    
    Parameters:
    ----------
    tfrecords_dir: str
        Directory containing the .tfrecord files.
        
    path_dict: str
        Full path to the dictionary that links the tags with the clusters.

    split: tuple
        Specifies the number of train/validation/test files to use when reading the .tfrecord files.
        If values add up to 100, they will be treated as percentages; otherwise, they will be treated as actual number of files to parse.

    which_split: tuple
        Applies boolean mask to the datasets obtained with split. Specifies which datasets are actually returned.

    sample_rate: int
        Specifies the sample rate used to process audio tracks.

    batch_size: int
        Specifies the dataset batch_size.

    block_length: int
        Controls the number of input elements that are processed concurrently.

    cycle_length: int
        Controls the number of input elements that are processed concurrently.

    shuffle: bool
        If True, shuffles the dataset with buffer size = shuffle_buffer_size.

    shuffle_buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_length: int
        Specifies the desired window length (in seconds).

    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    
    num_mels: int
        The number of mels in the mel-spectrogram.
  
    with_tids: list
        If not None, contains the tids to be trained on.

    repeat: int
        If not None, repeats the dataset only for a given number of epochs (default is repeat indefinitely).

    as_tuple: bool
        If True, discards tid's and transforms features into (audio, tags) tuples.
        
    filter_genres_path: string
        path to genres.txt, explained in the argument parser    
    '''

    tfrecords = []

    # scan all the files in tfrecords_dir
    for file in os.listdir(os.path.expanduser(tfrecords_dir)):
        # Example waveform_2.tfrecord file.split(...)=["waveform","2.tfrecord"]
        if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:
            tfrecords.append(os.path.abspath(
                os.path.join(tfrecords_dir, file)))  # The abspath ensures that the string is in path format

    return generate_datasets(tfrecords, audio_format, path_dict, split=split, which_split=which_split,
                             sample_rate=sample_rate, batch_size=batch_size,
                             block_length=block_length, cycle_length=cycle_length, shuffle=shuffle,
                             shuffle_buffer_size=shuffle_buffer_size,
                             window_length=window_length, window_random=window_random,
                             num_mels=num_mels, repeat=repeat, as_tuple=as_tuple, filter_genres_path=filter_genres_path,
                             coocurrence_matrix=coocurrence_matrix)
