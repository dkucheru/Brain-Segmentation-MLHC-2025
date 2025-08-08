import sys

from constants.ids_for_sets import ids_dHCP

sys.path.insert(0, '..')

import numpy as np

import tensorflow as tf
import os
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm
import constants.global_constants as gc


def calc_percent(num, percent):
    res = int(percent / 100 * num)
    return res


class Dataloader2DMultiLabel:
    def __init__(self, data_parent_path: str,
                 train_axis, label_map, batch_size, scale,
                 parent_label_index=None, sublabels_indexes=None, pseudo_data_parent_path: str = None):
        self.data_parent_path = data_parent_path
        self.train_axis = train_axis
        self.label_map = label_map
        self.batch_size = batch_size
        self.scale = scale
        self.parent_label_index = parent_label_index
        self.sublabels_indexes = sublabels_indexes
        self.pseudo_data_parent_path = pseudo_data_parent_path

    def get_labels_settings(self):
        return len(self.label_map) + 1, self.parent_label_index, np.sort(self.sublabels_indexes), self.scale

    def add_general_label(self, one_hot_encoded):
        num_labels, parent_label_index, sublabels_indexes, scale = self.get_labels_settings()

        # setting interesting_columns to all zeros before running the loop
        interesting_columns = tf.convert_to_tensor(np.zeros((scale, scale, scale, 1)), dtype=tf.float64)

        for i in sublabels_indexes:
            current_column = tf.reshape(one_hot_encoded[..., i], (scale, scale, scale, 1))
            interesting_columns = tf.concat([interesting_columns, current_column], 3)
        interesting_columns = interesting_columns[..., 1:]

        one_in_interesting_columns = tf.math.reduce_sum(interesting_columns, axis=-1) > 0
        one_in_interesting_columns = tf.cast(one_in_interesting_columns, tf.int64) * (parent_label_index + 1)
        gen_label_mask = tf.one_hot(one_in_interesting_columns, num_labels + 1, dtype=tf.float64)
        gen_label_mask = gen_label_mask[..., 1:]
        return one_hot_encoded + gen_label_mask

    def make_unobserved(self, labels):
        num_labels, parent_label_index, sublabels_indexes, scale = self.get_labels_settings()

        print(f'\n\n\n\t\t\t Making {sublabels_indexes} unobserved')
        # take just the parent label column
        parent_column = tf.reshape(labels[..., parent_label_index], (scale, scale, scale, 1))

        sublabels_indexes = np.sort(sublabels_indexes)
        # setting unobserved_mask to all zeros before running the loop
        unobserved_mask = tf.zeros((scale, scale, scale, num_labels), dtype=tf.dtypes.float64)

        # each iteration of the loop creates 1 mask and adds it to the final unobserved_mask
        # each mask is just a (256,256,256,num_labels) array full of zeros, where in the column of current sublabels
        # we have 1s if the parent label is present in that array
        for sublabel in sublabels_indexes:
            zeros_on_the_left = tf.zeros((scale, scale, scale, sublabel), dtype=tf.dtypes.float64)
            zeros_on_the_right = tf.zeros((scale, scale, scale, num_labels - (sublabel + 1)),
                                          dtype=tf.dtypes.float64)
            add_zeros = tf.concat([zeros_on_the_left, parent_column, zeros_on_the_right], 3)

            # now let`s reduce the original values of the sublabels from the current mask
            # (needs to be done, so we don`t get 3s at the end)

            sublabel_column = tf.reshape(labels[..., sublabel], (scale, scale, scale, 1))
            original_state = tf.concat([zeros_on_the_left, sublabel_column, zeros_on_the_right], 3)
            add_zeros = add_zeros * 2 - original_state
            unobserved_mask += add_zeros

        return labels + unobserved_mask

    def prepare_dataset(self, dirs, repeat, pseudo_ids=None, percent_dataload=100, set_id=0, partial=False,
                        shuffle=False,
                        return_parent=False, make_general_label=True):
        def reshape_function(data_filename, label_filename):

            data = tf.io.read_file(data_filename)
            label = tf.io.read_file(label_filename)

            data = tf.io.decode_raw(data, tf.float64, little_endian=True)
            label = tf.io.decode_raw(label, tf.float64)
            # Get rid of the first 16 bytes as it is the numpy header
            data = data[16:]
            label = label[16:]
            scale = self.scale
            data = tf.reshape(data, (scale, scale, scale))
            # turns MRI data points into numbers in range of 0 to 1 (Ex: 99 -> 0.9 ; 45 -> 0.45)
            data = tf.math.divide(tf.subtract(data, tf.reduce_min(data)),
                                  tf.subtract(tf.reduce_max(data), tf.reduce_min(data)))

            for desired_label, actual_label in self.label_map.items():
                if type(actual_label) == list:
                    for i in actual_label:
                        label = tf.where(tf.equal(label, i), desired_label * tf.ones_like(label, dtype=tf.float64),
                                         label)
                else:
                    if actual_label != desired_label:
                        label = tf.where(tf.equal(label, actual_label),
                                         desired_label * tf.ones_like(label, dtype=tf.float64), label)

            label = tf.reshape(label, (scale, scale, scale))

            if self.train_axis == gc.AXIS_CORONAL:
                print('\n\t\t\t NOT THIS PLANE: CHANGE TO SAGITTAL')
            elif self.train_axis == gc.AXIS_SAGITTAL:
                # add 1 to length of label_map because we do not describe label 0 in the map
                all_labels = len(self.label_map) + 1
                label = tf.cast(label, tf.int64)
                label = tf.one_hot(label, all_labels)
                label = tf.cast(label, tf.float64)
                if make_general_label:
                    label = self.add_general_label(one_hot_encoded=label)
                else:
                    print('\n\n\n\t\t\t didnt create general label')
                if partial:
                    label = self.make_unobserved(labels=label)
                else:
                    print('\n\n\n\t\t\t didnt apply partial mask')

                return tf.data.Dataset.from_tensor_slices(([data[i, :, :] for i in range(scale)],
                                                           [label[i, :, :] for i in range(scale)]))
            elif self.train_axis == gc.AXIS_AXIAL:
                print('\n\t\t\t NOT THIS PLANE: CHANGE TO SAGITTAL')
            else:
                raise Exception("Invalid train axis!")

        data_fullylabeled_filepaths = []
        label_fullylabeled_filepaths = []

        data_partial_filepaths = []
        label_partial_filepaths = []

        parent_filepaths = []
        num_samples = 0

        if percent_dataload != 100:
            fully_labeled_dirs = ids_dHCP.get_patient_ids_by_percent_dHCP(percent_dataload, set_id)
            print(f'\n\n\t\tfully_labeled_dirs: \n\t\t{fully_labeled_dirs}')
            partially_labeled_dirs = [x for x in dirs if x not in fully_labeled_dirs]
        else:
            fully_labeled_dirs = dirs
            partially_labeled_dirs = []

        for patient_id in tqdm(dirs):
            for scan_id in os.listdir(os.path.join(self.data_parent_path, patient_id)):
                if ".tsv" not in scan_id:
                    session = f'{scan_id.split("-")[1]}-{scan_id.split("-")[2]}'
                    raw_file_name = f'{patient_id}_{session}_T2w_RESHAPED-{self.scale}.npy'
                    mask_file_name = f'{patient_id}_{session}_desc-drawem9_dseg_RESHAPED-{self.scale}.npy'

                    data_filepath = os.path.join(self.data_parent_path, patient_id, scan_id, 'anat', raw_file_name)
                    label_filepath = os.path.join(self.data_parent_path, patient_id, scan_id, 'anat',
                                                  mask_file_name)

                    parent_filepaths.append(os.path.join(self.data_parent_path, patient_id, scan_id))

                    if patient_id in fully_labeled_dirs:
                        data_fullylabeled_filepaths.append(data_filepath)
                        label_fullylabeled_filepaths.append(label_filepath)
                    else:
                        data_partial_filepaths.append(data_filepath)
                        label_partial_filepaths.append(label_filepath)

                    num_samples += 1

        if pseudo_ids is not None:
            print('started loading pseudo labels')
            for patient_id in pseudo_ids:
                for scan_id in os.listdir(os.path.join(self.pseudo_data_parent_path, patient_id)):
                    data_filepath = os.path.join(self.pseudo_data_parent_path, patient_id, scan_id, 'raw.npy')
                    label_filepath = os.path.join(self.pseudo_data_parent_path, patient_id, scan_id, 'prediction.npy')

                    parent_filepaths.append(os.path.join(self.pseudo_data_parent_path, patient_id, scan_id))

                    data_fullylabeled_filepaths.append(data_filepath)
                    label_fullylabeled_filepaths.append(label_filepath)

                    num_samples += 1

        if partial:

            datasetPartial = tf.data.Dataset.from_tensor_slices((data_partial_filepaths, label_partial_filepaths))
            datasetPartial = datasetPartial.map(reshape_function, num_parallel_calls=AUTOTUNE)
            datasetPartial = datasetPartial.flat_map(lambda x: x)
            print(
                f'{100 - percent_dataload}% of data got partially labeled. Which is {len(label_partial_filepaths)} out '
                f'of {num_samples}:')
            for path in label_partial_filepaths:
                print(f'\t{path}')

            partial = False

            datasetFull = tf.data.Dataset.from_tensor_slices(
                (data_fullylabeled_filepaths, label_fullylabeled_filepaths))
            datasetFull = datasetFull.map(reshape_function, num_parallel_calls=AUTOTUNE)
            datasetFull = datasetFull.flat_map(lambda x: x)
            print(
                f'Rest ({percent_dataload}%) of data is fully labeled. Which is {len(data_fullylabeled_filepaths)} out '
                f'of {num_samples}:')

            dataset = datasetPartial.concatenate(datasetFull)


        else:
            dataset = tf.data.Dataset.from_tensor_slices((data_fullylabeled_filepaths, label_fullylabeled_filepaths))
            print(
                f'We are using {percent_dataload}% of data available. Which is {len(data_fullylabeled_filepaths)} out '
                f'of {num_samples}')
            dataset = dataset.map(reshape_function, num_parallel_calls=AUTOTUNE)
            dataset = dataset.flat_map(lambda x: x)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=384, reshuffle_each_iteration=True)

        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)

        if return_parent:
            return dataset, num_samples, parent_filepaths
        return dataset, num_samples
