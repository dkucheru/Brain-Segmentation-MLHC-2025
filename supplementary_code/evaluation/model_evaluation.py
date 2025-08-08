import copy
import os
import sys
import shutil
from pathlib import Path

sys.path.append('..')
sys.path.insert(0, '../..')

from evaluation.segmentation_utils import *


class Evaluation:
    def __init__(self, batch_size, mult_score, parent_child_map, model, test_dataset, test_len,
                 test_parent_filepaths, label_tag_map, evaluation_metrics, current_model_local_dir):
        self.batch_size = batch_size
        self.mult_score = mult_score
        self.parent_child_map = parent_child_map
        self.model = model
        self.test_dataset = test_dataset
        self.test_len = test_len
        self.test_parent_filepaths = test_parent_filepaths
        self.label_tag_map = label_tag_map
        self.evaluation_metrics = evaluation_metrics
        self.current_model_local_dir = current_model_local_dir

    def turn_to_binary(self, binary):
        current_batch_size = len(binary)
        arr_orig = copy.deepcopy(binary)
        # Step 1: Find the maximum number for each vector, turn it to value 1, and make the rest of the values in the
        # current vector 0s
        arr_max = np.max(binary, axis=len(binary.shape) - 1, keepdims=True)
        binary[binary < arr_max] = 0
        binary[binary == arr_max] = 1

        # Step 2: Find vectors with 1 at index 8 and add value 1 at the index of the maximum value between indexes 5 to 7
        # Also if 5,6,7 got selected as argmax - add 1 to index 8
        for parent, sublabels in self.parent_child_map.items():
            for i in range(current_batch_size):
                for j in range(256):
                    for k in range(256):
                        # calculate if any of the sub labels were assigned 1 at Step1
                        res = 0
                        for label in sublabels:
                            res += binary[i][j][k][label]
                        # if the main label is 1 , then choose the most probable sub label and assign 1 to it as well
                        if binary[i][j][k][parent] == 1:
                            labels_probs = arr_orig[i][j][k]
                            sublabels_probs = []

                            for label in sublabels:
                                sublabels_probs.append(labels_probs[label])

                            biggest_sublabel_prob = max(sublabels_probs)

                            for label in sublabels:
                                if labels_probs[label] == biggest_sublabel_prob:
                                    binary[i][j][k][label] = 1
                                    break
                        # if some of those labels was marked as 1, then make main label as 1 as well
                        elif res > 0:
                            for label in sublabels:
                                if binary[i][j][k][label] == 1:
                                    binary[i][j][k][parent] = 1
        return binary

    def create_predictions(self, save_pred_files=False, del_outs=True):
        test_id = 0
        img_slice = None
        mask = None
        prediction = None
        predictions_path = os.path.join(self.current_model_local_dir, 'predictions')
        mkdir(predictions_path)

        num_batches = math.ceil(self.test_len * self.mult_score / self.batch_size)
        for d, l in self.test_dataset.take(num_batches):
            if img_slice is None:
                print(f'{test_id + 1} / {self.test_len}')
                img_slice = d.numpy()
                mask = l.numpy()

                prediction = self.model.predict(d)
                prediction = self.turn_to_binary(prediction)
                prediction = np.expand_dims(prediction, axis=-1)
                prediction = np.squeeze(prediction, axis=-1)
                print(f'first batch length = {len(img_slice)}', flush=True)
            else:
                print(f'{len(img_slice)}', flush=True)
                img_slice = np.concatenate((img_slice, d.numpy()), axis=0)
                mask = np.concatenate((mask, l.numpy()), axis=0)
                prediction_slice = self.model.predict(d)
                prediction_slice = self.turn_to_binary(prediction_slice)
                prediction_slice = np.expand_dims(prediction_slice, axis=-1)
                prediction_slice = np.squeeze(prediction_slice, axis=-1)
                prediction = np.concatenate((prediction, prediction_slice), axis=0)
                print(f'slices num={img_slice.shape[0]}', flush=True)
            if img_slice.shape[0] == 256:
                path_parts = split_path_string(self.test_parent_filepaths[test_id])
                patient_id = path_parts[-2]


                scan = path_parts[-1]
                scan_id_full_name = scan.split("-")[0]
                scan_number = scan_id_full_name.split("_")[1]
                df_row = {'patient_id': patient_id, 'age_id': int(scan_number)}

                for label_num, label_tag in self.label_tag_map.items():

                    t = np.zeros((256, 256, 256))
                    p = np.zeros((256, 256, 256))
                    tFull = copy.deepcopy(mask)  # mask in 3D form
                    pFull = copy.deepcopy(prediction)  # predictions in 3D form

                    for x in range(len(t)):
                        for y in range(len(t[0])):
                            for z in range(len(t[0][0])):
                                t[x][y][z] = tFull[x][y][z][label_num]
                                p[x][y][z] = pFull[x][y][z][label_num]

                    dice_coeff_test = dice_coef(t, p)
                    haus_distance = hausdorff_coef(t.astype(bool), p.astype(bool))
                    vol_disparity = disparity(t, p)

                    df_row[f'dsc_{label_tag}'] = dice_coeff_test
                    df_row[f'hd_{label_tag}'] = haus_distance
                    df_row[f'vol_disparity_{label_tag}'] = vol_disparity

                print(f'Extracted evaluation metric row: {df_row}', flush=True)
                self.evaluation_metrics = self.evaluation_metrics.append(df_row, ignore_index=True)

                test_path = os.path.join(predictions_path, str(patient_id), scan_number)

                mkdir(test_path)
                if save_pred_files:
                    np.save(os.path.join(test_path, 'raw'), img_slice[:311, :, :])
                    np.save(os.path.join(test_path, 'mask'), mask)
                    np.save(os.path.join(test_path, 'prediction'), prediction)
                    print(f'saved files to {test_path}')
                else:
                    print('we do not save masks and prediction files at all')
                test_id += 1

                img_slice = None
                mask = None
                prediction = None

        return self.evaluation_metrics

    def save_stats(self, evaluation_metrics):
        print(evaluation_metrics)
        stats_path = os.path.join(self.current_model_local_dir, 'stats')
        mkdir(stats_path)
        evaluation_metrics.to_csv(os.path.join(stats_path, 'evaluation_metrics_no_postproc.csv'), index=False)
        print(f'saved file to {stats_path}')

        age_labels = ['preterm', 'at term', '8 year']
        dcs_metrics = ['dsc_' + x for x in self.label_tag_map.values()]
        hd_metrics = ['hd_' + x for x in self.label_tag_map.values()]
        vol_metrics = ['vol_disparity_' + x for x in self.label_tag_map.values()]

        eval_summary = ''

        for i, age_tag in enumerate(age_labels):
            eval_summary += f'Age tag: {age_tag}\n'
            i += 1
            means = evaluation_metrics[evaluation_metrics['age_id'] == i].mean()
            stds = evaluation_metrics[evaluation_metrics['age_id'] == i].std(ddof=0)

            for dcs_metric in dcs_metrics:
                eval_summary += f'{dcs_metric}: {means[dcs_metric]:.4f} ± {stds[dcs_metric]:.4f}\n'
            eval_summary += '\n'
            for hd_metric in hd_metrics:
                eval_summary += f'{hd_metric}: {means[hd_metric]:.4f} ± {stds[hd_metric]:.4f}\n'
            for vol_metric in vol_metrics:
                eval_summary += f'{vol_metric}: {means[vol_metric]:.4f} ± {stds[vol_metric]:.4f}\n'
            eval_summary += '\n\n'

        print(eval_summary)
        print()
        print('Saving evaluation summary...')
        # write evaluation summary to file
        with open(os.path.join(stats_path, 'evaluation_summary_nopostproc.txt'), 'w') as f:
            f.write(eval_summary)
        print(f'Saved to {stats_path}')

        print('Done.')


# -------------------------------------------- Helper functions --------------------------------------------------------

def split_path_string(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
