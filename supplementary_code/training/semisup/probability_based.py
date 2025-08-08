import copy
import math
import os
from dotenv import load_dotenv
import sys
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc as pythongc

sys.path.append('..')
sys.path.insert(0, '../..')
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import constants.paths as paths
import constants.global_constants as gc
from constants.ids_for_sets import ids_dHCP, ids_HCP
from constants.ids_for_sets.ids_dHCP import get_patient_ids_by_percent_dHCP
from evaluation.model_evaluation import Evaluation, mkdir, split_path_string
from dataloader import Dataloader2DMultiLabel
from training.unet import custom_loss_iu
from evaluation.basic_evaluation import rm
from predict.predict_arguments import parse

# _____________________________________ Finished imports. Start the code _______________________________________________

# load .env parameters
load_dotenv()

# paths
dHCP_dir = os.getenv('dHCP_dir')
local_models_storage = os.getenv('local_models_storage')
basic_semisup_models_dir = os.getenv('basic_semisup_models_dir')
pseudo_labels_dir = os.getenv('pseudo_labels_dir')

WANDB_KEY = os.getenv('WANDB_KEY')
wandb.login(key=WANDB_KEY)

# Read and load prediction parameters
pred_params = parse()

multi_label_models_storage = local_models_storage
mult_score = 256
multi_label_model_name = ''

pseudo_percent = 100 - pred_params.Dataload
predictions_path = os.path.join(pseudo_labels_dir, str(pseudo_percent), f'{pred_params.ModelName}_PredBased')
mkdir(predictions_path)

# Load the model
model_dir = os.path.join(multi_label_models_storage, pred_params.ModelName)
pred_model = keras.models.load_model(os.path.join(model_dir, 'model'),
                                     custom_objects={'custom_loss_iu': custom_loss_iu})

# Set up Dataloader and label maps
label_map = gc.label_conversion_map(pred_params.Label)
label_tag_map = gc.label_tag_map(pred_params.Label)
if pred_params.Label == gc.LABEL_DHCP_TCV:
    dataloader = Dataloader2DMultiLabel(data_parent_path=dHCP_dir,
                                        pseudo_data_parent_path=predictions_path,
                                        train_axis=gc.DEFAULT_AXIS,
                                        label_map=label_map,
                                        batch_size=gc.BATCH_SIZE,
                                        scale=gc.Scale,
                                        parent_label_index=10,
                                        sublabels_indexes=[2, 3, 7])
    clean_pred_label_map = {10: [2, 3, 7]}
else:
    raise Exception(f'unknown label: {pred_params.Label}')

# Get those train IDS which were not used in training of the model
ids_fully_labeled_data = get_patient_ids_by_percent_dHCP(percent=pred_params.Dataload, set_id=pred_params.SetID)
rest_of_train_data = [x for x in ids_dHCP.train_set_dHCP if x not in ids_fully_labeled_data]

# Prepare the dataset for prediction
print('\nPreparing the dataset for predictions...')
predict_dataset, predict_len, predict_parent_filepaths = dataloader.prepare_dataset(dirs=rest_of_train_data,
                                                                                    repeat=False,
                                                                                    return_parent=True,
                                                                                    make_general_label=True)

# Make predictions with the model and clean the results.
# Cleaning is performed the next way:
# 1. All predicted labels except for the sub-labels of TCV are being replaced with labels from mask
# 2. If after step1 in some pixel of prediction TCV is combined with non-TCV label - we annulate that non-TCV predicted
#    label and put 1 to the sub-label which had the highest model-generated probability.

if pred_params.Label == gc.LABEL_DHCP_TCV:
    evaluation_parent_map_for_casting_preds_to_binary = {10: [2, 3, 7]}
else:
    raise Exception(f'unknown label: {pred_params.Label}')

evaluation = Evaluation(
    batch_size=dataloader.batch_size,
    mult_score=mult_score,
    parent_child_map=evaluation_parent_map_for_casting_preds_to_binary,
    model=pred_model,
    test_dataset=predict_dataset,
    test_len=predict_len,
    test_parent_filepaths=predict_parent_filepaths,
    label_tag_map=label_tag_map,
    evaluation_metrics=None,
    current_model_local_dir=None)

test_id = 0
img_slice = None
mask = None
prediction = None
prediction_probs = None


def prepare(pred):
    pred = np.expand_dims(pred, axis=-1)
    pred = np.squeeze(pred, axis=-1)
    return pred


print('we are going to use debugged code for cleaning')
print('cleaning up the predictions')
num_batches = math.ceil(predict_len * mult_score / gc.BATCH_SIZE)
for d, l in tqdm(predict_dataset.take(num_batches)):
    if img_slice is None:
        print(f'{test_id + 1} / {predict_len}')
        img_slice = d.numpy()
        mask = l.numpy()

        prediction_probs = pred_model.predict(d)
        prediction = prepare(evaluation.turn_to_binary(copy.deepcopy(prediction_probs)))
        prediction_probs = prepare(prediction_probs)
    else:
        print(f'{len(img_slice)}', flush=True)
        img_slice = np.concatenate((img_slice, d.numpy()), axis=0)
        mask = np.concatenate((mask, l.numpy()), axis=0)
        prediction_slice_probs = pred_model.predict(d)
        prediction_slice = prepare(evaluation.turn_to_binary(copy.deepcopy(prediction_slice_probs)))
        prediction = np.concatenate((prediction, prediction_slice), axis=0)

        prediction_slice_probs = prepare(prediction_slice_probs)
        prediction_probs = np.concatenate((prediction_probs, prediction_slice_probs), axis=0)

    if img_slice.shape[0] == 256:
        path_parts = split_path_string(predict_parent_filepaths[test_id])
        patient_id = path_parts[-2]


        scan = path_parts[-1]
        scan_id_full_name = scan.split("-")[0]
        scan_number = scan_id_full_name.split("_")[1]
        df_row = {'patient_id': patient_id, 'age_id': int(scan_number)}

        # Cleaning the labels
        cleaned_prediction = copy.deepcopy(mask)  # took all true labels
        for parent, sublabels in clean_pred_label_map.items():
            for x in range(256):
                for y in range(256):
                    for z in range(256):
                        probs_arr = prediction_probs[x][y][z]

                        # if TCV is known from the true labels :
                        # replace the true number under sub-label index with the predicted number
                        flag = 0
                        if cleaned_prediction[x][y][z][parent] > 0:
                            for label_num in sublabels:
                                cleaned_prediction[x][y][z][label_num] = copy.deepcopy(prediction[x][y][z][label_num])
                                flag += cleaned_prediction[x][y][z][label_num]

                            # check for case when more than one sub-label is predicted as 1
                            if flag > 1 :
                                raise Exception(f'bug in prediction creation; got array: \n'
                                                f'\t\tcleaned_prediction: {cleaned_prediction[x][y][z]}\n'
                                                f'\t\tprediction: {prediction[x][y][z]} \n'
                                                f'\t\tprobs_arr: {probs_arr} \n')
                            # check if none of the sub-labels were predicted by the model, but were present in the true mask
                            if flag < 1:
                                # get the biggest probability within sub-labels
                                highest_prob = max([probs_arr[i] for i in sublabels])
                                # choose the most probable sub-label
                                for label_num in sublabels:
                                    if probs_arr[label_num] == highest_prob:
                                        cleaned_prediction[x][y][z][label_num] = 1
                                        break

        # Removing the non-existing in real datasets TCV label before saving to files
        cleaned_prediction = cleaned_prediction[:, :, :, 0:cleaned_prediction.shape[3] - 1]

        test_path = os.path.join(predictions_path, str(patient_id), scan_number)
        mkdir(test_path)
        # save prediction as shape(256,256,256) and not (256,256,256,N)
        cleaned_prediction = np.argmax(cleaned_prediction, axis=3)
        # change type to float64 so that the file type is the same as original data
        img_slice = img_slice.astype('float64')
        cleaned_prediction = cleaned_prediction.astype('float64')
        print(f'shape of cleaned_prediction : {cleaned_prediction.shape}')
        np.save(os.path.join(test_path, 'raw'), img_slice)
        np.save(os.path.join(test_path, 'prediction'), cleaned_prediction)
        print(f'saved files to {test_path}')

        test_id += 1

        img_slice = None
        mask = None
        prediction = None
        prediction_probs = None

print('-------------------------------------------------------')
print('\t\t SAVED ALL PSEUDO_LABELS')
print('-------------------------------------------------------')
print('\t\t STARTING FITTING THE MODEL')

print('\nPreparing training dataset...')
train_dataset, train_len = dataloader.prepare_dataset(dirs=ids_fully_labeled_data,
                                                      pseudo_ids=rest_of_train_data,
                                                      partial=False,
                                                      set_id=pred_params.SetID,
                                                      repeat=True, shuffle=True,
                                                      make_general_label=True)

print('\nPreparing validation dataset...')
val_dataset, val_len = dataloader.prepare_dataset(ids_dHCP.validation_set_dHCP, repeat=True,
                                                  make_general_label=True)

print('\nPreparing test dataset...')
test_dataset, test_len, test_parent_filepaths = dataloader.prepare_dataset(ids_dHCP.test_set_dHCP, repeat=False,
                                                                               return_parent=True,
                                                                               make_general_label=True)

loss_fn = custom_loss_iu
SemiSup_MODEL_NAME = f'SemiSup_ProbBased_FIXED_{pred_params.ModelName}'
fitted_model_dir = os.path.join(basic_semisup_models_dir, SemiSup_MODEL_NAME)

run = wandb.init(
    # set the wandb project where this run will be logged
    project="2d_segmentation",
    id=pred_params.JobID,
    name=pred_params.JobID,
    # track hyperparameters and run metadata with wandb.config
    config={
        "loss": loss_fn,
        "metric": "accuracy",
        "epoch": pred_params.Epochs,
        "learning_rate": pred_params.LearningRate,
        "batch_size": gc.BATCH_SIZE,
    },
)
try:
    print('\t\t Trying to load the model from the model directory')
    model = keras.models.load_model(os.path.join(model_dir, 'model'), custom_objects={'custom_loss_iu': loss_fn})
    print(f'model loaded from dir: {model_dir}/model')
except:
    raise Exception(f'could not load the model from dir: {model_dir}/model')

mkdir(os.path.join(fitted_model_dir, 'model'))
mkdir(os.path.join(fitted_model_dir, 'checkpoint'))

patience = 30
print(f'patience={patience}')
early_stopping = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
csv_logger = keras.callbacks.CSVLogger(os.path.join(fitted_model_dir, 'checkpoint', 'training_log'))
model_checkpoints = keras.callbacks.ModelCheckpoint(os.path.join(fitted_model_dir, 'checkpoint'), save_best_only=True)

model.fit(x=train_dataset,
          validation_data=val_dataset,
          validation_steps=math.ceil(max(1, val_len * mult_score // gc.BATCH_SIZE)),
          epochs=200,
          steps_per_epoch=math.ceil(max(1, train_len * mult_score // gc.BATCH_SIZE)),
          callbacks=[early_stopping, csv_logger, model_checkpoints,
                     WandbMetricsLogger(log_freq=5),  # logging metrics on graphs
                     WandbModelCheckpoint("models"),
                     ])

model.save(os.path.join(fitted_model_dir, 'model'))
run.finish()  # optional
print('Evaluating model...')
model.evaluate(test_dataset, steps=math.ceil(test_len * mult_score // gc.BATCH_SIZE))

dcs_metrics = ['dsc_' + x for x in label_tag_map.values()]
hd_metrics = ['hd_' + x for x in label_tag_map.values()]
vol_metrics = ['vol_disparity_' + x for x in label_tag_map.values()]
evaluation_metrics = pd.DataFrame(columns=['patient_id', 'age_id'] + dcs_metrics + hd_metrics + vol_metrics)
evaluation = Evaluation(
    batch_size=dataloader.batch_size,
    mult_score=mult_score,
    model=model,
    test_dataset=test_dataset,
    test_len=test_len,
    test_parent_filepaths=test_parent_filepaths,
    label_tag_map=label_tag_map,
    evaluation_metrics=evaluation_metrics,
    current_model_local_dir=fitted_model_dir,
    parent_child_map=evaluation_parent_map_for_casting_preds_to_binary)

evaluation_metrics = evaluation.create_predictions(save_pred_files=False)
evaluation.save_stats(evaluation_metrics)

rm(predictions_path)
