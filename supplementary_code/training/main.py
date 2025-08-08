import math
import sys
from dotenv import load_dotenv

sys.path.append('..')
sys.path.insert(0, '../..')

import pandas as pd
import gc as pythongc
import numpy as np
import keras
import tensorflow
import keras.losses
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.data import AUTOTUNE
import constants.ids_for_sets.ids_dHCP as ids_dHCP
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import constants.global_constants as gc
from dataloader import Dataloader2DMultiLabel
from arguments_parser import *
from unet import custom_loss_iu, UNet2D, LossHUC, weighted_loss, \
    weighted_normalized_loss
from evaluation.model_evaluation import mkdir
from evaluation.model_evaluation import Evaluation

# load .env parameters
load_dotenv()

# paths
dHCP_dir = os.getenv('dHCP_dir')
local_models_storage = os.getenv('local_models_storage')
local_models_storage_HCP = os.getenv('local_models_storage_HCP')
basic_semisup_models_dir = os.getenv('basic_semisup_models_dir')
pseudo_labels_dir = os.getenv('pseudo_labels_dir')

# login into wandb API
WANDB_KEY = os.getenv('WANDB_KEY')
wandb.login(key=WANDB_KEY)

params = parse_input_arguments()
print(params)
label_map = gc.label_conversion_map(params.Label)
label_tag_map = gc.label_tag_map(params.Label)

if params.PartiallyObserved:
    MODEL_NAME = f'{params.ModelTag}_Partial_{params.Dataload}pc_{gc.get_string_tag(params.Axis)}_{gc.get_string_tag(params.Label)}_job{params.JobID}'
else:
    MODEL_NAME = f'{params.ModelTag}_FullyLabeled_{params.Dataload}pc_{gc.get_string_tag(params.Axis)}_{gc.get_string_tag(params.Label)}_job{params.JobID}'

data_parent_path = dHCP_dir
local_models_storage = local_models_storage

current_model_local_dir = os.path.join(local_models_storage, MODEL_NAME)
print(f'current_model_local_dir: {current_model_local_dir}')
print(label_map)
print(f'BATCH_SIZE={gc.BATCH_SIZE}')
if params.Label == gc.LABEL_DHCP_TCV:
    dataloader = Dataloader2DMultiLabel(data_parent_path, params.Axis, label_map, gc.BATCH_SIZE, gc.Scale,
                                        parent_label_index=10,
                                        sublabels_indexes=[2, 3, 7])
else:
    raise Exception(f'unknown label: {params.Label}')


print('\nPreparing training dataset...')
train_dataset, train_len = dataloader.prepare_dataset(ids_dHCP.train_set_dHCP,
                                                      partial=params.PartiallyObserved,
                                                      percent_dataload=params.Dataload,
                                                      set_id=params.SetID,
                                                      repeat=True, shuffle=True)
print('\nPreparing validation dataset...')
val_dataset, val_len = dataloader.prepare_dataset(ids_dHCP.validation_set_dHCP, repeat=True)

print('\nPreparing test dataset...')
test_dataset, test_len, test_parent_filepaths = dataloader.prepare_dataset(ids_dHCP.test_set_dHCP, repeat=False,
                                                                           return_parent=True)

print(params.LossName)
if params.LossName == 'custom_loss_iu':
    loss_fn = custom_loss_iu
elif params.LossName == 'weighted_loss':
    loss_fn = weighted_loss
elif params.LossName == 'loss_huc':
    loss_fn = LossHUC(parent_index=dataloader.parent_label_index,
                              child_indexes=dataloader.sublabels_indexes)
elif params.LossName == 'weighted_norm':
    loss_fn = weighted_normalized_loss
else:
    raise ValueError(f'Unknown loss function: {params.LossName}')

get_custom_objects().update({params.LossName: loss_fn})  # log the custom loss function into the model

run = wandb.init(
    # set the wandb project where this run will be logged
    project="2d_segmentation",
    id=params.JobID,
    name=params.JobID,
    # track hyperparameters and run metadata with wandb.config
    config={
        "loss": loss_fn,
        "metric": "accuracy",
        "epoch": params.Epochs,
        "learning_rate": params.LearningRate,
        "batch_size": gc.BATCH_SIZE,
    },
)
config = run.config

mult_score = 256

try:
    print('\t\t Trying to load the model from checkpoints')
    model = keras.models.load_model(params.CheckpointDir, custom_objects={params.LossName: loss_fn})
    print(f'model loaded from checkpoint: {params.CheckpointDir}')
except:
    print(f'\t\t Could not load checkpoint from {params.CheckpointDir}\n\t\t Creating a new model')
    try:
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=gc.get_learning_rate(params.LearningRate))
        print(f'\n\nUsing Adam starting from {gc.get_learning_rate(params.LearningRate)}\n\n')
    except:
        raise Exception(f'unknown learning rate: {gc.get_learning_rate(params.LearningRate)}')

    print('Adapting normalizer to training set')
    num_train_batches = math.ceil(train_len * mult_score / gc.BATCH_SIZE)
    t = train_dataset.take(num_train_batches)
    print(t)
    t = t.map(lambda x, y: x).prefetch(AUTOTUNE)
    normalizer = keras.layers.Normalization(axis=None)
    normalizer.adapt(t, steps=num_train_batches)
    print('Normalizer adapted to train set')
    print('Mean:', normalizer.mean)
    print('Variance:', normalizer.variance)

    model = UNet2D((gc.Scale, gc.Scale, 1), normalizer=normalizer, num_classes=len(label_map) + 1)
    model.compile(optimizer=optimizer,
                  loss=params.LossName,  # compile the model with the custom loss function
                  metrics=[config.metric])  # wandb metrics

patience = 30
print(f'patience={patience}')
early_stopping = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
csv_logger = keras.callbacks.CSVLogger(os.path.join(params.CheckpointDir, 'training_log'))
model_checkpoints = keras.callbacks.ModelCheckpoint(params.CheckpointDir, save_best_only=True)

mkdir(os.path.join(current_model_local_dir, 'model'))

model.fit(x=train_dataset,
          validation_data=val_dataset,
          validation_steps=max(1, math.ceil(val_len * mult_score / gc.BATCH_SIZE)),
          epochs=params.Epochs,
          steps_per_epoch=max(1, math.ceil(train_len * mult_score / gc.BATCH_SIZE)),
          callbacks=[early_stopping,
                     csv_logger,
                     model_checkpoints,
                     WandbMetricsLogger(log_freq=5),  # logging metrics on graphs
                     WandbModelCheckpoint("models")], )

model.save(os.path.join(current_model_local_dir, 'model'))

run.finish()  # optional

print('Evaluating model...')
model.evaluate(test_dataset, steps=test_len * mult_score // gc.BATCH_SIZE)

dcs_metrics = ['dsc_' + x for x in label_tag_map.values()]
hd_metrics = ['hd_' + x for x in label_tag_map.values()]
vol_metrics = ['vol_disparity_' + x for x in label_tag_map.values()]

evaluation_metrics = pd.DataFrame(columns=['patient_id', 'age_id'] + dcs_metrics + hd_metrics + vol_metrics)

if params.Label == gc.LABEL_DHCP_TCV:
    evaluation_parent_map_for_casting_preds_to_binary = {10: [2, 3, 7]}
else:
    raise Exception(f'unknown label: {params.Label}')

evaluation = Evaluation(
    batch_size=dataloader.batch_size,
    mult_score=mult_score,
    parent_child_map=evaluation_parent_map_for_casting_preds_to_binary,
    model=model,
    test_dataset=test_dataset,
    test_len=test_len,
    test_parent_filepaths=test_parent_filepaths,
    label_tag_map=label_tag_map,
    evaluation_metrics=evaluation_metrics,
    current_model_local_dir=current_model_local_dir)

evaluation_metrics = evaluation.create_predictions(save_pred_files=False)
evaluation.save_stats(evaluation_metrics)
