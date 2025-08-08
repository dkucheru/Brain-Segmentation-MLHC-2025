import argparse
import os

from constants import global_constants as gc


class GlobalArguments:
    def __init__(self, model_tag, axis, epochs, label, partially_observed, dataload, set_id, checkpoint_dir, job_id,
                 evaluated_model_name, loss_name, lr, model_dir):
        self.ModelTag = model_tag
        self.Axis = axis
        self.Epochs = epochs
        self.Label = label
        self.PartiallyObserved = partially_observed
        self.Dataload = dataload
        self.SetID = set_id
        self.CheckpointDir = checkpoint_dir
        self.JobID = job_id
        self.EvaluatedModelName = evaluated_model_name
        self.LossName = loss_name
        self.LearningRate = lr
        self.ModelDir = model_dir

    def __str__(self):
        return f'\n\n\t\t GLOBAL ARGUMENTS FOR THIS SESSION: ' \
               f'\nModelTag: {self.ModelTag}' \
               f'\nAxis: {self.Axis}' \
               f'\nEpochs: {self.Epochs}' \
               f'\nLabel: {self.Label}' \
               f'\nPartiallyObserved: {self.PartiallyObserved}' \
               f'\nDataload: {self.Dataload}' \
               f'\nSetID: {self.SetID}' \
               f'\nCheckpointDir: {self.CheckpointDir}' \
               f'\nJobID: {self.JobID}' \
               f'\nEvaluatedModelName: {self.EvaluatedModelName}' \
               f'\nLossName: {self.LossName}' \
               f'\nLearningRate: {self.LearningRate}' \
               f'\nModelDir: {self.ModelDir}'

    def __repr__(self):
        return str(self)


def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, help='Model tag to save', default=gc.DEFAULT_MODELTAG)
    parser.add_argument('--axis', type=int, help='Axis to train on',
                        choices=[gc.AXIS_CORONAL, gc.AXIS_SAGITTAL, gc.AXIS_AXIAL], default=gc.DEFAULT_AXIS)
    parser.add_argument('--epochs', type=int, help='Epochs used in training', default=1)
    parser.add_argument('--label', type=int, help='Label type to train on',
                        choices=[gc.LABEL_DHCP_TCV],
                        default=gc.DEFAULT_LABEL)
    parser.add_argument('--partialyobserved', default=False, action='store_true')
    parser.add_argument('--fullyLabeled', default=False, action='store_true')
    parser.add_argument('--dataload', type=int,
                        help='Percentage of fully-labeled MRIS used for training out of original '
                             'dataset', default=100)
    parser.add_argument('--set_id', type=int, help='Id of a set containing 5% of training data', default=0)
    parser.add_argument('--checkpoint_dir', type=str, help='path to save and look for the checkpoint file',
                        default=os.path.join(os.getcwd(), "checkpoints")
                        )
    parser.add_argument('--loss', type=str, help='loss function used during training',
                        choices=['custom_loss_iu', 'loss_huc', 'weighted_loss','weighted_norm'], default='custom_loss_iu')
    parser.add_argument('--job_id', default=None, help='PBS job id')
    parser.add_argument('--eval', type=str, help='Model name to evaluate', default='We are training, not evaluating')
    parser.add_argument('--lr', default='')
    parser.add_argument('--model_dir', type=str, default='')

    args = parser.parse_args()
    return GlobalArguments(model_tag=args.tag,
                           axis=args.axis,
                           epochs=args.epochs,
                           label=args.label,
                           partially_observed=args.partialyobserved,
                           dataload=args.dataload,
                           set_id=args.set_id,
                           checkpoint_dir=args.checkpoint_dir,
                           job_id=args.job_id,
                           evaluated_model_name=args.eval,
                           loss_name=args.loss,
                           lr=args.lr,
                           model_dir=args.model_dir
                           )
