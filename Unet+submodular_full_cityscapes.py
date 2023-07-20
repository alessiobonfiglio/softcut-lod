import colorsys
import math
import os
from datetime import datetime

import albumentations as albm
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.loggers.neptune import NeptuneLogger
from skimage import measure
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
from torch_submod.graph_cuts import TotalVariation2dWeighted
from torchvision import transforms

from datasets_dic import get_dataset_by_name_and_splits
from loss_dic import get_loss_function_by_name, loss_support_mask, loss_support_ignore_index
from utils import get_monitor_mode, get_preprocessing_params, BIoU, std_for_list_of_tensors, weighted_mean, weighted_std

NUM_OF_WORKERS = 3

VERSION_NUMBER = 1

# Get configuration from file and cli
base_conf = OmegaConf.load(os.path.join("config", "Unet+submodular_multiclass_full_resolution.yml"))
cli_conf = OmegaConf.from_cli()
helper_conf = OmegaConf.create({"num_of_workers": NUM_OF_WORKERS, "extra_tags": []})
# Merge configurations and extract some optional parameters
conf = OmegaConf.merge(helper_conf, base_conf, cli_conf)

# Generate the experiment name and tags
EXPERIMENT_NAME = "Unet(" + conf.params.segmentation_architecture + "+" + conf.params.encoder + ")" + (
    "_submodular" if conf.params.enable_graphcut else "") + "_multiclass_full_resolution"
EXPERIMENT_TAGS = [EXPERIMENT_NAME]
EXPERIMENT_TAGS.extend(conf.extra_tags)

# Set configuration immutable
OmegaConf.set_readonly(conf, True)
OmegaConf.set_struct(conf, True)

# Add additional tags
if conf.params.refine_with_isotonic_regression:
    EXPERIMENT_TAGS.append("isotonic")

# Set the global seed for the RNGs
pl.seed_everything(conf.params.seed)

# Use the encoder preprocessing if we use a pretrained model
if conf.params.encoder_weights is not None:
    normalize_params = get_preprocessing_params(conf.params.encoder, conf.params.encoder_weights)
else:
    normalize_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}  # the default params used by albumentation

# Define the augmentations
train_transform = albm.Compose(
    [
        #albm.PadIfNeeded(256, 256),
        # albm.ShiftScaleRotate(shift_limit=0.0),
        # albm.RandomSizedCrop((100, 256), 256, 256),
        # albm.Flip(),
        # albm.GaussNoise(),
        # albm.ColorJitter(),
        albm.Normalize(**normalize_params),
        ToTensorV2(),
    ]
)

test_transform = albm.Compose(
    [
        #albm.PadIfNeeded(256, 256),
        # albm.CenterCrop(256, 256),
        albm.Normalize(**normalize_params),
        ToTensorV2()
    ]
)

EXPERIMENT_TAGS.append("no_augmentation")


class MulticlassSegDataset(Dataset):

    def __init__(self, image_list, image_folder, mask_folder, augmentation=None, background_channel=True):

        self.image_list = [os.path.join(image_folder , x) for x in image_list]
        self.mask_list = [os.path.join(mask_folder , x[:-15] + "gtFine_color.png") for x in image_list]
        self.augmentation = augmentation
        self.background_channel = background_channel

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        x = Image.open(self.image_list[index]).convert('RGB')
        x = np.array(x)
        masks = Image.open(self.mask_list[index]).convert('RGB')
        masks = np.array(masks)
        
        y = ()
        
        
        #cars
        tmp = np.ones_like(masks[:,:,0], bool)
        tmp[(masks != [0,  0,142]).any(axis=2)] = False
        y = *y, tmp
        
        #person
        tmp = np.ones_like(masks[:,:,0], bool)
        tmp[(masks != [220, 20, 60]).any(axis=2)] = False
        y = *y, tmp
        
        #road
        tmp = np.ones_like(masks[:,:,0], bool)
        tmp[(masks != [128, 64,128]).any(axis=2)] = False
        y = *y, tmp

        if self.augmentation is not None:
            y = [m.astype(float) for m in y]
            transformed = self.augmentation(image=x, masks=y)
            x = transformed["image"]
            y = tuple(transformed["masks"])
        else:
            y = tuple([transforms.ToTensor()(m) for m in y])
            x = transforms.ToTensor()(x)

        #fix albmumentation bug
        if type(y[0]) == np.ndarray:
            y = tuple([torch.from_numpy(m) for m in y])


        if self.background_channel:
            yyy = torch.logical_or(y[0], y[1])
            for yy in y[2:]:
                yyy = torch.logical_or(yyy, yy)
            y = torch.logical_not(yyy), *y

        return x, y


parent_dir = os.path.join("/ds_shared", "cityscapes")
training_image_list = glob.glob(os.path.join(parent_dir, "leftImg8bit", "train", "*", "*"))
test_image_list = glob.glob(os.path.join(parent_dir, "leftImg8bit", "val", "*", "*"))

training_image_list = [os.path.join(os.path.split(os.path.split(x)[0])[1], os.path.split(x)[1]) for x in training_image_list]
test_image_list = [os.path.join(os.path.split(os.path.split(x)[0])[1], os.path.split(x)[1]) for x in test_image_list]

train_ds = MulticlassSegDataset(training_image_list, os.path.join(parent_dir, "leftImg8bit", "train"), os.path.join(parent_dir, "gtFine", "train"), augmentation=train_transform)
test_ds = MulticlassSegDataset(test_image_list, os.path.join(parent_dir, "leftImg8bit", "val"), os.path.join(parent_dir, "gtFine", "val"), augmentation=test_transform)

train_ds, val_ds = torch.utils.data.random_split(train_ds, [2598, 377])

train_loader = DataLoader(train_ds, batch_size=conf.params.batch_size, num_workers=conf.num_of_workers,
                          shuffle=conf.params.shuffle_training_set)
val_loader = DataLoader(val_ds, batch_size=conf.params.batch_size, num_workers=conf.num_of_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=conf.num_of_workers)


# Define the LightningModule that contains the NN
class CNNWithGC(pl.LightningModule):
    def __init__(self, lr, batch_size,
                 n_classes,
                 use_graph_cut, class_weight_type,
                 segmentation_model, encoder_name, encoder_weights, encoder_depth,
                 normalize_params,
                 accuracy_threshold=conf.params.accuracy_threshold,
                 skip_connection=False,
                 use_isotonic_regression=False,
                 max_tv_iters=0,
                 num_workers=1, use_multithreading_for_graphcut=False,
                 **kwargs):
        super().__init__()

        self.use_graph_cut = use_graph_cut

        seg_model = getattr(smp, segmentation_model)(encoder_name=encoder_name,
                                                     encoder_weights=encoder_weights,
                                                     encoder_depth=encoder_depth,
                                                     in_channels=3,
                                                     classes=n_classes,
                                                     **kwargs)
        self.encoder = seg_model.encoder
        self.decoder = seg_model.decoder
        self.segmentation_head = seg_model.segmentation_head

        if self.use_graph_cut:
            self.gc = TotalVariation2dWeighted(refine=use_isotonic_regression,
                                               num_workers=4,
                                               multiprocess=use_multithreading_for_graphcut,
                                               tv_args={'max_iters': max_tv_iters})
            self.use_isotonic_regression = use_isotonic_regression
            self.max_tv_iters = max_tv_iters
            seg_model_2 = getattr(smp, segmentation_model)(encoder_name=encoder_name,
                                                           encoder_weights=encoder_weights,
                                                           encoder_depth=encoder_depth,
                                                           in_channels=3,
                                                           classes=1,
                                                           **kwargs)
            self.weights_decoder = seg_model_2.decoder
            self.weights_head = seg_model_2.segmentation_head

        self.learning_rate = lr
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.accuracy_threshold = accuracy_threshold
        self.example_input_array = self.split(torch.unsqueeze(train_ds.__getitem__(0)[0], 0))[0:1,...]
        self.normalize_params = normalize_params
        self.class_weight_type = class_weight_type
        self.skip_connection = skip_connection

        self.save_hyperparameters()

        self.saved_images = 0

        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.train_IoUs = []
        self.val_IoUs = []
        self.test_IoUs = []
        self.train_statistics_weights = []
        self.val_statistics_weights = []
        self.test_statistics_weights = []

        if conf.log_debug_image and self.use_graph_cut:
            self.last_pre_cut_image = None
            self.last_weights_row = None
            self.last_weights_col = None

    def forward(self, x):
        encoding = self.encoder(x)
        pixel_scores = self.decoder(*encoding)
        pixel_scores = self.segmentation_head(pixel_scores)

        if self.use_graph_cut:
            weights_row = self.weights_decoder(*encoding)
            weights_row = self.weights_head(weights_row)

            # compute the weights for the columns using the
            # same network that we used for the weights of the
            # rows, but transposing the encoding before and after
            weights_col = map(lambda t: torch.transpose(t, -1, -2), encoding)
            weights_col = self.weights_decoder(*weights_col)
            weights_col = self.weights_head(weights_col)
            weights_col = torch.transpose(weights_col, -1, -2)

            # we remove the unneeded rows/columns
            weights_row = weights_row[:, :, :, :-1]
            weights_col = weights_col[:, :, :-1, :]

            # we remove the channel dimension
            # because TotalVariation2dWeighted needs tensors in the shape [BS,H,W]
            weights_row = torch.squeeze(weights_row, 1)
            weights_col = torch.squeeze(weights_col, 1)

            # the CNNs compute the log of the weights, so we need to transform them
            weights_row = torch.nn.Softplus()(weights_row)
            weights_col = torch.nn.Softplus()(weights_col)

            if conf.log_debug_image and self.use_graph_cut:
                self.last_pre_cut_image = pixel_scores
                self.last_weights_row = weights_row
                self.last_weights_col = weights_col

            if self.skip_connection:
                identity = pixel_scores

            # use channels as batch size to exploit parallelism
            gc_res = []
            for i in pixel_scores.unbind(1):
                gc_res.append(self.gc(i, weights_row, weights_col).unsqueeze(1))
            pixel_scores = torch.cat(gc_res, 1)

            if self.skip_connection:
                pixel_scores = pixel_scores + identity
        
        pixel_scores = torch.nn.Softmax(dim=1)(pixel_scores)

        return pixel_scores

    def compute_accuracy(self, y_hat, y):
        correct = (y_hat == y.bool())
        return correct.sum() / torch.numel(correct)

    def compute_multiclass_image(self, y, y_norm):
        out = np.array(Image.new(mode="RGB", size=tuple(y.size()[:-3:-1])))
        class_ids = torch.argmax(y, dim=0)
        not_background = 1 - y_norm[0]
        for id in torch.unique(class_ids):
            if id != 0:
                color = colorsys.hsv_to_rgb((id / (self.n_classes - 1)).item(), 1.0, 1.0)
                color = [int(math.floor(i * 255)) for i in color]
                out[torch.logical_and((class_ids == id).bool(), not_background).numpy()] = list(color)
        return Image.fromarray(out).convert("RGB")
    
    def extra_test_metrics(self, y_hat, y):
        from sklearn import metrics
        import numpy
        import warnings

        n_elem = torch.numel(y[0])
        y_hat_tresholded = y_hat > self.accuracy_threshold
        y = [i.detach().cpu().numpy() for i in y]
        y_hat = y_hat.detach().cpu().numpy()
        y_hat_tresholded = y_hat_tresholded.detach().cpu().numpy()

        tn_norm_tot = []
        fp_norm_tot = []
        fn_norm_tot = []
        tp_norm_tot = []
        false_positive_rate_tot = []
        false_negative_rate_tot = []
        true_negative_rate_tot = []
        negative_predictive_value_tot = []
        false_discovery_rate_tot = []
        recall_tot = []
        precision_tot = []
        f1_tot = []
        matthews_corr_tot = []
        roc_auc_tot = []
        pr_auc_tot = []
        cross_entropy_tot = []
        balanced_accuracy_tot = []
        cohen_kappa_tot = []

        suffix = "_class0"
        yy = y[0].flatten()
        yy_hat = y_hat[0, ...].flatten()
        yy_hat_tresholded = y_hat_tresholded[0, ...].flatten()

        tn, fp, fn, tp = metrics.confusion_matrix(yy, yy_hat_tresholded, labels=[0., 1.]).ravel()
        tn_norm = tn / n_elem
        fp_norm = fp / n_elem
        fn_norm = fn / n_elem
        tp_norm = tp / n_elem
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            false_positive_rate = numpy.nan_to_num(fp / (fp + tn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            false_negative_rate = numpy.nan_to_num(fn / (tp + fn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            true_negative_rate = numpy.nan_to_num(tn / (tn + fp), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            negative_predictive_value = numpy.nan_to_num(tn / (tn + fn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            false_discovery_rate = numpy.nan_to_num(fp / (tp + fp), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            recall = numpy.nan_to_num(tp / (tp + fn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            precision = numpy.nan_to_num(tp / (tp + fp), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            f1 = numpy.nan_to_num(2. * (precision * recall) / (precision + recall), nan=numpy.nan, posinf=numpy.nan,
                                  neginf=numpy.nan)
            balanced_accuracy = numpy.nan_to_num((recall + true_negative_rate) / 2., nan=numpy.nan, posinf=numpy.nan,
                                                 neginf=numpy.nan)
            cohen_kappa = numpy.nan_to_num(2. * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)),
                                           nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            matthews_corr = numpy.nan_to_num((tp * tn - fp * fn) / numpy.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
                                             nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
            pr_auc = numpy.nan_to_num(metrics.average_precision_score(yy, yy_hat, pos_label=1.), nan=numpy.nan,
                                      posinf=numpy.nan, neginf=numpy.nan)
        if len(numpy.unique(yy)) > 1:
            roc_auc = metrics.roc_auc_score(yy, yy_hat, labels=[0., 1.])
        else:
            roc_auc = numpy.nan
        cross_entropy = metrics.log_loss(yy, yy_hat, eps=1e-7, labels=[0., 1.])

        self.log('test_tp' + suffix, tp_norm, reduce_fx=numpy.nanmean)
        self.log('test_tp' + suffix + '_std', tp_norm, reduce_fx=numpy.nanstd)
        self.log('test_fp' + suffix, fp_norm, reduce_fx=numpy.nanmean)
        self.log('test_fp' + suffix + '_std', fp_norm, reduce_fx=numpy.nanstd)
        self.log('test_fn' + suffix, fn_norm, reduce_fx=numpy.nanmean)
        self.log('test_fn' + suffix + '_std', fn_norm, reduce_fx=numpy.nanstd)
        self.log('test_tn' + suffix, tn_norm, reduce_fx=numpy.nanmean)
        self.log('test_tn' + suffix + '_std', tn_norm, reduce_fx=numpy.nanstd)
        self.log('test_false_positive_rate' + suffix, false_positive_rate, reduce_fx=numpy.nanmean)
        self.log('test_false_positive_rate' + suffix + '_std', false_positive_rate, reduce_fx=numpy.nanstd)
        self.log('test_false_negative_rate' + suffix, false_negative_rate, reduce_fx=numpy.nanmean)
        self.log('test_false_negative_rate' + suffix + '_std', false_negative_rate, reduce_fx=numpy.nanstd)
        self.log('test_true_negative_rate' + suffix, true_negative_rate, reduce_fx=numpy.nanmean)
        self.log('test_true_negative_rate' + suffix + '_std', true_negative_rate, reduce_fx=numpy.nanstd)
        self.log('test_negative_predictive_value' + suffix, negative_predictive_value, reduce_fx=numpy.nanmean)
        self.log('test_negative_predictive_value' + suffix + '_std', negative_predictive_value, reduce_fx=numpy.nanstd)
        self.log('test_false_discovery_rate' + suffix, false_discovery_rate, reduce_fx=numpy.nanmean)
        self.log('test_false_discovery_rate' + suffix + '_std', false_discovery_rate, reduce_fx=numpy.nanstd)
        self.log('test_recall' + suffix, recall, reduce_fx=numpy.nanmean)
        self.log('test_recall' + suffix + '_std', recall, reduce_fx=numpy.nanstd)
        self.log('test_precision' + suffix, precision, reduce_fx=numpy.nanmean)
        self.log('test_precision' + suffix + '_std', precision, reduce_fx=numpy.nanstd)
        self.log('test_f1' + suffix, f1, reduce_fx=numpy.nanmean)
        self.log('test_f1' + suffix + '_std', f1, reduce_fx=numpy.nanstd)
        self.log('test_matthews_corr' + suffix, matthews_corr, reduce_fx=numpy.nanmean)
        self.log('test_matthews_corr' + suffix + '_std', matthews_corr, reduce_fx=numpy.nanstd)
        self.log('test_roc_auc' + suffix, roc_auc, reduce_fx=numpy.nanmean)
        self.log('test_roc_auc' + suffix + '_std', roc_auc, reduce_fx=numpy.nanstd)
        self.log('test_pr_auc' + suffix, pr_auc, reduce_fx=numpy.nanmean)
        self.log('test_pr_auc' + suffix + '_std', pr_auc, reduce_fx=numpy.nanstd)
        self.log('test_cross_entropy' + suffix, cross_entropy, reduce_fx=numpy.nanmean)
        self.log('test_cross_entropy' + suffix + '_std', cross_entropy, reduce_fx=numpy.nanstd)
        self.log('test_balanced_accuracy' + suffix, balanced_accuracy, reduce_fx=numpy.nanmean)
        self.log('test_balanced_accuracy' + suffix + '_std', balanced_accuracy, reduce_fx=numpy.nanstd)
        self.log('test_cohen_kappa' + suffix, cohen_kappa, reduce_fx=numpy.nanmean)
        self.log('test_cohen_kappa' + suffix + '_std', cohen_kappa, reduce_fx=numpy.nanstd)

        tn_norm_tot.append(tn_norm)
        fp_norm_tot.append(fp_norm)
        fn_norm_tot.append(fn_norm)
        tp_norm_tot.append(tp_norm)
        false_positive_rate_tot.append(false_positive_rate)
        false_negative_rate_tot.append(false_negative_rate)
        true_negative_rate_tot.append(true_negative_rate)
        negative_predictive_value_tot.append(negative_predictive_value)
        false_discovery_rate_tot.append(false_discovery_rate)
        recall_tot.append(recall)
        precision_tot.append(precision)
        f1_tot.append(f1)
        matthews_corr_tot.append(matthews_corr)
        roc_auc_tot.append(roc_auc)
        pr_auc_tot.append(pr_auc)
        cross_entropy_tot.append(cross_entropy)
        balanced_accuracy_tot.append(balanced_accuracy)
        cohen_kappa_tot.append(cohen_kappa)

        for i in range(1, self.n_classes):
            suffix = "_class" + str(i)
            yy = y[i].flatten()
            yy_hat = y_hat[i, ...].flatten()
            yy_hat_tresholded = y_hat_tresholded[i, ...].flatten()

            tn, fp, fn, tp = metrics.confusion_matrix(yy, yy_hat_tresholded, labels=[0., 1.]).ravel()
            tn_norm = tn / n_elem
            fp_norm = fp / n_elem
            fn_norm = fn / n_elem
            tp_norm = tp / n_elem
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                false_positive_rate = numpy.nan_to_num(fp / (fp + tn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                false_negative_rate = numpy.nan_to_num(fn / (tp + fn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                true_negative_rate = numpy.nan_to_num(tn / (tn + fp), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                negative_predictive_value = numpy.nan_to_num(tn / (tn + fn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                false_discovery_rate = numpy.nan_to_num(fp / (tp + fp), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                recall = numpy.nan_to_num(tp / (tp + fn), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                precision = numpy.nan_to_num(tp / (tp + fp), nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                f1 = numpy.nan_to_num(2. * (precision * recall) / (precision + recall), nan=numpy.nan, posinf=numpy.nan,
                                      neginf=numpy.nan)
                balanced_accuracy = numpy.nan_to_num((recall + true_negative_rate) / 2., nan=numpy.nan, posinf=numpy.nan,
                                                     neginf=numpy.nan)
                cohen_kappa = numpy.nan_to_num(2. * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)),
                                               nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                matthews_corr = numpy.nan_to_num((tp * tn - fp * fn) / numpy.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
                                                 nan=numpy.nan, posinf=numpy.nan, neginf=numpy.nan)
                pr_auc = numpy.nan_to_num(metrics.average_precision_score(yy, yy_hat, pos_label=1.), nan=numpy.nan,
                                          posinf=numpy.nan, neginf=numpy.nan)
            if len(numpy.unique(yy)) > 1:
                roc_auc = metrics.roc_auc_score(yy, yy_hat, labels=[0., 1.])
            else:
                roc_auc = numpy.nan
            cross_entropy = metrics.log_loss(yy, yy_hat, eps=1e-7, labels=[0., 1.])

            self.log('test_tp' + suffix, tp_norm, reduce_fx=numpy.nanmean)
            self.log('test_tp' + suffix + '_std', tp_norm, reduce_fx=numpy.nanstd)
            self.log('test_fp' + suffix, fp_norm, reduce_fx=numpy.nanmean)
            self.log('test_fp' + suffix + '_std', fp_norm, reduce_fx=numpy.nanstd)
            self.log('test_fn' + suffix, fn_norm, reduce_fx=numpy.nanmean)
            self.log('test_fn' + suffix + '_std', fn_norm, reduce_fx=numpy.nanstd)
            self.log('test_tn' + suffix, tn_norm, reduce_fx=numpy.nanmean)
            self.log('test_tn' + suffix + '_std', tn_norm, reduce_fx=numpy.nanstd)
            self.log('test_false_positive_rate' + suffix, false_positive_rate, reduce_fx=numpy.nanmean)
            self.log('test_false_positive_rate' + suffix + '_std', false_positive_rate, reduce_fx=numpy.nanstd)
            self.log('test_false_negative_rate' + suffix, false_negative_rate, reduce_fx=numpy.nanmean)
            self.log('test_false_negative_rate' + suffix + '_std', false_negative_rate, reduce_fx=numpy.nanstd)
            self.log('test_true_negative_rate' + suffix, true_negative_rate, reduce_fx=numpy.nanmean)
            self.log('test_true_negative_rate' + suffix + '_std', true_negative_rate, reduce_fx=numpy.nanstd)
            self.log('test_negative_predictive_value' + suffix, negative_predictive_value, reduce_fx=numpy.nanmean)
            self.log('test_negative_predictive_value' + suffix + '_std', negative_predictive_value, reduce_fx=numpy.nanstd)
            self.log('test_false_discovery_rate' + suffix, false_discovery_rate, reduce_fx=numpy.nanmean)
            self.log('test_false_discovery_rate' + suffix + '_std', false_discovery_rate, reduce_fx=numpy.nanstd)
            self.log('test_recall' + suffix, recall, reduce_fx=numpy.nanmean)
            self.log('test_recall' + suffix + '_std', recall, reduce_fx=numpy.nanstd)
            self.log('test_precision' + suffix, precision, reduce_fx=numpy.nanmean)
            self.log('test_precision' + suffix + '_std', precision, reduce_fx=numpy.nanstd)
            self.log('test_f1' + suffix, f1, reduce_fx=numpy.nanmean)
            self.log('test_f1' + suffix + '_std', f1, reduce_fx=numpy.nanstd)
            self.log('test_matthews_corr' + suffix, matthews_corr, reduce_fx=numpy.nanmean)
            self.log('test_matthews_corr' + suffix + '_std', matthews_corr, reduce_fx=numpy.nanstd)
            self.log('test_roc_auc' + suffix, roc_auc, reduce_fx=numpy.nanmean)
            self.log('test_roc_auc' + suffix + '_std', roc_auc, reduce_fx=numpy.nanstd)
            self.log('test_pr_auc' + suffix, pr_auc, reduce_fx=numpy.nanmean)
            self.log('test_pr_auc' + suffix + '_std', pr_auc, reduce_fx=numpy.nanstd)
            self.log('test_cross_entropy' + suffix, cross_entropy, reduce_fx=numpy.nanmean)
            self.log('test_cross_entropy' + suffix + '_std', cross_entropy, reduce_fx=numpy.nanstd)
            self.log('test_balanced_accuracy' + suffix, balanced_accuracy, reduce_fx=numpy.nanmean)
            self.log('test_balanced_accuracy' + suffix + '_std', balanced_accuracy, reduce_fx=numpy.nanstd)
            self.log('test_cohen_kappa' + suffix, cohen_kappa, reduce_fx=numpy.nanmean)
            self.log('test_cohen_kappa' + suffix + '_std', cohen_kappa, reduce_fx=numpy.nanstd)

            tn_norm_tot.append(tn_norm)
            fp_norm_tot.append(fp_norm)
            fn_norm_tot.append(fn_norm)
            tp_norm_tot.append(tp_norm)
            false_positive_rate_tot.append(false_positive_rate)
            false_negative_rate_tot.append(false_negative_rate)
            true_negative_rate_tot.append(true_negative_rate)
            negative_predictive_value_tot.append(negative_predictive_value)
            false_discovery_rate_tot.append(false_discovery_rate)
            recall_tot.append(recall)
            precision_tot.append(precision)
            f1_tot.append(f1)
            matthews_corr_tot.append(matthews_corr)
            roc_auc_tot.append(roc_auc)
            pr_auc_tot.append(pr_auc)
            cross_entropy_tot.append(cross_entropy)
            balanced_accuracy_tot.append(balanced_accuracy)
            cohen_kappa_tot.append(cohen_kappa)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            tn_norm_tot = numpy.nanmean(tn_norm_tot)
            fp_norm_tot = numpy.nanmean(fp_norm_tot)
            fn_norm_tot = numpy.nanmean(fn_norm_tot)
            tp_norm_tot = numpy.nanmean(tp_norm_tot)
            false_positive_rate_tot = numpy.nanmean(false_positive_rate_tot)
            false_negative_rate_tot = numpy.nanmean(false_negative_rate_tot)
            true_negative_rate_tot = numpy.nanmean(true_negative_rate_tot)
            negative_predictive_value_tot = numpy.nanmean(negative_predictive_value_tot)
            false_discovery_rate_tot = numpy.nanmean(false_discovery_rate_tot)
            recall_tot = numpy.nanmean(recall_tot)
            precision_tot = numpy.nanmean(precision_tot)
            f1_tot = numpy.nanmean(f1_tot)
            matthews_corr_tot = numpy.nanmean(matthews_corr_tot)
            roc_auc_tot = numpy.nanmean(roc_auc_tot)
            pr_auc_tot = numpy.nanmean(pr_auc_tot)
            cross_entropy_tot = numpy.nanmean(cross_entropy_tot)
            balanced_accuracy_tot = numpy.nanmean(balanced_accuracy_tot)
            cohen_kappa_tot = numpy.nanmean(cohen_kappa_tot)

        suffix = "_tot"
        self.log('test_tp' + suffix, tp_norm_tot, reduce_fx=numpy.nanmean)
        self.log('test_tp' + suffix + '_std', tp_norm_tot, reduce_fx=numpy.nanstd)
        self.log('test_fp' + suffix, fp_norm_tot, reduce_fx=numpy.nanmean)
        self.log('test_fp' + suffix + '_std', fp_norm_tot, reduce_fx=numpy.nanstd)
        self.log('test_fn' + suffix, fn_norm_tot, reduce_fx=numpy.nanmean)
        self.log('test_fn' + suffix + '_std', fn_norm_tot, reduce_fx=numpy.nanstd)
        self.log('test_tn' + suffix, tn_norm_tot, reduce_fx=numpy.nanmean)
        self.log('test_tn' + suffix + '_std', tn_norm_tot, reduce_fx=numpy.nanstd)
        self.log('test_false_positive_rate' + suffix, false_positive_rate_tot, reduce_fx=numpy.nanmean)
        self.log('test_false_positive_rate' + suffix + '_std', false_positive_rate_tot, reduce_fx=numpy.nanstd)
        self.log('test_false_negative_rate' + suffix, false_negative_rate_tot, reduce_fx=numpy.nanmean)
        self.log('test_false_negative_rate' + suffix + '_std', false_negative_rate_tot, reduce_fx=numpy.nanstd)
        self.log('test_true_negative_rate' + suffix, true_negative_rate_tot, reduce_fx=numpy.nanmean)
        self.log('test_true_negative_rate' + suffix + '_std', true_negative_rate_tot, reduce_fx=numpy.nanstd)
        self.log('test_negative_predictive_value' + suffix, negative_predictive_value_tot, reduce_fx=numpy.nanmean)
        self.log('test_negative_predictive_value' + suffix + '_std', negative_predictive_value_tot, reduce_fx=numpy.nanstd)
        self.log('test_false_discovery_rate' + suffix, false_discovery_rate_tot, reduce_fx=numpy.nanmean)
        self.log('test_false_discovery_rate' + suffix + '_std', false_discovery_rate_tot, reduce_fx=numpy.nanstd)
        self.log('test_recall' + suffix, recall_tot, reduce_fx=numpy.nanmean)
        self.log('test_recall' + suffix + '_std', recall_tot, reduce_fx=numpy.nanstd)
        self.log('test_precision' + suffix, precision_tot, reduce_fx=numpy.nanmean)
        self.log('test_precision' + suffix + '_std', precision_tot, reduce_fx=numpy.nanstd)
        self.log('test_f1' + suffix, f1_tot, reduce_fx=numpy.nanmean)
        self.log('test_f1' + suffix + '_std', f1_tot, reduce_fx=numpy.nanstd)
        self.log('test_matthews_corr' + suffix, matthews_corr_tot, reduce_fx=numpy.nanmean)
        self.log('test_matthews_corr' + suffix + '_std', matthews_corr_tot, reduce_fx=numpy.nanstd)
        self.log('test_roc_auc' + suffix, roc_auc_tot, reduce_fx=numpy.nanmean)
        self.log('test_roc_auc' + suffix + '_std', roc_auc_tot, reduce_fx=numpy.nanstd)
        self.log('test_pr_auc' + suffix, pr_auc_tot, reduce_fx=numpy.nanmean)
        self.log('test_pr_auc' + suffix + '_std', pr_auc_tot, reduce_fx=numpy.nanstd)
        self.log('test_cross_entropy' + suffix, cross_entropy_tot, reduce_fx=numpy.nanmean)
        self.log('test_cross_entropy' + suffix + '_std', cross_entropy_tot, reduce_fx=numpy.nanstd)
        self.log('test_balanced_accuracy' + suffix, balanced_accuracy_tot, reduce_fx=numpy.nanmean)
        self.log('test_balanced_accuracy' + suffix + '_std', balanced_accuracy_tot, reduce_fx=numpy.nanstd)
        self.log('test_cohen_kappa' + suffix, cohen_kappa_tot, reduce_fx=numpy.nanmean)
        self.log('test_cohen_kappa' + suffix + '_std', cohen_kappa_tot, reduce_fx=numpy.nanstd)

    @staticmethod
    def get_class_weight_fn(weight_type=None):

        eps = 1e-5
        num_of_classes = 2

        def no_weight(tot, pos):
            return 1.0, 1.0

        def linear_weight(tot, pos):
            # pos_w is 1 when the classes are balanced, eps to prevent division by zero
            return (tot / (pos + eps)) / num_of_classes, 1.0

        def constant_weight(tot, pos):
            return conf.params.class_weights[0], conf.params.class_weights[1]

        if weight_type == "linear":
            return linear_weight
        elif weight_type == "precomputed":
            return constant_weight
        return no_weight

    @staticmethod
    def get_loss_function(use_mask=False, weight_type=None):

        if loss_support_mask(conf.params.loss_function):
            def loss_helper(y_hat, y, masked_y, binary_mask):
                pos_w, neg_w = CNNWithGC.get_class_weight_fn(weight_type=weight_type)(binary_mask.sum(), masked_y.sum())
                pos_mask = masked_y * pos_w
                neg_mask = (1.0 - masked_y) * neg_w
                weights = pos_mask + neg_mask
                loss = get_loss_function_by_name(conf.params.loss_function, False)(y_hat, y)

                # mask the loss
                loss = loss * weights.detach()
                return loss

            def loss_with_mask(y_hat, y, binary_mask, weighted_mask):
                # compute the loss using weights to compensate the class imbalance:
                # use the binary mask for pos_w to take into account only revealed labels
                masked_y = y * binary_mask

                loss = loss_helper(y_hat, y, masked_y, binary_mask)
                loss = loss * weighted_mask.detach()

                # Now we have to reduce the loss
                loss = torch.mean(loss)
                return loss

            def loss_without_mask(y_hat, y, binary_mask=None, weighted_mask=None):
                # compute the loss using weights to compensate the class imbalance:
                loss = loss_helper(y_hat, y, y, torch.ones_like(y))

                # Now we have to reduce the loss
                loss = torch.mean(loss)
                return loss
        elif loss_support_ignore_index(conf.params.loss_function):
            def loss_with_mask(y_hat, y, binary_mask, weighted_mask):
                binary_mask = (y * torch.logical_not(torch.gt(binary_mask, 0))).view(y.size(0), 1, -1).detach()
                loss = get_loss_function_by_name(conf.params.loss_function, False)(y_hat, y, ignore_index=binary_mask)
                return loss

            def loss_without_mask(y_hat, y, binary_mask=None, weighted_mask=None):
                loss = get_loss_function_by_name(conf.params.loss_function, False)(y_hat, y)
                return loss
        else:
            def loss_with_mask(y_hat, y, binary_mask, weighted_mask):
                loss = get_loss_function_by_name(conf.params.loss_function, False)(y_hat, y)
                return loss

            def loss_without_mask(y_hat, y, binary_mask=None, weighted_mask=None):
                loss = get_loss_function_by_name(conf.params.loss_function, False)(y_hat, y)
                return loss

        if use_mask:
            return loss_with_mask
        else:
            return loss_without_mask

    def on_train_epoch_start(self):
        self.train_accuracies = []
        self.train_IoUs = []
        self.train_statistics_weights = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.split(x)
        y = [self.split(yy.unsqueeze(1)).squeeze(1) for yy in y]
        y_hat = self(x)

        # In order to reveal only part of the training labels, we generate a mask
        # by sampling a Bernoulli distribution, and use it to mask the loss.
        # We also divide the mask by the fraction of revealed labels in order to restore the loss magnitude
        binary_mask = torch.distributions.bernoulli.Bernoulli(probs=conf.params.reveled_labels).sample(
            sample_shape=y[0].shape).type_as(y[0])
        # Set the weight to 0 in case of no labels considered
        w = torch.nan_to_num(torch.numel(binary_mask) / binary_mask.sum(), nan=0.0)
        mask = binary_mask * w

        # compute the loss
        loss_fn = self.get_loss_function(use_mask=conf.params.reveled_labels < 1.0, weight_type=self.class_weight_type)

        loss = loss_fn(y_hat[:, 0, ...], y[0], binary_mask, mask)
        self.log('train_loss_class0', loss)
        for i in range(1, self.n_classes):
            tmp = loss_fn(y_hat[:, i, ...], y[i], binary_mask, mask)
            self.log('train_loss_class' + str(i), loss)
            loss = loss + tmp
        loss /= self.n_classes

        self.log('train_loss_tot', loss)

        accuracy = self.compute_accuracy(torch.argmax(y_hat, dim=1) == 0, y[0])
        self.log('train_accuracy_class0', accuracy)
        for i in range(1, self.n_classes):
            tmp = self.compute_accuracy(torch.argmax(y_hat, dim=1) == i, y[i])
            self.log('train_accuracy_class' + str(i), tmp)
            accuracy = accuracy + tmp
        accuracy /= self.n_classes

        self.log('train_accuracy_tot', accuracy)
        self.train_accuracies.append(accuracy)

        jaccard_index = BIoU(self.accuracy_threshold)(y_hat[:, 0, ...], y[0])
        self.log('train_IoU_class0', jaccard_index)
        for i in range(1, self.n_classes):
            tmp = BIoU(self.accuracy_threshold)(y_hat[:, i, ...], y[i])
            self.log('train_IoU_class' + str(i), tmp)
            jaccard_index = jaccard_index + tmp
        jaccard_index /= self.n_classes

        self.log('train_IoU', jaccard_index)
        self.train_IoUs.append(jaccard_index)

        tmp = 0
        for i in y:
            tmp = tmp + i.bool().sum()
        self.train_statistics_weights.append(tmp)

        return loss

    def on_training_epoch_end(self):
        weighted_accuracy = weighted_mean(self.train_accuracies, self.train_statistics_weights)
        weighted_accuracy_std = weighted_std(self.train_accuracies, self.train_statistics_weights, weighted_accuracy)
        weighted_IoU = weighted_mean(self.train_IoUs, self.train_statistics_weights)
        weighted_IoU_std = weighted_std(self.train_IoUs, self.train_statistics_weights, weighted_IoU)
        self.log('train_weighted_accuracy', weighted_accuracy)
        self.log('train_weighted_accuracy_std', weighted_accuracy_std)
        self.log('train_weighted_IoU', weighted_IoU)
        self.log('train_weighted_IoU_std', weighted_IoU_std)

    def on_validation_epoch_start(self):
        self.val_accuracies = []
        self.val_IoUs = []
        self.val_statistics_weights = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.split(x)
        y_hat = self(x)
        y_hat = self.recombine(y_hat)

        # compute the loss
        loss_fn = self.get_loss_function(weight_type=self.class_weight_type)

        loss = loss_fn(y_hat[:, 0, ...], y[0])
        self.log('val_loss_class0', loss)
        for i in range(1, self.n_classes):
            tmp = loss_fn(y_hat[:, i, ...], y[i])
            self.log('val_loss_class' + str(i), loss)
            loss = loss + tmp
        loss /= self.n_classes

        self.log('val_loss_tot', loss)

        accuracy = self.compute_accuracy(torch.argmax(y_hat, dim=1) == 0, y[0])
        self.log('val_accuracy_class0', accuracy)
        for i in range(1, self.n_classes):
            tmp = self.compute_accuracy(torch.argmax(y_hat, dim=1) == i, y[i])
            self.log('val_accuracy_class' + str(i), tmp)
            accuracy = accuracy + tmp
        accuracy /= self.n_classes

        self.log('val_accuracy_tot', accuracy)
        self.val_accuracies.append(accuracy)

        jaccard_index = BIoU(self.accuracy_threshold)(y_hat[:, 0, ...], y[0])
        self.log('val_IoU_class0', jaccard_index)
        for i in range(1, self.n_classes):
            tmp = BIoU(self.accuracy_threshold)(y_hat[:, i, ...], y[i])
            self.log('val_IoU_class' + str(i), tmp)
            jaccard_index = jaccard_index + tmp
        jaccard_index /= self.n_classes

        self.log('val_IoU_tot', jaccard_index)
        self.log("val_IoU_tot_std", jaccard_index, reduce_fx=std_for_list_of_tensors)
        self.val_IoUs.append(jaccard_index)

        tmp = 0
        for i in y:
            tmp = tmp + i.bool().sum()
        self.val_statistics_weights.append(tmp)

    def on_validation_epoch_end(self):
        weighted_accuracy = weighted_mean(self.val_accuracies, self.val_statistics_weights)
        weighted_accuracy_std = weighted_std(self.val_accuracies, self.val_statistics_weights, weighted_accuracy)
        weighted_IoU = weighted_mean(self.val_IoUs, self.val_statistics_weights)
        weighted_IoU_std = weighted_std(self.val_IoUs, self.val_statistics_weights, weighted_IoU)
        self.log('val_weighted_accuracy', weighted_accuracy)
        self.log('val_weighted_accuracy_std', weighted_accuracy_std)
        self.log('val_weighted_IoU', weighted_IoU)
        self.log('val_weighted_IoU_std', weighted_IoU_std)

    def on_test_epoch_start(self):
        self.saved_images = 0

        self.test_accuracies = []
        self.test_IoUs = []
        self.test_statistics_weights = []
        
    def split(self, x):
        overlap = 32
        size = 256
        
        x = torch.nn.functional.pad(x, (overlap, overlap, overlap, overlap))

        x = x.unfold(3, size + overlap * 2, size).unfold(2, size + overlap * 2, size)

        x = x.permute(0, 2, 3, 1, 4, 5)

        return x.flatten(0, 2).contiguous()
    
    def recombine(self, x):
        overlap = 32
        size = 256
        channels = 4
        bs = 1
        H = 1024
        W = 2048
        
        x = x.unflatten(0, (bs, H // size, W // size)).permute(0, 3, 1, 2, 5, 4)[:, :, :, :, overlap:-overlap, overlap:-overlap]
        
        x = x.contiguous().view(bs, channels, -1, size * size).permute(0, 1, 3, 2).contiguous().view(bs, channels * size * size, -1)
        
        x = torch.nn.functional.fold(x, output_size=(H, W), kernel_size=size, stride=size)

        return x.contiguous()
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.split(x)
        y_hat_split = self(x)
        y_hat = self.recombine(y_hat_split)

        # compute the loss
        loss_fn = self.get_loss_function(weight_type=self.class_weight_type)

        # log the test metrics
        loss = loss_fn(y_hat[:, 0, ...], y[0])
        self.log('test_loss_class0', loss)
        for i in range(1, self.n_classes):
            tmp = loss_fn(y_hat[:, i, ...], y[i])
            self.log('test_loss_class' + str(i), tmp)
            loss = loss + tmp
        loss /= self.n_classes

        self.log('test_loss_tot', loss)
        self.log('test_loss_tot_std', loss, reduce_fx=std_for_list_of_tensors)

        accuracy = self.compute_accuracy(torch.argmax(y_hat, dim=1) == 0, y[0])
        self.log('test_accuracy_class0', accuracy)
        for i in range(1, self.n_classes):
            tmp = self.compute_accuracy(torch.argmax(y_hat, dim=1) == i, y[i])
            self.log('test_accuracy_class' + str(i), tmp)
            accuracy = accuracy + tmp
        accuracy /= self.n_classes

        self.log('test_accuracy_tot', accuracy)
        self.log('test_accuracy_tot_std', accuracy, reduce_fx=std_for_list_of_tensors)
        self.test_accuracies.append(accuracy)

        jaccard_index = BIoU(self.accuracy_threshold)(y_hat[:, 0, ...], y[0])
        self.log('test_IoU_class0', jaccard_index)
        for i in range(1, self.n_classes):
            tmp = BIoU(self.accuracy_threshold)(y_hat[:, i, ...], y[i])
            self.log('test_IoU_class' + str(i), tmp)
            jaccard_index = jaccard_index + tmp
        jaccard_index /= self.n_classes

        self.log('test_IoU_tot', jaccard_index)
        self.log("test_IoU_tot_std", jaccard_index, reduce_fx=std_for_list_of_tensors)
        self.test_IoUs.append(jaccard_index)

        tmp = 0
        for i in y:
            tmp = tmp + i.bool().sum()
        self.test_statistics_weights.append(tmp)

        # compute the output segmentation
        y_hat_prob = y_hat.squeeze(0)
        #seg_per_class = (y_hat_prob > self.accuracy_threshold).bool()
        seg_per_class = torch.zeros_like(y_hat_prob).scatter_(0, torch.argmax(y_hat_prob, dim=0).unsqueeze(0), 1.)
        seg_per_class = seg_per_class.detach().cpu()
        y_hat_prob = y_hat_prob.detach().cpu()
        
        self.extra_test_metrics(y_hat_prob, y)

        seg_multiclass = self.compute_multiclass_image(y_hat.squeeze(0).detach().cpu(), y_hat_prob)

        tmp = torch.cat(y, 0).detach().cpu()
        seg_multiclass_gt = self.compute_multiclass_image(tmp, tmp)

        # compute the number of distinct figures
        _, num = measure.label(seg_per_class[0].int().numpy(), return_num=True)
        for i in range(1, self.n_classes):
            _, tmp = measure.label(seg_per_class[i].int().numpy(), return_num=True)
            num += tmp
        self.log("test_num_distinct_figures", num)
        self.log("test_num_distinct_figures_std", num, reduce_fx=std_for_list_of_tensors)

        # denormalize the input image in order to view it
        mean = torch.tensor(self.normalize_params['mean']).type_as(x).view(-1, 1, 1)
        std = torch.tensor(self.normalize_params['std']).type_as(x).view(-1, 1, 1)
        x = x * std + mean
        x = (x * 255).type(torch.uint8)
        x = torch.squeeze(x, 0)

        #yy_1 = self.split(y[1])
        #yy_2 = self.split(y[2])
        #for b in range(y_hat_split.size(0))
            ## merge the images and log them
            #if self.saved_images < conf.images_to_log:
                #if conf.log_debug_image and self.use_graph_cut:
                    #fig, axarr = plt.subplots(3, 4, figsize=(10, 6.6))
                    #out_plot = axarr[1][0].imshow(self.last_weights_row[b].detach().cpu().squeeze())
                    #axarr[1][0].axis('off')
                    #axarr[1][0].set_title("Row weights")
                    #fig.colorbar(out_plot, ax=axarr[1][0])
                    #out_plot = axarr[1][1].imshow(self.last_weights_col[b].detach().cpu().squeeze())
                    #axarr[1][1].axis('off')
                    #axarr[1][1].set_title("Column weights")
                    #fig.colorbar(out_plot, ax=axarr[1][1])
                    #out_plot = axarr[1][2].imshow(self.last_pre_cut_image[b, 1, ...].detach().cpu())
                    #axarr[1][2].axis('off')
                    #axarr[1][2].set_title("Pre cut output (class 0)")
                    #fig.colorbar(out_plot, ax=axarr[1][2])
                    #out_plot = axarr[1][3].imshow(self.last_pre_cut_image[b, 2, ...].detach().cpu())
                    #axarr[1][3].axis('off')
                    #axarr[1][3].set_title("Pre cut output (class 1)")
                    #fig.colorbar(out_plot, ax=axarr[1][3])

                    #axarr[2][0].imshow(
                        #transforms.ToPILImage()(seg_per_class[1, ...].type(torch.uint8) * 255).convert('1', dither=Image.NONE))
                    #axarr[2][0].axis('off')
                    #axarr[2][0].set_title("Output (class 0)")
                    #axarr[2][1].imshow(
                        #transforms.ToPILImage()(seg_per_class[2, ...].type(torch.uint8) * 255).convert('1', dither=Image.NONE))
                    #axarr[2][1].axis('off')
                    #axarr[2][1].set_title("Output (class 1)")
                    #axarr[2][2].imshow(transforms.ToPILImage()(yy_1[b].detach().cpu()).convert('1', dither=Image.NONE))
                    #axarr[2][2].axis('off')
                    #axarr[2][2].set_title("Ground Truth (class 0)")
                    #axarr[2][3].imshow(transforms.ToPILImage()(yy_2[b].detach().cpu()).convert('1', dither=Image.NONE))
                    #axarr[2][3].axis('off')
                    #axarr[2][3].set_title("Ground Truth (class 1)")

                    #axarr = axarr[0]
                #else:
                    #fig, axarr = plt.subplots(1, 3, figsize=(7.5, 2.2))
                #axarr[0].imshow(transforms.ToPILImage()(x[b].detach().cpu()).convert("RGB"))
                #axarr[0].axis('off')
                #axarr[0].set_title("Input")
                #axarr[1].imshow(seg_multiclass)
                #axarr[1].axis('off')
                #axarr[1].set_title("Output")
                #axarr[2].imshow(seg_multiclass_gt)
                #axarr[2].axis('off')
                #axarr[2].set_title("Ground Truth")
                #plt.tight_layout()
                ## fig.show()
                #fig.savefig("/exp/ti/" + str(self.saved_images) + ".png")
                #self.logger.log_image("Test images results", fig)
                #plt.close('all')
                #self.saved_images += 1

    def on_test_epoch_end(self):
        weighted_accuracy = weighted_mean(self.test_accuracies, self.test_statistics_weights)
        weighted_accuracy_std = weighted_std(self.test_accuracies, self.test_statistics_weights, weighted_accuracy)
        weighted_IoU = weighted_mean(self.test_IoUs, self.test_statistics_weights)
        weighted_IoU_std = weighted_std(self.test_IoUs, self.test_statistics_weights, weighted_IoU)
        self.log('test_weighted_accuracy', weighted_accuracy)
        self.log('test_weighted_accuracy_std', weighted_accuracy_std)
        self.log('test_weighted_IoU', weighted_IoU)
        self.log('test_weighted_IoU_std', weighted_IoU_std)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, conf.params.optimizer)(filter(lambda p: p.requires_grad, self.parameters()),
                                                                lr=self.learning_rate)
        return optimizer

    # def train_dataloader(self):
    #   return DataLoader(train_ds, batch_size=self.batch_size, num_workers=conf.num_of_workers,
    #       shuffle=conf.params.shuffle_training_set)


# Generate the model
model = CNNWithGC(conf.params.learning_rate, conf.params.batch_size,
                  conf.params.n_classes,
                  conf.params.enable_graphcut, conf.params.use_class_weights,
                  conf.params.segmentation_architecture,
                  conf.params.encoder, conf.params.encoder_weights, conf.params.encoder_depth,
                  normalize_params,
                  accuracy_threshold=conf.params.accuracy_threshold,
                  skip_connection=conf.params.skip_connection_at_graphcut,
                  use_isotonic_regression=conf.params.refine_with_isotonic_regression,
                  max_tv_iters=conf.params.max_tv_iters,
                  num_workers=conf.num_of_workers,
                  use_multithreading_for_graphcut=conf.params.use_multiple_threads_for_graph_cut)

# Create the trainer with the relative callbacks
neptune_logger = NeptuneLogger(params=dict(conf.params),
                               experiment_name=EXPERIMENT_NAME,
                               tags=EXPERIMENT_TAGS,
                               # offline_mode=True,
                               close_after_fit=False)

callbacks = []

checkpointCallback = pl.callbacks.ModelCheckpoint(
    dirpath=os.path.join(os.environ['STORAGE_DIR'],
                         EXPERIMENT_NAME + "(" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ")", "ckpt"),
    filename='{epoch}-{val_loss:.2f}',
    monitor=conf.params.monitored_quantity_for_checkpoints,
    save_top_k=1,
    mode=get_monitor_mode(conf.params.monitored_quantity_for_checkpoints),
    every_n_val_epochs=1)

callbacks.append(checkpointCallback)

if conf.save_debug_checkpoint:
    checkpointCallback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.environ['STORAGE_DIR'],
                             EXPERIMENT_NAME + "(" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ")_debug", "ckpt"),
        filename='{epoch}-{val_loss:.2f}',
        monitor=conf.params.monitored_quantity_for_checkpoints,
        save_top_k=-1,
        mode=get_monitor_mode(conf.params.monitored_quantity_for_checkpoints),
        every_n_val_epochs=5)

    callbacks.append(checkpointCallback)

# if the fine tuning is enabled create the relative callback
if conf.params.num_epochs_before_fine_tuning is not None:
    class TransferLearningAndFineTuningCallback(BaseFinetuning):
        def __init__(self, unfreeze_at_epoch, encoder_lr_during_finetuning):
            super().__init__()
            self._unfreeze_at_epoch = unfreeze_at_epoch
            self.encoder_lr_during_finetuning = encoder_lr_during_finetuning

        def freeze_before_training(self, pl_module):
            self.freeze(pl_module.encoder, train_bn=False)

        def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
            if current_epoch == self._unfreeze_at_epoch:
                self.unfreeze_and_add_param_group(modules=pl_module.encoder, lr=self.encoder_lr_during_finetuning,
                                                  optimizer=optimizer, train_bn=False)


    callbacks.append(TransferLearningAndFineTuningCallback(conf.params.num_epochs_before_fine_tuning,
                                                           conf.params.encoder_lr_during_finetuning))

trainer = pl.Trainer(max_epochs=conf.params.max_epochs,
                     callbacks=callbacks,
                     logger=neptune_logger,
                     weights_summary='full',
                     check_val_every_n_epoch=1,
                     track_grad_norm=conf.track_grad_norm,
                     gradient_clip_val=conf.params.gradient_clip_value,
                     limit_train_batches=conf.params.limit_train_batches,
                     gpus=1,
                     # auto_scale_batch_size='binsearch',
                     # auto_lr_find=True,
                     # limit_train_batches=10,
                     # limit_val_batches=1,
                     # limit_test_batches=50,
                     deterministic=True)

# trainer.tune(model)

# Train the model
trainer.fit(model, train_loader, val_loader)
neptune_logger.experiment.log_text("path_to_best_checkpoint", trainer.checkpoint_callback.best_model_path)

# Reload the best checkpoint
model = CNNWithGC.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
model.freeze()

# Evaluate the model on the test set:
#   Save a copy of the input image, the output of the nn
#   the thresholded output and the ground truth;
#   moreover compute the number of distinct figures in each output
trainer.test(model, test_loader)
neptune_logger.experiment.stop()
