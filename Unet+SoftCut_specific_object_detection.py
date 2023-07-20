import os
import sys
from datetime import datetime

import albumentations as albm
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
from torch.utils.data import DataLoader
from torch_submod.graph_cuts import TotalVariation2dWeighted
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from softcut.SoftCut_gpu import SoftGraphCut
from datasets_dic import get_dataset_by_name_and_splits
from loss_dic import get_loss_function_by_name, loss_support_mask, loss_support_ignore_index
from utils import get_monitor_mode, get_preprocessing_params, BIoUWithLogits, std_for_list_of_tensors, weighted_mean, weighted_std

NUM_OF_WORKERS = 3

VERSION_NUMBER = 1

# Get configuration from file and cli
base_conf = OmegaConf.load(os.path.join("config", "Unet+submodular_specific_object_detection.yml"))
cli_conf = OmegaConf.from_cli()
helper_conf = OmegaConf.create({"num_of_workers": NUM_OF_WORKERS, "extra_tags": []})
# Merge configurations and extract some optional parameters
conf = OmegaConf.merge(helper_conf, base_conf, cli_conf)

# Generate the experiment name and tags
EXPERIMENT_NAME = "general_CNN(" + conf.params.segmentation_architecture + "+" + conf.params.encoder + ")" + (
    "_SoftCut_apprx" if conf.params.enable_graphcut else "") + "_specific_object_detection"
EXPERIMENT_TAGS = [EXPERIMENT_NAME]
EXPERIMENT_TAGS.extend(conf.extra_tags)

# Set configuration immutable
OmegaConf.set_readonly(conf, True)
OmegaConf.set_struct(conf, True)

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
        albm.PadIfNeeded(256, 256),
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
        albm.PadIfNeeded(256, 256),
        # albm.CenterCrop(256, 256),
        albm.Normalize(**normalize_params),
        ToTensorV2()
    ]
)

EXPERIMENT_TAGS.append("no_augmentation")

# Define the Dataset and the DataLoader
train_ds, val_ds, test_ds = get_dataset_by_name_and_splits(conf.params.dataset, os.environ['PRIVATE_DATASET_DIR'],
                                                           conf.params.ds_splits,
                                                           tiling_factor=conf.params.tiling_factor,
                                                           training_augmentation=train_transform,
                                                           validation_augmentation=train_transform,
                                                           testing_augmentation=test_transform)

train_loader = DataLoader(train_ds, batch_size=conf.params.batch_size, num_workers=conf.num_of_workers,
                          shuffle=conf.params.shuffle_training_set)
val_loader = DataLoader(val_ds, batch_size=conf.params.batch_size, num_workers=conf.num_of_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=conf.num_of_workers)


# Define the LightningModule that contains the NN
class CNNWithGC(pl.LightningModule):
    def __init__(self, lr, batch_size,
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
                                                     in_channels=4,
                                                     classes=1,
                                                     **kwargs)
        self.encoder = seg_model.encoder
        self.decoder = seg_model.decoder
        self.segmentation_head = seg_model.segmentation_head

        if self.use_graph_cut:
            self.gc = SoftGraphCut((256, 256))
            print("GC enabled")
            
            seg_model_2 = getattr(smp, segmentation_model)(encoder_name=encoder_name,
                                                           encoder_weights=encoder_weights,
                                                           encoder_depth=encoder_depth,
                                                           in_channels=4,
                                                           classes=1,
                                                           **kwargs)
            self.weights_decoder = seg_model_2.decoder
            self.weights_head = seg_model_2.segmentation_head

        self.learning_rate = lr
        self.batch_size = batch_size
        self.accuracy_threshold = accuracy_threshold
        img_input_example, _, point_input_example = train_ds.__getitem__(0)
        img_input_example = img_input_example.unsqueeze(0)
        point_input_example = torch.tensor(point_input_example).type_as(img_input_example).unsqueeze(0)
        self.example_input_array = (img_input_example, point_input_example)
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

    def forward(self, x, p):
        p = p.unsqueeze(1)
        input_tensor = torch.cat((x, p), 1)
        encoding = self.encoder(input_tensor)
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
            # (use the softplus instead of exp for numerical stability) -> no
            # (use relu because seems to perform the same way but it's faster)
            weights_row = torch.nn.Softplus()(weights_row)
            weights_col = torch.nn.Softplus()(weights_col)

            if conf.log_debug_image and self.use_graph_cut:
                self.last_pre_cut_image = pixel_scores
                self.last_weights_row = weights_row
                self.last_weights_col = weights_col

            if self.skip_connection:
                identity = pixel_scores
            pixel_scores = self.gc(pixel_scores, weights_col, weights_row)
            if self.skip_connection:
                pixel_scores = pixel_scores + identity
                
        # remove channel dimensions to match the mask shape
        pixel_scores = torch.squeeze(pixel_scores, 1)

        return pixel_scores

    def compute_accuracy_on_positive_class_with_logits(self, y_hat, y):
        y_hat_prob = torch.sigmoid(y_hat)
        y_bool = y.bool()
        thresholded = torch.logical_and((y_hat_prob > self.accuracy_threshold).bool() == y_bool, y_bool)
        positive_tot = y_bool.sum()
        return thresholded.sum() / positive_tot, positive_tot / torch.numel(thresholded)

    def compute_accuracy_with_logits(self, y_hat, y):
        y_hat_prob = torch.sigmoid(y_hat)
        thresholded = (y_hat_prob > self.accuracy_threshold).bool() == y.bool()
        return thresholded.sum() / torch.numel(thresholded)

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
                loss = get_loss_function_by_name(conf.params.loss_function)(y_hat, y)

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
                loss = get_loss_function_by_name(conf.params.loss_function)(y_hat, y, ignore_index=binary_mask)
                return loss

            def loss_without_mask(y_hat, y, binary_mask=None, weighted_mask=None):
                loss = get_loss_function_by_name(conf.params.loss_function)(y_hat, y)
                return loss
        else:
            def loss_with_mask(y_hat, y, binary_mask, weighted_mask):
                loss = get_loss_function_by_name(conf.params.loss_function)(y_hat, y)
                return loss

            def loss_without_mask(y_hat, y, binary_mask=None, weighted_mask=None):
                loss = get_loss_function_by_name(conf.params.loss_function)(y_hat, y)
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
        x, y, p = batch
        p = p.type_as(x)

        y_hat = self(x, p)

        # In order to reveal only part of the training labels, we generate a mask
        # by sampling a Bernoulli distribution, and use it to mask the loss.
        # We also divide the mask by the fraction of revealed labels in order to restore the loss magnitude
        binary_mask = torch.distributions.bernoulli.Bernoulli(probs=conf.params.reveled_labels).sample(
            sample_shape=y.shape).type_as(y)
        # Set the weight to 0 in case of no labels considered
        w = torch.nan_to_num(torch.numel(binary_mask) / binary_mask.sum(), nan=0.0)
        mask = binary_mask * w

        # compute the loss
        loss_fn = self.get_loss_function(use_mask=conf.params.reveled_labels < 1.0, weight_type=self.class_weight_type)
        loss = loss_fn(y_hat, y, binary_mask, mask)

        self.log('train_loss', loss)

        accuracy = self.compute_accuracy_with_logits(y_hat, y)
        self.log('train_accuracy', accuracy)
        self.train_accuracies.append(accuracy)

        jaccard_index = BIoUWithLogits(self.accuracy_threshold)(y_hat, y)
        self.log('train_IoU', jaccard_index)
        self.train_IoUs.append(jaccard_index)

        self.train_statistics_weights.append(y.bool().sum())

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
        x, y, p = batch
        p = p.type_as(x)
        
        y_hat = self(x, p)

        # compute the loss
        loss_fn = self.get_loss_function(weight_type=self.class_weight_type)
        loss = loss_fn(y_hat, y)

        self.log('val_loss', loss)
        accuracy = self.compute_accuracy_with_logits(y_hat, y)

        self.log('val_accuracy', accuracy)
        self.val_accuracies.append(accuracy)

        jaccard_index = BIoUWithLogits(self.accuracy_threshold)(y_hat, y)
        self.log('val_IoU', jaccard_index)
        self.log("val_IoU_std", jaccard_index, reduce_fx=std_for_list_of_tensors)
        self.val_IoUs.append(jaccard_index)

        self.val_statistics_weights.append(y.bool().sum())

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

    def test_step(self, batch, batch_idx):
        x, y, p = batch
        p = p.type_as(x)
        
        y_hat = self(x, p)

        # compute the loss
        loss_fn = self.get_loss_function(weight_type=self.class_weight_type)
        loss = loss_fn(y_hat, y)

        # log the test metrics
        self.log('test_loss', loss)
        self.log('test_loss_std', loss, reduce_fx=std_for_list_of_tensors)

        accuracy = self.compute_accuracy_with_logits(y_hat, y)
        self.log('test_accuracy', accuracy)
        self.log('test_accuracy_std', accuracy, reduce_fx=std_for_list_of_tensors)
        self.test_accuracies.append(accuracy)

        jaccard_index = BIoUWithLogits(self.accuracy_threshold)(y_hat, y)
        self.log('test_IoU', jaccard_index)
        self.log("test_IoU_std", jaccard_index, reduce_fx=std_for_list_of_tensors)
        self.test_IoUs.append(jaccard_index)

        self.test_statistics_weights.append(y.bool().sum())

        # compute the output segmentation
        y_hat_prob = torch.sigmoid(y_hat)
        seg = (y_hat_prob > self.accuracy_threshold).bool()
        seg = seg.detach().cpu()

        # compute the number of distinct figures
        _, num = measure.label(seg.int().numpy(), return_num=True)
        self.log("test_num_distinct_figures", num)
        self.log("test_num_distinct_figures_std", num, reduce_fx=std_for_list_of_tensors)

        # denormalize the input image in order to view it
        mean = torch.tensor(self.normalize_params['mean']).type_as(x).view(-1, 1, 1)
        std = torch.tensor(self.normalize_params['std']).type_as(x).view(-1, 1, 1)
        x = x * std + mean
        x = (x * 255).type(torch.uint8)
        x = torch.squeeze(x, 0)

        # merge the images and log them
        if self.saved_images < conf.images_to_log:
            if conf.log_debug_image and self.use_graph_cut:
                fig, axarr = plt.subplots(2, 4, figsize=(10, 4.4))
                out_plot = axarr[1][0].imshow(self.last_weights_row.detach().cpu().squeeze())
                axarr[1][0].axis('off')
                axarr[1][0].set_title("Row weights")
                fig.colorbar(out_plot, ax=axarr[1][0])
                out_plot = axarr[1][1].imshow(self.last_weights_col.detach().cpu().squeeze())
                axarr[1][1].axis('off')
                axarr[1][1].set_title("Column weights")
                fig.colorbar(out_plot, ax=axarr[1][1])
                out_plot = axarr[1][2].imshow(self.last_pre_cut_image.detach().cpu().squeeze())
                axarr[1][2].axis('off')
                axarr[1][2].set_title("Pre cut output")
                fig.colorbar(out_plot, ax=axarr[1][2])
                axarr[1][3].axis('off')
                axarr = axarr[0]
            else:
                fig, axarr = plt.subplots(1, 4, figsize=(10, 2.2))
            axarr[0].imshow(transforms.ToPILImage()(x.detach().cpu()).convert("RGB"))
            axarr[0].axis('off')
            axarr[0].set_title("Input")
            out_plot = axarr[1].imshow(y_hat.detach().cpu().squeeze())
            axarr[1].axis('off')
            axarr[1].set_title("Output")
            fig.colorbar(out_plot, ax=axarr[1])
            axarr[2].imshow(transforms.ToPILImage()(seg.type(torch.uint8) * 255).convert('1', dither=Image.NONE))
            axarr[2].axis('off')
            axarr[2].set_title("Output (thresholded)")
            axarr[3].imshow(transforms.ToPILImage()(y.detach().cpu()).convert('1', dither=Image.NONE))
            axarr[3].axis('off')
            axarr[3].set_title("Ground Truth")
            plt.tight_layout()
            # fig.show()
            self.logger.log_image("Test images results", fig)
            plt.close('all')
            self.saved_images += 1

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
                     # limit_train_batches=1,
                     # limit_val_batches=1,
                     # limit_test_batches=5,
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
