import os
from datetime import datetime

import albumentations as albm
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from numpy import mean, std
from omegaconf import OmegaConf
from pytorch_lightning.loggers.neptune import NeptuneLogger
from skimage import measure
from torch import nn
from torch.nn import functional as f
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from datasets_dic import get_dataset_by_name
from utils import get_monitor_mode, get_min_hw, SubsetWithTransformations, BIoUWithLogits, std_for_list_of_tensors

NUM_OF_WORKERS = 2

EXPERIMENT_NAME = "submodular_nips17_CNN"
EXPERIMENT_TAGS = [EXPERIMENT_NAME]

# Get configuration from file and cli
base_conf = OmegaConf.load(os.path.join("config", EXPERIMENT_NAME + ".yml"))
cli_conf = OmegaConf.from_cli()
# Override default locale value of NUM_OF_WORKERS
if not OmegaConf.is_none(cli_conf, "num_of_workers"):
    NUM_OF_WORKERS = cli_conf.pop("num_of_workers")
# Check if no custom parameters are added
if cli_conf.is_empty():
    EXPERIMENT_TAGS.append("as_in_the_paper")
# Merge configurations and extract some optional parameters
conf = OmegaConf.merge(base_conf, cli_conf)
if not OmegaConf.is_none(conf, "extra_tags"):
    EXPERIMENT_TAGS.extend(conf.extra_tags)
if OmegaConf.is_none(conf, "num_of_workers"):
    conf.num_of_workers = NUM_OF_WORKERS
# Set configuration immutable
OmegaConf.set_readonly(conf, True)
OmegaConf.set_struct(conf, True)

# Set the global seed for the RNGs
pl.seed_everything(conf.params.seed)

# Define the Dataset and the DataLoader
ds = get_dataset_by_name(conf.params.dataset, os.environ['PRIVATE_DATASET_DIR'])
train_ds, val_ds, test_ds = random_split(ds, conf.params.ds_splits)

if conf.params.crop_images_to_min_during_training:
    train_ds = SubsetWithTransformations(train_ds, transforms.CenterCrop(get_min_hw(train_ds)))

train_loader = DataLoader(train_ds, batch_size=conf.params.batch_size, num_workers=conf.num_of_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=conf.num_of_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=conf.num_of_workers)


# Define the LightningModule that contains the NN
class CNN(pl.LightningModule):

    def __init__(self, lr, batch_size):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 1, 3, padding=1))
        self.learning_rate = lr
        self.batch_size = batch_size
        self.example_input_array = torch.unsqueeze(train_ds.__getitem__(0)[0], 0)
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        segmentation = self.cnn(x)
        return segmentation

    def predict_segmentation(self, x):
        y = self(x)
        y = torch.sigmoid(y)
        return y

    @staticmethod
    def compute_accuracy_with_logits(y_hat, y):
        y_hat_prob = torch.sigmoid(y_hat)
        thresholded = (y_hat_prob > conf.params.accuracy_threshold).bool() == y.bool()
        return thresholded.sum() / torch.numel(thresholded)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = getattr(f, conf.params.loss_function)(y_hat, y, reduction='none')

        # In order to reveal only part of the training labels, we generate a mask
        # by sampling a Bernoulli distribution, and use it to mask the loss.
        # We also divide the mask by the fraction of revealed labels in order to restore the loss magnitude
        mask = torch.distributions.bernoulli.Bernoulli(probs=conf.params.reveled_labels).sample(sample_shape=y.shape).type_as(y)
        mask = mask * torch.numel(mask) / mask.sum()
        loss = loss * mask

        # Now we have to reduce the loss
        loss = torch.mean(loss)

        self.log('train_loss', loss)
        accuracy = CNN.compute_accuracy_with_logits(y_hat, y)
        self.log('train_accuracy', accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = getattr(f, conf.params.loss_function)(y_hat, y)
        self.log('val_loss', loss)
        accuracy = CNN.compute_accuracy_with_logits(y_hat, y)
        self.log('val_accuracy', accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = getattr(f, conf.params.loss_function)(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_loss_std', loss, reduce_fx=std_for_list_of_tensors)
        accuracy = CNN.compute_accuracy_with_logits(y_hat, y)
        self.log('test_accuracy', accuracy)
        
        jaccard_index = BIoUWithLogits(conf.params.accuracy_threshold)(y_hat, y)
        self.log('test_IoU', jaccard_index)
        self.log("test_IoU_std", jaccard_index, reduce_fx=std_for_list_of_tensors)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, conf.params.optimizer)(self.parameters(), lr=self.learning_rate)
        return optimizer


# Generate the model
model = CNN(conf.params.learning_rate, conf.params.batch_size)

# Create the trainer with the relative callbacks
neptune_logger = NeptuneLogger(params=dict(conf.params),
                               experiment_name=EXPERIMENT_NAME,
                               tags=EXPERIMENT_TAGS
                               # offline_mode=True,
                               close_after_fit=False)

checkpointCallback = pl.callbacks.ModelCheckpoint(
    dirpath=os.path.join(os.environ['STORAGE_DIR'],
                         EXPERIMENT_NAME + "(" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ")", "ckpt"),
    filename='{epoch}-{val_loss:.2f}',
    monitor=conf.params.monitored_quantity_for_checkpoints,
    save_top_k=1,
    mode=get_monitor_mode(conf.params.monitored_quantity_for_checkpoints),
    period=1)

trainer = pl.Trainer(max_epochs=conf.params.max_epochs,
                     callbacks=[checkpointCallback],
                     logger=neptune_logger,
                     weights_summary='full',
                     check_val_every_n_epoch=1,
                     gpus=1,
                     deterministic=True)

# Train the model
trainer.fit(model, train_loader, val_loader)
model.freeze()
neptune_logger.experiment.log_text("path_to_best_checkpoint", trainer.checkpoint_callback.best_model_path)

# Reload the best checkpoint
model = CNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

# Evaluate the model on the test set
trainer.test(model, test_loader)

# Save a copy of the input image, the output of the nn
# the thresholded output and the ground truth;
# moreover compute the number of distinct figures in each output
num_of_figures = []
for imgs in test_loader:
    x, y = imgs
    out = model.predict_segmentation(model.transfer_batch_to_device(x)).detach().cpu()
    out = torch.squeeze(out, 0)
    seg = out > conf.params.accuracy_threshold
    _, num = measure.label(seg.int().numpy(), return_num=True)
    num_of_figures.append(num)

    fig, axarr = plt.subplots(1, 4, figsize=(10, 2.2))
    axarr[0].imshow(transforms.ToPILImage()(torch.squeeze(x, 0)).convert("RGB"))
    axarr[0].axis('off')
    axarr[0].set_title("Input")
    axarr[1].imshow(transforms.ToPILImage()(out))
    axarr[1].axis('off')
    axarr[1].set_title("Output")
    axarr[2].imshow(transforms.ToPILImage()(seg.type(torch.uint8) * 255).convert("1"))
    axarr[2].axis('off')
    axarr[2].set_title("Output (thresholded)")
    axarr[3].imshow(transforms.ToPILImage()(torch.squeeze(y, 0)).convert("1"))
    axarr[3].axis('off')
    axarr[3].set_title("Ground Truth")
    plt.tight_layout()
    neptune_logger.log_image("Test images results", fig)
    plt.close('all')
neptune_logger.log_metric("test_num_distinct_figures", mean(num_of_figures))
neptune_logger.log_metric("test_num_distinct_figures_std", std(num_of_figures))

neptune_logger.experiment.stop()
