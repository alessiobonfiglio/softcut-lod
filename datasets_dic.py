import glob
import os
import random

from segmentation_dataset import SegDataset, SegDatasetV2, MulticlassSegDataset, SemisupervisedSegDataset


def get_dataset_by_name(name, dataset_dir, augmentation=None):
    if name == "horses":
        return SegDataset(os.path.join(dataset_dir, "horses"), "imgs", "segs", augmentation)
    if name == "weizmann_horse_rgb":
        return SegDataset(os.path.join(dataset_dir, "weizmann_horse_db"), "rgb", "figure_ground", augmentation)
    if name == "weizmann_horse_gray":
        return SegDataset(os.path.join(dataset_dir, "weizmann_horse_db"), "gray", "figure_ground", augmentation)
    raise ValueError("Dataset {} not found".format(name))


def get_dataset_by_name_and_splits(name, dataset_dir, splits, tiling_factor=(1, 1), training_augmentation=None,
                                   validation_augmentation=None, testing_augmentation=None):
    background_channel = False
    semisupervised = False

    if name == "horses" and len(splits) == 3:
        parent_dir = os.path.join(dataset_dir, "horses")
        image_list = glob.glob(os.path.join(parent_dir, "imgs", "*"))
        mask_list = glob.glob(os.path.join(parent_dir, "segs", "*"))
        image_list.sort()
        mask_list.sort()

    elif name == "weizmann_horse_rgb" and len(splits) == 3:
        parent_dir = os.path.join(dataset_dir, "weizmann_horse_db")
        image_list = glob.glob(os.path.join(parent_dir, "rgb", "*"))
        mask_list = glob.glob(os.path.join(parent_dir, "figure_ground", "*"))
        image_list.sort()
        mask_list.sort()

    elif name == "weizmann_horse_gray" and len(splits) == 3:
        parent_dir = os.path.join(dataset_dir, "weizmann_horse_db")
        image_list = glob.glob(os.path.join(parent_dir, "gray", "*"))
        mask_list = glob.glob(os.path.join(parent_dir, "figure_ground", "*"))
        image_list.sort()
        mask_list.sort()

    elif name == "cityscapes_traffic_signs_downsampled":
        parent_dir = os.path.join(dataset_dir, "cityscapes_traffic_signs_downsampled")
        training_image_list = glob.glob(os.path.join(parent_dir, "train_imgs", "*"))
        training_mask_list = glob.glob(os.path.join(parent_dir, "train_masks", "*"))
        validation_image_list = glob.glob(os.path.join(parent_dir, "val_imgs", "*"))
        validation_mask_list = glob.glob(os.path.join(parent_dir, "val_masks", "*"))
        testing_image_list = glob.glob(os.path.join(parent_dir, "test_imgs", "*"))
        testing_mask_list = glob.glob(os.path.join(parent_dir, "test_masks", "*"))
        training_image_list.sort()
        training_mask_list.sort()
        validation_image_list.sort()
        validation_mask_list.sort()
        testing_image_list.sort()
        testing_mask_list.sort()
    elif name == "cityscapes_coarse_traffic_signs_downsampled":
        parent_dir = os.path.join(dataset_dir, "cityscapes_coarse_traffic_signs_downsampled")
        training_image_list = glob.glob(os.path.join(parent_dir, "train_imgs", "*"))
        training_mask_list = glob.glob(os.path.join(parent_dir, "train_masks", "*"))
        validation_image_list = glob.glob(os.path.join(parent_dir, "val_imgs", "*"))
        validation_mask_list = glob.glob(os.path.join(parent_dir, "val_masks", "*"))
        testing_image_list = glob.glob(os.path.join(parent_dir, "test_imgs", "*"))
        testing_mask_list = glob.glob(os.path.join(parent_dir, "test_masks", "*"))
        training_image_list.sort()
        training_mask_list.sort()
        validation_image_list.sort()
        validation_mask_list.sort()
        testing_image_list.sort()
        testing_mask_list.sort()
    elif name == "cityscapes_2c_downsampled":
        parent_dir = os.path.join(dataset_dir, "cityscapes_2c_downsampled")
        training_image_list = glob.glob(os.path.join(parent_dir, "train_imgs", "*"))
        training_mask_ts_list = glob.glob(os.path.join(parent_dir, "train_masks", "*_ts.png"))
        training_mask_car_list = glob.glob(os.path.join(parent_dir, "train_masks", "*_car.png"))
        validation_image_list = glob.glob(os.path.join(parent_dir, "val_imgs", "*"))
        validation_mask_ts_list = glob.glob(os.path.join(parent_dir, "val_masks", "*_ts.png"))
        validation_mask_car_list = glob.glob(os.path.join(parent_dir, "val_masks", "*_car.png"))
        testing_image_list = glob.glob(os.path.join(parent_dir, "test_imgs", "*"))
        testing_mask_ts_list = glob.glob(os.path.join(parent_dir, "test_masks", "*_ts.png"))
        testing_mask_car_list = glob.glob(os.path.join(parent_dir, "test_masks", "*_car.png"))

        training_mask_ts_list = [i.replace("_ts.png", ".png") for i in training_mask_ts_list]
        validation_mask_ts_list = [i.replace("_ts.png", ".png") for i in validation_mask_ts_list]
        testing_mask_ts_list = [i.replace("_ts.png", ".png") for i in testing_mask_ts_list]
        training_mask_car_list = [i.replace("_car.png", ".png") for i in training_mask_car_list]
        validation_mask_car_list = [i.replace("_car.png", ".png") for i in validation_mask_car_list]
        testing_mask_car_list = [i.replace("_car.png", ".png") for i in testing_mask_car_list]

        training_image_list.sort()
        training_mask_ts_list.sort()
        training_mask_car_list.sort()
        validation_image_list.sort()
        validation_mask_ts_list.sort()
        validation_mask_car_list.sort()
        testing_image_list.sort()
        testing_mask_ts_list.sort()
        testing_mask_car_list.sort()

        training_mask_ts_list = [i.replace(".png", "_ts.png") for i in training_mask_ts_list]
        validation_mask_ts_list = [i.replace(".png", "_ts.png") for i in validation_mask_ts_list]
        testing_mask_ts_list = [i.replace(".png", "_ts.png") for i in testing_mask_ts_list]
        training_mask_car_list = [i.replace(".png", "_car.png") for i in training_mask_car_list]
        validation_mask_car_list = [i.replace(".png", "_car.png") for i in validation_mask_car_list]
        testing_mask_car_list = [i.replace(".png", "_car.png") for i in testing_mask_car_list]

        training_mask_list = (training_mask_ts_list, training_mask_car_list)
        validation_mask_list = (validation_mask_ts_list, validation_mask_car_list)
        testing_mask_list = (testing_mask_ts_list, testing_mask_car_list)
    elif name == "cityscapes_2c+background_downsampled":
        background_channel = True
        parent_dir = os.path.join(dataset_dir, "cityscapes_2c_downsampled")
        training_image_list = glob.glob(os.path.join(parent_dir, "train_imgs", "*"))
        training_mask_ts_list = glob.glob(os.path.join(parent_dir, "train_masks", "*_ts.png"))
        training_mask_car_list = glob.glob(os.path.join(parent_dir, "train_masks", "*_car.png"))
        validation_image_list = glob.glob(os.path.join(parent_dir, "val_imgs", "*"))
        validation_mask_ts_list = glob.glob(os.path.join(parent_dir, "val_masks", "*_ts.png"))
        validation_mask_car_list = glob.glob(os.path.join(parent_dir, "val_masks", "*_car.png"))
        testing_image_list = glob.glob(os.path.join(parent_dir, "test_imgs", "*"))
        testing_mask_ts_list = glob.glob(os.path.join(parent_dir, "test_masks", "*_ts.png"))
        testing_mask_car_list = glob.glob(os.path.join(parent_dir, "test_masks", "*_car.png"))

        training_mask_ts_list = [i.replace("_ts.png", ".png") for i in training_mask_ts_list]
        validation_mask_ts_list = [i.replace("_ts.png", ".png") for i in validation_mask_ts_list]
        testing_mask_ts_list = [i.replace("_ts.png", ".png") for i in testing_mask_ts_list]
        training_mask_car_list = [i.replace("_car.png", ".png") for i in training_mask_car_list]
        validation_mask_car_list = [i.replace("_car.png", ".png") for i in validation_mask_car_list]
        testing_mask_car_list = [i.replace("_car.png", ".png") for i in testing_mask_car_list]

        training_image_list.sort()
        training_mask_ts_list.sort()
        training_mask_car_list.sort()
        validation_image_list.sort()
        validation_mask_ts_list.sort()
        validation_mask_car_list.sort()
        testing_image_list.sort()
        testing_mask_ts_list.sort()
        testing_mask_car_list.sort()

        training_mask_ts_list = [i.replace(".png", "_ts.png") for i in training_mask_ts_list]
        validation_mask_ts_list = [i.replace(".png", "_ts.png") for i in validation_mask_ts_list]
        testing_mask_ts_list = [i.replace(".png", "_ts.png") for i in testing_mask_ts_list]
        training_mask_car_list = [i.replace(".png", "_car.png") for i in training_mask_car_list]
        validation_mask_car_list = [i.replace(".png", "_car.png") for i in validation_mask_car_list]
        testing_mask_car_list = [i.replace(".png", "_car.png") for i in testing_mask_car_list]

        training_mask_list = (training_mask_ts_list, training_mask_car_list)
        validation_mask_list = (validation_mask_ts_list, validation_mask_car_list)
        testing_mask_list = (testing_mask_ts_list, testing_mask_car_list)
    elif name == "cityscapes_2c_semi":
        semisupervised = True
        parent_dir = os.path.join(dataset_dir, "cityscapes_2c_semi")

        train_image_folder = os.path.join(parent_dir, "train_imgs")
        val_image_folder = os.path.join(parent_dir, "val_imgs")
        test_image_folder = os.path.join(parent_dir, "test_imgs")

        train_mask_folder = os.path.join(parent_dir, "train_masks")
        val_mask_folder = os.path.join(parent_dir, "val_masks")
        test_mask_folder = os.path.join(parent_dir, "test_masks")

        train_mask_list = [os.path.basename(x) for x in glob.glob(os.path.join(parent_dir, "train_masks", "*_mask_*"))]
        val_mask_list = [os.path.basename(x) for x in glob.glob(os.path.join(parent_dir, "val_masks", "*_mask_*"))]
        test_mask_list = [os.path.basename(x) for x in glob.glob(os.path.join(parent_dir, "test_masks", "*_mask_*"))]

        train_mask_list.sort()
        val_mask_list.sort()
        test_mask_list.sort()
    else:
        raise ValueError("Invalid dataset configuration (name={}, splits={})".format(name, splits))

    if len(splits) == 3 and not semisupervised:
        indices = [i for i in range(len(image_list))]
        random.shuffle(indices)

        end_train_index = len(image_list) - splits[1] - splits[2]
        training_image_list = [image_list[i] for i in indices[:splits[0]]]
        training_mask_list = [mask_list[i] for i in indices[:splits[0]]]
        validation_image_list = [image_list[i] for i in indices[end_train_index: end_train_index + splits[1]]]
        validation_mask_list = [mask_list[i] for i in indices[end_train_index: end_train_index + splits[1]]]
        testing_image_list = [image_list[i] for i in
                              indices[end_train_index + splits[1]: end_train_index + splits[1] + splits[2]]]
        testing_mask_list = [mask_list[i] for i in indices[end_train_index + splits[1]: end_train_index + splits[1] + splits[2]]]

    if len(splits) == 1 and not semisupervised:
        indices = [i for i in range(len(training_image_list))]
        random.shuffle(indices)

        training_image_list = [training_image_list[i] for i in indices[:splits[0]]]
        training_mask_list = training_mask_list if type(training_mask_list) is tuple else (training_mask_list,)
        training_mask_list = tuple(
            [[training_mask_list[j][i] for i in indices[:splits[0]]] for j in range(len(training_mask_list))])
        training_mask_list = training_mask_list[0] if len(training_mask_list) == 1 else training_mask_list

    if tiling_factor is None:
        tiling_factor = (1, 1)

    if semisupervised:
        training_set = SemisupervisedSegDataset(train_image_folder, train_mask_folder, train_mask_list,
                                                tiling_factor=tiling_factor, augmentation=training_augmentation)
        validation_set = SemisupervisedSegDataset(val_image_folder, val_mask_folder, val_mask_list,
                                                  tiling_factor=tiling_factor, augmentation=validation_augmentation)
        test_set = SemisupervisedSegDataset(test_image_folder, test_mask_folder, test_mask_list,
                                            tiling_factor=tiling_factor, augmentation=testing_augmentation)
    elif type(training_mask_list) is not tuple:
        training_set = SegDatasetV2(training_image_list, training_mask_list, tiling_factor=tiling_factor,
                                    augmentation=training_augmentation)
        validation_set = SegDatasetV2(validation_image_list, validation_mask_list, tiling_factor=tiling_factor,
                                      augmentation=validation_augmentation)
        test_set = SegDatasetV2(testing_image_list, testing_mask_list, tiling_factor=tiling_factor,
                                augmentation=testing_augmentation)
    else:
        training_set = MulticlassSegDataset(training_image_list, training_mask_list, tiling_factor=tiling_factor,
                                            augmentation=training_augmentation, background_channel=background_channel)
        validation_set = MulticlassSegDataset(validation_image_list, validation_mask_list, tiling_factor=tiling_factor,
                                              augmentation=validation_augmentation, background_channel=background_channel)
        test_set = MulticlassSegDataset(testing_image_list, testing_mask_list, tiling_factor=tiling_factor,
                                        augmentation=testing_augmentation, background_channel=background_channel)

    return training_set, validation_set, test_set
