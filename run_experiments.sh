#!/bin/sh


#run experiments of table 1
python submodular_nips17_CNN.py extra_tags=["horses","no_graph_cut","submdular_nips17"]
python submodular_nips17_CNN+sobmodular.py extra_tags=["horses","submodular","submdular_nips17"]
python submodular_nips17_CNN+submodular.py params.refine_with_isotonic_regression=True extra_tags=["horses","submodular+isotonic","submdular_nips17"]
python submodular_nips17_CNN+SoftCut_exact.py extra_tags=["horses","softcut_exact","submdular_nips17"]
python submodular_nips17_CNN+SoftCut.py extra_tags=["horses","softcut_apprx","submdular_nips17"]

python submodular_nips17_CNN.py extra_tags=["horses","no_graph_cut","submdular_nips17"] params.learning_rate=0.01
python submodular_nips17_CNN+sobmodular.py extra_tags=["horses","submodular","submdular_nips17"] params.learning_rate=0.01
python submodular_nips17_CNN+submodular.py params.refine_with_isotonic_regression=True extra_tags=["horses","submodular+isotonic","submdular_nips17"] params.learning_rate=0.01
python submodular_nips17_CNN+SoftCut_exact.py extra_tags=["horses","softcut_exact","submdular_nips17"] params.learning_rate=0.01
python submodular_nips17_CNN+SoftCut.py extra_tags=["horses","softcut_apprx","submdular_nips17"] params.learning_rate=0.01

python submodular_nips17_CNN.py extra_tags=["horses","no_graph_cut","submdular_nips17"] params.optimizer="Adagrad"
python submodular_nips17_CNN+sobmodular.py extra_tags=["horses","submodular","submdular_nips17"] params.optimizer="Adagrad"
python submodular_nips17_CNN+submodular.py params.refine_with_isotonic_regression=True extra_tags=["horses","submodular+isotonic","submdular_nips17"] params.optimizer="Adagrad"
python submodular_nips17_CNN+SoftCut_exact.py extra_tags=["horses","softcut_exact","submdular_nips17"] params.optimizer="Adagrad"
python submodular_nips17_CNN+SoftCut.py extra_tags=["horses","softcut_apprx","submdular_nips17"] params.optimizer="Adagrad"

python submodular_nips17_CNN.py extra_tags=["horses","no_graph_cut","submdular_nips17"] params.learning_rate=0.01 params.optimizer="Adagrad"
python submodular_nips17_CNN+sobmodular.py extra_tags=["horses","submodular","submdular_nips17"] params.learning_rate=0.01 params.optimizer="Adagrad"
python submodular_nips17_CNN+submodular.py params.refine_with_isotonic_regression=True extra_tags=["horses","submodular+isotonic","submdular_nips17"] params.learning_rate=0.01 params.optimizer="Adagrad"
python submodular_nips17_CNN+SoftCut_exact.py extra_tags=["horses","softcut_exact","submdular_nips17"] params.learning_rate=0.01 params.optimizer="Adagrad"
python submodular_nips17_CNN+SoftCut.py extra_tags=["horses","softcut_apprx","submdular_nips17"] params.learning_rate=0.01 params.optimizer="Adagrad"


#run experiments of table 2
python Unet+submodular.py extra_tags=["unet","cityscapes_binary","no_graph_cut"] params.enable_graphcut=False
python Unet+submodular.py extra_tags=["unet","cityscapes_binary","submodular+isotonic"]
python Unet+SoftCut_exact.py extra_tags=["unet","cityscapes_binary","softcut_exact"]
python Unet+SoftCut.py extra_tags=["unet","cityscapes_binary","softcut_apprx"]

python Unet+submodular.py params.enable_graphcut=False extra_tags=["unet","cityscapes_binary","no_graph_cut"] params.encoder='resnet34'
python Unet+submodular.py extra_tags=["unet","cityscapes_binary","submodular+isotonic"] params.encoder='resnet34'
python Unet+SoftCut_exact.py extra_tags=["unet","cityscapes_binary","softcut_exact"] params.encoder='resnet34'
python Unet+SoftCut.py extra_tags=["unet","cityscapes_binary","softcut_apprx"] params.encoder='resnet34'

python Unet+submodular.py params.enable_graphcut=False extra_tags=["unet","cityscapes_binary","no_graph_cut"] params.optimizer="Adagrad"
python Unet+submodular.py extra_tags=["unet","cityscapes_binary","submodular+isotonic"] params.optimizer="Adagrad"
python Unet+SoftCut.py extra_tags=["unet","cityscapes_binary","softcut_apprx"] params.optimizer="Adagrad"


#run experiments of table 3
python Unet+submodular_specific_object_detection.py extra_tags=["unet","cityscapes_specific_object_detection","no_graph_cut"] params.enable_graphcut=False
python Unet+submodular_specific_object_detection.py params.refine_with_isotonic_regression=True extra_tags=["unet","cityscapes_specific_object_detection","submodular+isotonic"]
python Unet+SoftCut.py extra_tags=["unet","cityscapes_specific_object_detection","softcut_apprx"]

python Unet+submodular_specific_object_detection.py params.enable_graphcut=False extra_tags=["unet","cityscapes_specific_object_detection","no_graph_cut"] params.encoder='resnet34'
python Unet+submodular_specific_object_detection.py params.refine_with_isotonic_regression=True extra_tags=["unet","cityscapes_specific_object_detection","submodular+isotonic"] params.encoder='resnet34'
python Unet+SoftCut.py extra_tags=["unet","cityscapes_specific_object_detection","softcut_apprx"] params.encoder='resnet34'


#run experiments of table 4
python Unet+submodular_multiclass.py extra_tags=["unet","cityscapes_multiclass","no_graph_cut"] params.enable_graphcut=False
python Unet+submodular_multiclass.py params.refine_with_isotonic_regression=True extra_tags=["unet","cityscapes_multiclass","submodular+isotonic"]
python Unet+SoftCut_multiclass.py extra_tags=["unet","cityscapes_multiclass","softcut_apprx"]


#run experiments of table 5
python Unet+submodular_full_cityscapes.py num_of_workers=4 params.batch_size=1 extra_tags=["unet","cityscapes_fullres","no_graph_cut"] params.enable_graphcut=False
python Unet+submodular_full_cityscapes.py params.refine_with_isotonic_regression=True num_of_workers=4 params.batch_size=1 extra_tags="unet","cityscapes_fullres","submodular+isotonic"]
python Unet+SoftCut_full_cityscapes.py num_of_workers=4 params.batch_size=1 extra_tags=["unet","cityscapes_fullres","softcut_apprx"]


