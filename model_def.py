import collections
import logging
from typing import Any, Dict, Tuple

import pedl
#from pedl.trial import get_gang_addrs

import run_classifier
#import radam
from run_classifier import main
import tensorflow as tf
from pedl.frameworks.tensorflow.estimator_trial import EstimatorTrial
import pedl

def get_tf_flags():

    flags = tf.flags
    FLAGS = flags.FLAGS

#    tf.flags.DEFINE_string(
#        "tpu_name", None,
#        "The Cloud TPU to use for training. This should be either the name "
#        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#        "url.")
#
#    tf.flags.DEFINE_string(
#        "tpu_zone", None,
#        "[Optional] GCE zone where the Cloud TPU is located in. If not "
#        "specified, we will attempt to automatically detect the GCE project from "
#        "metadata.")
#
#    tf.flags.DEFINE_string(
#        "gcp_project", None,
#        "[Optional] Project name for the Cloud TPU-enabled project. If not "
#        "specified, we will attempt to automatically detect the GCE project from "
#        "metadata.")
#
#    tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
#
#    flags.DEFINE_integer(
#        "num_tpu_cores", 8,
#        "Only used if `use_tpu` is True. Total number of TPU cores to use.")

    # combined_dict = hparams + pedl.get_data_config()

    combined_dict = {
        "data_dir": "/home/hphan/data/glue/",
        "output_dir": "/home/hphan/output/",
        "init_checkpoint": None,
        #"init_checkpoint": "/home/hphan/data/albert_base/model.ckpt",
        "albert_config_file": "/home/hphan/data/albert_base/albert_config.json",
        "spm_model_file": "/home/hphan/data/albert_base/30k-clean.model",
        "do_train": True,
        "do_eval": True,
        "do_predict": False,
        "do_lower_case": True,
        "max_seq_length": 128,
        "optimizer":"adamw",
        "task_name":"MNLI",
        "warmup_step":1000,
        "learning_rate":3e-5,
        "train_step":10000,
        "save_checkpoints_steps":100,
        "train_batch_size":64,
    }

    for key in combined_dict:
        FLAGS[key].value = combined_dict[key]

    return flags

class ALBERTTrial(EstimatorTrial):

    def __init__(self, hparams):
        self.flags = get_tf_flags()

    def build_estimator(self, hparams):
        run_type="build_estimator"
        estimator = main(self.flags,run_type)
        return estimator

    def build_train_spec(self, hparams):
        run_type="build_train_spec"
        train_spec = main(self.flags, run_type)
        return train_spec

    def build_validation_spec(self, hparams):
        run_type="build_eval_spec"
        eval_spec = main(self.flags, run_type)
        return eval_spec
