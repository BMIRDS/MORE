from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
from model import Word2Vec
from config import Options


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
"""

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data", None, "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_boolean(
    "use_ontologies", True,
    "If true, use UMLS ontologies ")
flags.DEFINE_string(
    "ckpt_dir", None, "Directory from which to load the model"
)
flags.DEFINE_boolean(
    "train_model", True,
    "Set to True to Train and False to Eval ")
flags.DEFINE_string(
    "eval_data", None, "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 300, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 100,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 10,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS


def main(_):
    # Train a word2vec model.
    if FLAGS.train_model:
        if not FLAGS.train_data or not FLAGS.save_path:
            print("--train_data and --save_path must be specified.")
            sys.exit(1)
        opts = Options(FLAGS)
        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
            with tf.device("/gpu:2"):
                model = Word2Vec(opts, session)
                for i in range(opts.epochs_to_train):
                    print("Beginning epoch {}".format(i))
                    model.train()  # Process one epoch
                #  Perform a final save.
                model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"), global_step=model.global_step)
    else:
        opts = Options(FLAGS)
        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
            with tf.device("/cpu:0"):
                model = Word2Vec(opts, session)
                model.get_embeddings_from_ckpt('./Results/wup_lch_nam/com_30_p/')
                model.get_eval_sims("./Results/wup_lch_nam/com_30_p_report.csv")

if __name__ == "__main__":
    tf.app.run()

