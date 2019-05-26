import os
from utils import *
import tensorflow as tf
import threading
import time
import sys


class Word2Vec(object):
    """Word2Vec model (Skipgram)."""
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        if self._options.use_ontologies:
            # Get the index keys and matrices
            idx_keys, self.matrix = load_matrix("./similarities/pickle/combined_3_key_idx.pkl",
                                                        "./similarities/pickle/combined_3_matrix.pkl")
            self._w_to_sim_idx = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(idx_keys),
                                                                           default_value=-1)
            self._session.run(tf.tables_initializer())

        if self._options.train_model:
            print("Train Model")
            self.build_graph()
        else:
            print("Eval Mode")
            self.build_eval_graph()

    """Build the graph for the forward pass."""
    def forward(self, examples, labels):

        opts = self._options

        # Declare all variables we need. Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(tf.random_uniform([opts.vocab_size, opts.emb_dim], -init_width, init_width), name="emb")
        self._emb = emb

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="sm_w_t")

        # Softmax bias: [vocab_size].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [opts.batch_size, 1])

        # Negative sampling.
        # https://www.tensorflow.org/api_docs/python/tf/nn/fixed_unigram_candidate_sampler
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist())
        )

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb, sampled_w, transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits, sampled_ids

    """Modified positive objective function."""
    def modify_loss(self, true_logits, sims):

        self._num_ont_pairs = tf.cast(tf.count_nonzero(sims), tf.int32)
        self._num_tot_pairs = tf.cast(tf.size(sims), tf.int32)

        sims = tf.where(
            tf.equal(tf.constant(0, dtype=tf.float32), sims),
            tf.ones_like(sims),
            sims
        )

        # Average ones like and sims
        p = tf.divide(tf.math.add(tf.ones_like(true_logits), sims), 2)
        logit_q = true_logits
        sig = tf.sigmoid(logit_q)

        # Calculate the new true cross entropy loss, using the modified softmax
        # Use clip by value to make sure the we are not doing log(0)
        xent = p * -tf.log(tf.clip_by_value(sig, 1e-10, 1.0))
        return xent

    """Modified negative objective function."""
    def modify_neg_sampling_loss(self, sampled_logits, sample_sims):
        self._num_ont_neg_pairs = tf.cast(tf.count_nonzero(sample_sims), tf.int32)
        self._num_tot_neg_pairs = tf.cast(tf.size(sample_sims), tf.int32)

        p = tf.divide(tf.math.add(tf.zeros_like(sampled_logits), sample_sims), 2)
        logit_q = sampled_logits
        sig = tf.sigmoid(logit_q)

        # Calculate the new true cross entropy loss, using the modified softmax
        # Use clip by value to make sure the we are not doing log(0)
        xent = (1 - p) * -tf.log(tf.clip_by_value(1 - sig, 1e-10, 1.0))
        return xent

    """Build the graph for the NCE loss."""
    def nce_loss(self, true_logits, sampled_logits, sims, sample_sims):
        # cross-entropy(logits, labels)
        opts = self._options
        if opts.use_ontologies:
            true_xent = self.modify_loss(true_logits, sims)
            sampled_xent = self.modify_neg_sampling_loss(sampled_logits, sample_sims)
        else:
            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits), logits=true_logits)
            sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    """Build the graph to optimize the loss function."""
    def optimize(self, loss):
        # Optimizer nodes. Linear learning rate decay.
        opts = self._options

        # Remove LR Decay for adam
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)

        self._lr = lr
        optimizer = tf.train.GradientDescentOptimizer(lr)

        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train

    """Build the graph for the full model."""
    def build_graph(self):
        opts = self._options

        # The training data. A text file.
        word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
         labels) = word2vec.skipgram_word2vec(filename=opts.train_data, batch_size=opts.batch_size,
            window_size=opts.window_size, min_count=opts.min_count, subsample=opts.subsample)

        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)
        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
          self._word2id[w] = i

        true_logits, sampled_logits, sampled_ids = self.forward(examples, labels)

        sims = None
        sample_sims = None
        if opts.use_ontologies:
            # Get example words and label words
            # Example (target) words are used to predict label (context) words
            example_words = tf.gather(self._id2word, examples)
            label_words = tf.gather(self._id2word, labels)
            sample_words = tf.gather(self._id2word, sampled_ids)

            # Get the similarity matrix indices of the example words
            example_idxs = self._w_to_sim_idx.lookup(example_words)
            label_idxs = self._w_to_sim_idx.lookup(label_words)

            sample_idxs = self._w_to_sim_idx.lookup(sample_words)
            multiply = tf.constant([opts.batch_size])
            sample_stack = tf.reshape(tf.tile(sample_idxs, multiply), [multiply[0], tf.shape(sample_idxs)[0]])
            examples_stack = tf.transpose([example_idxs])
            examples_stack = tf.tile(examples_stack, [1, opts.num_samples])
            sample_mat_idxs = tf.stack([examples_stack, sample_stack], axis=2)


            # Zip example and label indices to get matrix coordinates
            mat_idxs = tf.stack([example_idxs, label_idxs], axis=1)

            # Get similarity scores from matrix
            # On TF GPU, if an out of bound index is found, a 0 is stored in the corresponding output value.
            # Default value -1 will ensure that a 0 is returned when word is not in ontology
            sims = tf.gather_nd(self.matrix, mat_idxs)
            sample_sims = tf.gather_nd(self.matrix, sample_mat_idxs)

        loss = self.nce_loss(true_logits, sampled_logits, sims, sample_sims)
        tf.summary.scalar("NCE loss", loss)

        self._loss = loss
        self.optimize(loss)

        self.saver = tf.train.Saver()

        if opts.ckpt_dir is None:
            # Properly initialize all variables.
            tf.global_variables_initializer().run()
        else:
            checkpoint = tf.train.latest_checkpoint(opts.ckpt_dir)
            print("Restoring from Checkpoint")
            self.saver.restore(self._session, checkpoint)


    """Save the vocabulary to a file so the model can be reloaded."""
    def save_vocab(self):
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in range(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
                f.write("%s %d\n" % (vocab_word, opts.vocab_counts[i]))

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break

    """Train the model."""
    def train(self):
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch, self._words])

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)
        workers = []
        for _ in range(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time, last_summary_time = initial_words, time.time(), 0
        last_checkpoint_time = 0
        if opts.use_ontologies:
            all_num_ont_pairs, all_num_tot_pairs, all_num_neg_ont, all_num_neg_tot = 0, 0, 0, 0
        while True:
            time.sleep(opts.statistics_interval)  # Reports our progress once a while.

            (epoch, step, loss, words, lr) = self._session.run([self._epoch, self.global_step, self._loss, self._words, self._lr])

            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)

            if opts.use_ontologies:
                (num_ont_pairs, num_tot_pairs, num_neg_ont, num_neg_tot) = self._session.run(
                    [self._num_ont_pairs, self._num_tot_pairs, self._num_ont_neg_pairs, self._num_tot_neg_pairs])

                # Record percentage of positive ontology word pairs and negative ontology word pairs
                all_num_ont_pairs += num_ont_pairs
                all_num_tot_pairs += num_tot_pairs
                all_num_neg_ont += num_neg_ont
                all_num_neg_tot += num_neg_tot
                print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f ont_percent = %5.3f neg_percent = %5.3f\r"
                      % (epoch, step, lr, loss, rate, 100*all_num_ont_pairs/float(all_num_tot_pairs),
                         100*all_num_neg_ont/float(all_num_neg_tot)), end="")
            else:
                print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r"
                    % (epoch, step, lr, loss, rate), end="")
            sys.stdout.flush()
            if now - last_summary_time > opts.summary_interval:
                summary_str = self._session.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                last_summary_time = now
            if now - last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session, os.path.join(opts.save_path, "model.ckpt"), global_step=step.astype(int))
            last_checkpoint_time = now
            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()
        return epoch

    """Build the graph for the full model."""
    def build_eval_graph(self):

        opts = self._options
        # The training data. A text file.
        word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
        (words, counts, words_per_epoch, self._epoch, self._words, examples, labels) = word2vec.skipgram_word2vec(
            filename=opts.train_data, batch_size=opts.batch_size, window_size=opts.window_size,
            min_count=opts.min_count, subsample=opts.subsample)

        (opts.vocab_words, opts.vocab_counts, opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)
        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

    """Calculate cosine similarity between two phrases."""
    def calc_sim(self, phrase_1_idxs, phrase_2_idxs):
        # Get the embeddings for all words in phrase 1 and take mean for final vector
        phrase_1_embeds = tf.nn.embedding_lookup(self._emb, phrase_1_idxs)
        phrase_1_embed = tf.reduce_mean(phrase_1_embeds, 0)
        # Get the embeddings for all words in phrase 2 and take mean for final vector
        phrase_2_embeds = tf.nn.embedding_lookup(self._emb, phrase_2_idxs)
        phrase_2_embed = tf.reduce_mean(phrase_2_embeds, 0)
        normalize_p1 = tf.nn.l2_normalize(phrase_1_embed, 0)
        normalize_p2 = tf.nn.l2_normalize(phrase_2_embed, 0)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_p1, normalize_p2))
        # Normalize cosine similarity from range [-1, 1] to [0, 1]
        cos_similarity = (cos_similarity + 1) / 2
        return cos_similarity

    """Calculate evaluation similarities."""
    def get_eval_sims(self, out_file, eval_ds):
        pairs_1, phys_1, expert_1 = read_eval_ds(eval_ds)
        unused_pairs = set()
        similarities_1 = []
        # Get evaluation similarities for evaluation dataset
        for i in range(len(pairs_1)):
            pair = pairs_1[i]
            phrase1 = pair[0]
            phrase2 = pair[1]
            phrase_1_idxs = [self._word2id[w] for w in clean_phrase(phrase1) if w in self._word2id]
            phrase_2_idxs = [self._word2id[w] for w in clean_phrase(phrase2) if w in self._word2id]

            if phrase_1_idxs and phrase_2_idxs:
                cos_similarity = self.calc_sim(phrase_1_idxs, phrase_2_idxs)
                sim = self._session.run(cos_similarity)
                sim *= 4
                similarities_1.append((pair, phys_1[i], expert_1[i], sim))
                print("{} — {}: {}".format(phrase1, phrase2, sim))
            else:
                unused_pairs.add(pair)
        # Write evaluation similarities to csv file
        write_report(out_file, similarities_1, unused_pairs)

    """Calculate embeddings from checkpoint folder for evaluation."""
    def get_embeddings_from_ckpt(self, ckpt_dir):
        init_width = 0.5 / self._options.emb_dim
        emb = tf.Variable(tf.random_uniform([self._options.vocab_size, self._options.emb_dim], -init_width, init_width), name="emb")
        self._emb = emb

        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(self._session, checkpoint)


