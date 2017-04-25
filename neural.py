from __future__ import print_function
import numpy as np
import random
from copy import deepcopy

from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from matplotlib.pyplot import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def loss_function(true_label, reconstructed_label):
  zero = tf.zeros(tf.shape(true_label), dtype=tf.float32)
  bool_vector = tf.not_equal(true_label, zero)
  new_true = tf.where(bool_vector, true_label, zero)
  new_reconstructed = tf.where(bool_vector, reconstructed_label, zero)
  count = tf.count_nonzero(true_label)
  count = tf.to_float(count)
  return tf.reduce_sum(tf.square(new_reconstructed - new_true)) / count


def autoencoder(gene_matrix, dimensions, genes_expressed, scale=True):
        if scale:
                min_max_scaler = preprocessing.MinMaxScaler()
                train_gene_mat = min_max_scaler.fit_transform(gene_matrix)
        else:
                train_gene_mat = gene_matrix

        input_placeholder = tf.placeholder(tf.float32, shape=[None, genes_expressed])
        h1s = dimensions

        with tf.name_scope("encode_1"):

                W1 = tf.Variable(tf.truncated_normal(shape=[genes_expressed, h1s], stddev=0.1))
                b1 = tf.Variable(tf.truncated_normal(shape=[h1s], stddev=0.1))

                h1 = tf.matmul(input_placeholder, W1) + b1

        with tf.name_scope("decode_1"):
                W2 = tf.Variable(tf.truncated_normal(shape=[h1s, genes_expressed], stddev=0.1))
                b2 = tf.Variable(tf.truncated_normal(shape=[genes_expressed], stddev=0.1))
                h2 = tf.matmul(h1, W2) + b2


        with tf.name_scope("loss_function"):
                mse = loss_function(input_placeholder, h2)

        with tf.name_scope("train"):
                backprop = tf.train.AdamOptimizer(0.01).minimize(mse)

        sess = tf.Session()

        train_step = 100000
        # train_step = 5
        train_batch = train_gene_mat.reshape((-1, genes_expressed))
        sess.run(tf.global_variables_initializer())

        for i in range(train_step):
                ind = i%train_gene_mat.shape[0]
                data = train_gene_mat[ind].reshape((1, genes_expressed))
                _, loss2, W_1, b_1 = sess.run([backprop, mse, W1, b1], feed_dict={input_placeholder:data})
                if i % 1000 == 0:
                        print(i, loss2)
        return W_1, b_1
        # for i in range(gene_matrix.shape[0]):
        #         if i%100==0:
        #                 print(i)
        #         s1 = sess.run(tf.matmul(input_placeholder, W1) + b1,
        #                 feed_dict={input_placeholder:gene_matrix[i].reshape((-1, genes_expressed))})
        #         gene_recon[i] = s1

def generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef):
        """
        generates data with multiple clusters.
        Checked.
        """
        mu = 3
        range_from_value = .1

        if n_clusters == 1:
                Z = np.random.multivariate_normal(mean = np.zeros([k,]), cov = np.eye(k), size = n).transpose()
                cluster_ids =  np.ones([n,])
        else:
                Z = np.zeros([k, n])
                cluster_ids = np.array([random.choice(range(n_clusters)) for i in range(n)])
                for id in list(set(cluster_ids)):
                        idxs = cluster_ids == id
                        cluster_mu = (np.random.random([k,]) - .5) * 5
                        Z[:, idxs] = np.random.multivariate_normal(mean = cluster_mu, cov = .05 * np.eye(k), size = idxs.sum()).transpose()

        A = np.random.random([d, k]) - .5
        mu = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * mu for i in range(d)])
        sigmas = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * sigma for i in range(d)])
        noise = np.zeros([d, n])
        for j in range(d):
                noise[j, :] = mu[j] + np.random.normal(loc = 0, scale = sigmas[j], size = n)
        X = (np.dot(A, Z) + noise).transpose()
        Y = deepcopy(X)
        Y[Y < 0] = 0
        rand_matrix = np.random.random(Y.shape)

        cutoff = np.exp(-decay_coef * (Y ** 2))
        zero_mask = rand_matrix < cutoff
        Y[zero_mask] = 0
        print('Fraction of zeros: %2.3f; decay coef: %2.3f' % ((Y == 0).mean(), decay_coef))
        return X, Y, Z.transpose(), cluster_ids
# a = [None]
# b = [None]

random.seed(35)
np.random.seed(32)
n = 200
d = 20
k = 2
sigma = .3
n_clusters = 3
decay_coef = .023 # 0.79
# decay_coef = .1 # 0.64
# decay_coef = .34 # 0.07



X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
# Y_mean = np.mean(Y, axis=0)
# Y_mean = Y - Y_mean
test, bias = autoencoder(Y, 2, 20, scale=False)


colors = ['red', 'blue', 'green']
cluster_ids = sorted(list(set(ids)))
factor_analysis_Zhat = Y @ test + bias
factor_std = np.std(factor_analysis_Zhat, axis=0)
factor_analysis_Zhat = factor_analysis_Zhat / factor_std
figure(figsize = [15, 5])
subplot(131)
for id in cluster_ids:
    scatter(factor_analysis_Zhat[ids == id, 0], factor_analysis_Zhat[ids == id, 1], color = colors[id - 1], s = 4)
    title('Neural Network Estimated Latent Positions')
    xlim([-4, 4])
    ylim([-4, 4])
subplot(132)
for id in cluster_ids:
    scatter(Z[ids == id, 0], Z[ids == id, 1], color = colors[id - 1], s = 4)
    title('True Latent Positions\nFraction of Zeros %2.3f' % (Y == 0).mean())
    xlim([-4, 4])
    ylim([-4, 4])
show()
