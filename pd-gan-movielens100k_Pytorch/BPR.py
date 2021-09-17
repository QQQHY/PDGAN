"""
BPR and MF
"""

from __future__ import division
import numpy as np
import random
from scipy import sparse
import pandas as pd
import _pickle as cPickle


def load_data(data_dir, train_file, num_users, num_items):
    # Step 1: Read training data and create sparse user-item matrix of local implicit feedback
    print('loading training matrix')
    headers = ["userId", "itemId", "ratings"]
    data = pd.read_csv(os.path.join(data_dir, train_file), names=headers, delimiter="\t")
    user_idx = data["userId"]
    item_idx = data["itemId"]
    ratings = data["ratings"]
    # ratings[ratings < 4] = 0    # make it aline with IRGAN setting

    training_matrix = sparse.csr_matrix((num_users, num_items), dtype='float32')
    temp = sparse.coo_matrix((np.ones_like(ratings), (user_idx, item_idx)), shape=(num_users, num_items),
                             dtype='float32')
    training_matrix = training_matrix + temp

    return training_matrix


class Args(object):
    def __init__(self, learning_rate=0.05,
                 lambda_theta=0.0025,
                 lambda_s=1):
        self.learning_rate = learning_rate
        self.lambda_theta = lambda_theta
        self.lambda_s = lambda_s


class BPR(object):
    def __init__(self, train_data, test_data, d, args, max_sample, max_iters, data_dir):
        """ Initialize the model
        train_data: user-item implicit feedback matrix
        test_data: leave-one-out method, in the form of user, item, rating triple
        d: dimension of latent factors
        args: defined by Args class
        max_sample: number of samples to be updated per iteration
        max_iter: maximum number of iterations
        """

        self.train_data = train_data
        self.test_data = test_data

        self.num_users, self.num_items = self.train_data.shape

        self.d = d
        self.learning_rate = args.learning_rate
        self.lambda_theta = args.lambda_theta
        self.lambda_s = args.lambda_s

        self.max_sample = max_sample
        self.max_iters = max_iters

        self.data_dir = data_dir

        self.user_factors = np.sqrt(1/d)*np.random.random_sample((self.num_users, self.d))
        self.item_factors = np.sqrt(1/d)*np.random.random_sample((self.num_items, self.d))

    def train(self):
        print('start BPR training...')
        sampler = Sampler(self.train_data)

        loss_samples = self.create_loss_samples(sampler)
        previous_loss = self.loss(loss_samples)
        print('initial loss = {0}'.format(previous_loss))

        for it in range(self.max_iters):
            print('starting iteration {0}'.format(it))
            for u, i, j in sampler.generate_samples(self.max_sample):
                self.update_factors(u, i, j)

            current_loss = self.loss(loss_samples)
            print('iteration {0}: loss = {1}'.format(it, current_loss))

            if np.abs(current_loss) > np.abs(previous_loss):
                if self.learning_rate > 0.001:
                    self.learning_rate = 0.001
                else:
                    break

            previous_loss = current_loss

    def create_loss_samples(self, sampler):
        # apply rule of thumb to decide num samples over which to create loss
        num_loss_samples = int(100 * self.num_items ** 0.5)
        print('sampling {0} <u,i,j> triples...'.format(num_loss_samples))
        loss_samples = [s for s in sampler.generate_samples(num_loss_samples)]
        return loss_samples

    def update_factors(self, u, i, j):
        # calculate x_uij
        x = np.dot(self.user_factors[u, :], self.item_factors[i, :] - self.item_factors[j, :])
        z = np.exp(-x) / (1.0 + np.exp(-x))

        # update user
        d = (self.item_factors[i, :] - self.item_factors[j, :])
        self.user_factors[u, :] += self.learning_rate * (z * d + self.lambda_theta * self.user_factors[u, :])

        # update item factors
        d = self.user_factors[u, :]
        self.item_factors[i, :] += self.learning_rate * (z * d + self.lambda_theta * self.item_factors[i, :])

        # update negative item factors
        d = -self.user_factors[u, :]
        self.item_factors[j, :] += self.learning_rate * (z * d + self.lambda_theta * self.item_factors[j, :])

    def loss(self, loss_samples):
        ranking_loss = 0
        for u, i, j in loss_samples:
            x = self.predict(u, i) - self.predict(u, j)
            ranking_loss += np.log(1.0 / (1.0 + np.exp(-x)))

        complexity = 0
        for u, i, j in loss_samples:
            complexity += self.lambda_theta * np.dot(self.user_factors[u], self.user_factors[u])
            complexity += self.lambda_theta * np.dot(self.item_factors[i], self.item_factors[i])
            complexity += self.lambda_theta * np.dot(self.item_factors[j], self.item_factors[j])

        return ranking_loss - complexity

    def predict(self, u, i):
        return np.dot(self.user_factors[u], self.item_factors[i])

    def evaluate(self):
        # AUC
        value = 0.0
        count = 0.0
        index = np.array(range(self.num_items))
        print('Evaluating AUC...with test data %d...' % (self.test_data.shape[0] + 1))
        t_test = []
        for u, i, _ in self.test_data:
            neg_items = np.delete(index, self.train_data[u].nonzero()[1])
            # if self.max_sample is None:
            #     pass
            # else:
            #     neg_items = random.sample(neg_items, 10000)
            t_count = 0
            t_value = 0
            for j in neg_items:
                count += 1
                t_count += 1
                if self.predict(u, i) - self.predict(u, j) > 0:
                    value += 1
                    t_value += 1
                else:
                    pass
            t_test.append(t_value/t_count)
        np.savetxt('BPR_t_test.csv', t_test, delimiter=' ')
        if count != 0:
            auc = value / count
        else:
            auc = np.nan
        print('AUC value for BPR is: %.4f, total value is %d, total count is %d' % (auc, value, count))
        return auc

    def save_factors(self, workdir, filename):
        print('saving BPR factors')
        cPickle.dump([self.user_factors, self.item_factors], open(workdir + filename, 'wb'))

    def load_factors(self, workdir, filename):
        print('loading BPR factors...')
        param = cPickle.load(open(workdir + filename, 'rb'), encoding='latin1')
        self.user_factors = param[0]
        self.item_factors = param[1]


class MF(object):
    def __init__(self, train_data, test_data, d, args, max_iters, data_dir):
        """ Initialize the model
        train_data: user-item implicit feedback matrix
        test_data: leave-one-out method, in the form of user, item, rating triple
        d: dimension of latent factors
        args: defined by Args class
        max_sample: number of samples to be updated per iteration
        max_iter: maximum number of iterations
        """
        self.train_data = train_data
        self.test_data = test_data
        self.num_users, self.num_items = train_data.shape

        self.d = d
        self.learning_rate = args.learning_rate
        self.regularization = args.lambda_theta
        self.max_iters = max_iters

        self.data_dir = data_dir

        self.user_factors = np.sqrt(1/d)*np.random.random_sample((self.num_users, self.d))
        self.item_factors = np.sqrt(1/d)*np.random.random_sample((self.num_items, self.d))

    def train(self):
        print('start MF training...')
        user_idx, item_idx = self.train_data.nonzero()
        num_entries = self.train_data.nnz
        for it in range(self.max_iters):
            print('iter %d' % it)
            for k in range(num_entries):
                u = user_idx[k]
                i = item_idx[k]
                delta = self.train_data[u, i] - np.dot(self.user_factors[u], self.item_factors[i])
                self.user_factors[u] = self.user_factors[u] + \
                    self.learning_rate * (delta * self.item_factors[i] - self.regularization * self.user_factors[u])
                self.item_factors[i] = self.item_factors[i] + \
                    self.learning_rate * (delta * self.user_factors[u] - self.regularization * self.item_factors[i])

    def predict(self, u, i):
        return np.dot(self.user_factors[u], self.item_factors[i])

    def evaluate(self):
        # AUC
        value = 0.0
        count = 0.0
        index = np.array(range(self.num_items))
        print('Evaluating AUC...with test data %d...' % (self.test_data.shape[0] + 1))
        for u, i, _ in self.test_data:
            neg_items = np.delete(index, self.train_data[u].nonzero()[1])
            # if self.max_sample is None:
            #     pass
            # else:
            #     neg_items = random.sample(neg_items, 10000)
            for j in neg_items:
                count += 1
                if self.predict(u, i) - self.predict(u, j) > 0:
                    value += 1
                else:
                    pass
        if count != 0:
            auc = value / count
        else:
            auc = np.nan
        print('AUC value for MF is: %.4f, total value is %d, total count is %d' % (auc, value, count))
        return auc

    def save_factors(self):
        print('saving MF factors...')
        np.save(os.path.join(self.data_dir, 'MF_user_factors_%d' % self.d), self.user_factors)
        np.save(os.path.join(self.data_dir, 'MF_item_factors_%d' % self.d), self.item_factors)

    def load_factors(self):
        print('loading MF factors...')
        self.user_factors = np.load(os.path.join(self.data_dir, 'MF_user_factors_%d.npy' % self.d))
        self.item_factors = np.load(os.path.join(self.data_dir, 'MF_item_factors_%d.npy' % self.d))


class Sampler(object):
    def __init__(self, data):
        self.data = data
        self.num_users, self.num_items = data.shape
        self.max_samples = None

    def sample_user(self):
        while True:
            u = random.randint(0, self.num_users - 1)
            if self.data[u].indices.any():
                break
        return u

    def sample_item(self, u):
        i = random.choice(self.data[u].indices)
        return i

    def sample_negative_item(self, u):
        items = self.data[u].indices
        j = random.randint(0, self.num_items - 1)
        while j in items:
            j = random.randint(0, self.num_items - 1)
        return j

    def sample_negative_item_for_u(self, u):
        index = np.array(range(self.num_items))
        u_items = len(self.data[u].indices)
        neg_items = np.delete(index, self.data[u].nonzero()[1])
        if u_items <= len(neg_items):
            neg_for_u = random.sample(neg_items, u_items)
        else:
            neg_for_u = []
            for i in range(u_items):
                neg_for_u.append(random.sample(neg_items, 1)[0])
        return neg_for_u

    def generate_samples(self, max_samples):
        # if max_samples are not given, use whole data set
        if max_samples is None:
            user_idx, item_idx = self.data.nonzero()
            for k in range(self.data.nnz):
                u = user_idx[k]
                i = item_idx[k]
                j = self.sample_negative_item(u)
                yield u, i, j
        else:
            num = min(self.data.nnz, max_samples)

            for _ in range(num):
                u = self.sample_user()
                i = self.sample_item(u)
                j = self.sample_negative_item(u)
                yield u, i, j

    def generate_samples_for_mf(self, max_samples):
        # if max_samples are not given, use whole data set
        if max_samples is None:
            user_idx, item_idx = self.data.nonzero()
            for k in range(self.data.nnz):
                u = user_idx[k]
                i = self.sample_item(u)
                yield u, i

        else:
            num = min(self.data.nnz, max_samples)

            for _ in range(num):
                u = self.sample_user()
                i = self.sample_item(u)
                yield u, i


if __name__ == '__main__':
    import os
    # import pandas as pd

    """
    Data preprocessing and loading
    """
    num_user = 943
    num_item = 1683

    workdir = 'ml-100k/'
    train_file = 'movielens-100k-train.txt'
    train_matrix = load_data(workdir, train_file, num_user, num_item)

    test_file = 'movielens-100k-test.txt'
    test_matrix = load_data(workdir, test_file, num_user, num_item)

    row, col = test_matrix.nonzero()
    val = test_matrix.data
    test_data = np.concatenate((row[:, None], col[:, None], val[:, None]), 1)

    """
      Creating models for comparison
    """
    args = Args()

    args.learning_rate = 0.01

    num_factors = 50
    max_sample_size = 10000  # the maximum number of sample to be updated in each iter, if none, train with all entries
    max_iters = 300

    model = BPR(train_matrix, test_data, num_factors, args, max_sample_size, max_iters, workdir)
    model.train()
    model.save_factors(workdir, 'bpr_embeddings_50.pkl')