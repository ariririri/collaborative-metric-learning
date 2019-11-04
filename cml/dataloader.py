from multiprocessing import Process, Queue
from scipy.sparse import lil_matrix

from collections import defaultdict
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from tqdm import tqdm

import torch
from torch.utils.data.sampler import BatchSampler

class UserItemDataset(torch.utils.data.Dataset):
    def __init__(self, user_item_path=None, item_feature_path=None, transform=None):
        # データに対して汎用的にしようとすると,ここで死ぬ.
        self.transform = transform
        self.user_item_path = "citeulike-t/users.dat"
        self.item_feature_path = "citeulike-t/tag-item.dat"

        tag_occurence_thres = 10

        self.user_item_matrix = self.load_user_item()
        self.user_pos_num = self.user_item_matrix.sum(axis=1).getA().reshape(-1)

        n_items = self.user_item_matrix.shape[1]
        self.item_feature = self.load_item_feature(n_items, tag_occurence_thres)

        # TODO 
        # pos, user_item_pos_pairsの意味の分離
        self.user_item_pos_pairs = np.asarray(self.user_item_matrix.nonzero())
        self.data_num = self.user_item_pos_pairs[0].size
        self.user_item_neg_pairs =  (1 -  self.user_item_matrix.toarray()).nonzero()
        self.neg_data_num = self.user_item_neg_pairs[0].size
        # TODO
        # neg_rate =1 以外のsupport
        self.neg_rate = 1
        # インスタンス変数はinitで定義したいが...
        self._pos_neg_pair()
    
    def _pos_neg_pair(self):
        pos = np.concatenate([self.user_item_pos_pairs[0], self.user_item_pos_pairs[1]]).reshape(2,-1).T
        n_items = self.user_item_matrix.shape[1]
        # だいたいはNegativeという過程のもと得に修正せず.
        neg_items = np.random.choice(n_items, self.user_pos_num.sum())
        # TODO 適切に分割しないとneg_rate != 1の時エラーになる
        # userのpairは合わせたい
        neg = np.concatenate([self.user_item_pos_pairs[0], 
              neg_items]).reshape(2,-1).T
        self.pos = pos
        self.neg = neg
        
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        out_pos = self.pos[idx]
        out_neg =  self.neg[idx]

        if self.transform:
            out_data = self.transform(out_data)
        
        return np.concatenate([out_pos, out_neg])

    def load_user_item(self):
        user_dict = defaultdict(set)
        for u, item_list in enumerate(open(self.user_item_path).readlines()):
            items = item_list.strip().split()
            # ignore the first element in each line, which is the number of items the user liked. 
            for item in items[1:]:
                user_dict[u].add(int(item))

        n_users = len(user_dict)
        n_items = max([item for items in user_dict.values() for item in items]) + 1

        user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
        for u, item_list in enumerate(open(self.user_item_path).readlines()):
            items = item_list.strip().split()
            # ignore the first element in each line, which is the number of items the user liked. 
            for item in items[1:]:
                user_item_matrix[u, int(item)] = 1
        
        return user_item_matrix
    
    def load_item_feature(self, n_items, tag_occurence_thres=10):
        n_features = 0
        for l in open("citeulike-t/tag-item.dat").readlines():
            items = l.strip().split(" ")
            if len(items) >= tag_occurence_thres:
                n_features += 1
        print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))
        features = dok_matrix((n_items, n_features), dtype=np.int32)
        feature_index = 0
        for l in open("citeulike-t/tag-item.dat").readlines():
            items = l.strip().split(" ")
            if len(items) >= tag_occurence_thres:
                features[[int(i) for i in items], feature_index] = 1
                feature_index += 1


class PosNegBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, user_item_matrix, batch_size):
        user_item_matrix = lil_matrix(user_item_matrix)
        self.user_item_pos_pairs = np.asarray(user_item_matrix.nonzero()).T
        self.user_item_neg_pairs =  (1 -  user_item_matrix.toarray()).nonzero().T
        self.neg_rate = 3
        
        self.dataset = user_item_matrix
        self.batch_size = batch_size
        self.count = 0

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            indices = []
            indices.append(self.user_item_pos_pairs[self.count: self.count+self.batch_size])
            indices.append(self.user_item_neg_pairs[self.neg_rate*self.count: self.neg_rate*(self.count+self.batch_size)])
            # TODO feature追加

            yield indices
            self.count += self.batch_size

    def __len__(self):
        # TODO lenがこれで妥当なのか...
        return len(self.dataset) // self.batch_size


#class PosNegCollater:
#    def __init__(self):
#
#    def __call__(self, batch)
### 
#for _ in tqdm(range(EVALUATION_EVERY_N_BATCHES), desc="Optimizing..."):
#            user_pos, neg = sampler.next_batch()
#            _, loss = sess.run((model.optimize, model.loss),



def sample_function(user_item_matrix, batch_size, n_negative, result_queue, check_negative=True):
    """
    :param user_item_matrix: the user-item matrix for positive user-item pairs
    :param batch_size: number of samples to return
    :param n_negative: number of negative samples per user-positive-item pair
    :param result_queue: the output queue
    :return: None
    """
    user_item_matrix = lil_matrix(user_item_matrix)
    user_item_pairs = numpy.asarray(user_item_matrix.nonzero()).T
    user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_matrix.rows)}
    while True:
        numpy.random.shuffle(user_item_pairs)
        for i in range(int(len(user_item_pairs) / batch_size)):

            user_positive_items_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]

            # sample negative samples
            negative_samples = numpy.random.randint(
                0,
                user_item_matrix.shape[1],
                size=(batch_size, n_negative))

            # Check if we sample any positive items as negative samples.
            # Note: this step can be optional as the chance that we sample a positive item is fairly low given a
            # large item set.
            if check_negative:
                for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                       negative_samples,
                                                       range(len(negative_samples))):
                    user = user_positive[0]
                    for j, neg in enumerate(negatives):
                        while neg in user_to_positive_set[user]:
                            negative_samples[i, j] = neg = numpy.random.randint(0, user_item_matrix.shape[1])
            result_queue.put((user_positive_items_pairs, negative_samples))


class WarpSampler(object):
    """
    A generator that, in parallel, generates tuples: user-positive-item pairs, negative-items
    of the shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, user_item_matrix, batch_size=10000, n_negative=10, n_workers=5, check_negative=True):
        self.result_queue = Queue(maxsize=n_workers*2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(user_item_matrix,
                                                      batch_size,
                                                      n_negative,
                                                      self.result_queue,
                                                      check_negative)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:  # type: Process
            p.terminate()
            p.join()