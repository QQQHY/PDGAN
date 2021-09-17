import linecache
import numpy as np
import utils as ut


# Get batch data from training set
def get_batch_data(file, index, size):  # 1,5->1,2,3,4,5
    user = []
    item = []
    label = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip()
        line = line.split()
        user.append(int(line[0]))
        user.append(int(line[0]))
        item.append(int(line[1]))
        item.append(int(line[2]))
        label.append(1.)
        label.append(0.)
    return user, item, label

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Get category of items
def get_category(file_in):
    category = {}
    #with open(file_in) as fin:
    with open(file_in,encoding='unicode_escape') as fin:
        for line in fin:
            line = line.split('|')
            iid = int(line[0]) - 1  # item id starts from 0
            category[iid] = line[6:24]
    return category


# Get training/testing data
def get_train_test_data(file_in):
    # only record user-item pairs with rating >=4
    user_item = {}
    with open(file_in) as fin:
        for line in fin:
            line = line.split()
            uid = int(line[0])
            iid = int(line[1])
            r = float(line[2])
            if uid in user_item:
                user_item[uid].append(iid)
            else:
                user_item[uid] = [iid]
    return user_item


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        # if method == 0:
        #     return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        # elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        # else:
        #     raise ValueError('method must be 0 or 1.')
    else:
        return 0.


def ndcg_at_k(r, k):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def diversity_by_category(selected_items, item_cate, cate_num):
    cate = []
    for iid in selected_items:
        try:
            cate.append(item_cate[iid])
        except KeyError:
            pass

    cate_count = np.count_nonzero(np.sum(np.asarray(cate, np.float32), axis=0))

    return cate_count/cate_num


def get_div_train_data(file_in):
    user_train_samples = {}

    with open(file_in) as fin:
        for line in fin:
            line = line.split('\t')
            uid = int(line[0])
            items = list(map(int, line[1:]))
            if uid in user_train_samples:
                user_train_samples[uid].append(items)
            else:
                user_train_samples[uid] = [items]

    return user_train_samples


def generate_pairwise_diversity_training_data(file_train, file_cate, file_out_pos, file_out_neg, user_num):
    pos_data = []  # data for output
    neg_data = []

    #########################################################################################
    # Load data
    #########################################################################################
    category = get_category(file_cate)
    user_item = get_train_test_data(file_train)

    # for each user, generate diversity set
    for i in range(0, user_num):
        uid = i;
        print('user:', uid)
        try:
            items = user_item[uid]
        except KeyError:
            pass
        # the number of trials for each user is set to be the number of viewed items
        for j in range(0, len(items)):
            first_item = items[j]
            pos_div_set = [first_item]  # make sure each viewed item is sampled
            pos_cate = [category[first_item]]
            num_cate = np.count_nonzero(np.sum(np.asarray(pos_cate, np.float32), axis=0))
            # the number of trials for each diversity set is the number of viewed items
            for k in range(0, len(items)):
                new_item = np.random.choice(items)
                try:
                    pos_cate.append(category[new_item])
                    new_num_cate = np.count_nonzero(np.sum(np.asarray(pos_cate, np.float32), axis=0))
                    if new_num_cate - num_cate > 0:
                        pos_div_set.append(new_item)
                        num_cate = new_num_cate
                    if len(pos_div_set) == 10:
                        break;
                except KeyError:
                    pass

            pos_div_set.sort()
            pos_data.append(str(uid) + '\t' + '\t'.join(str(x) for x in pos_div_set))

            neg_div_set = [first_item]  # make sure each viewed item is sampled
            neg_cate = np.asarray(category[first_item], np.int32).nonzero()[0]
            # the number of trials for each diversity set is the number of viewed items
            for k in range(0, len(items)):
                new_item = items[k]
                try:
                    if new_item not in neg_div_set:
                        new_cate = np.asarray(category[new_item], np.int32).nonzero()[0]
                        if np.array_equal(neg_cate, new_cate):
                            neg_div_set = np.append(neg_div_set, new_item)
                            if len(neg_div_set) > 10:  # due to tensorflow bug
                                neg_div_set = np.random.choice(neg_div_set, 10, replace=False)
                except KeyError:
                    pass
            neg_div_set.sort()
            neg_data.append(str(uid) + '\t' + '\t'.join(str(x) for x in neg_div_set))

    with open(file_out_pos, 'w')as fout:
        fout.write('\n'.join(pos_data))

    with open(file_out_neg, 'w')as fout:
        fout.write('\n'.join(neg_data))



if __name__ == '__main__':
    workdir = 'ml-100k/'

    # file_cate = workdir + 'ml-100k/u.item'
    # file_train = workdir + 'movielens-100k-train.txt'
    # user_num = 943
    # item_num = 1682
    #
    # file_out_pos = workdir + 'pos_diversity_train.txt'
    # file_out_neg = workdir + "neg_diversity_train.txt"
    #
    # generate_pairwise_diversity_training_data(file_train, file_cate, file_out_pos, file_out_neg, user_num)

    CATE_NUM = 18
    user_pos_train = ut.get_train_test_data((workdir + 'movielens-100k-train.txt'))
    item_cate = ut.get_category(workdir + 'ml-100k/u.item')

    div = 0
    for u in range(len(user_pos_train)):
        print(u)
        div += ut.diversity_by_category(user_pos_train[u], item_cate, CATE_NUM)

    print(div/len(user_pos_train))