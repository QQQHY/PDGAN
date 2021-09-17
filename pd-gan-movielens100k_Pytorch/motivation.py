import utils as ut
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# Get training/testing data
def get_data(file_in, user_item):
    # only record user-item pairs with rating >=4
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


if __name__ == '__main__':
    USER_NUM = 943
    ITEM_NUM = 1683
    CATE_NUM = 18

    workdir = 'ml-100k/'
    file_cate = workdir + 'ml-100k/u.item'
    user_item = {}
    user_item = get_data((workdir + 'movielens-100k-train.txt'), user_item)
    user_item = get_data(workdir + 'movielens-100k-test.txt', user_item)

    category = ut.get_category(file_cate)

    # for each user, find number of cate
    user_cate = []
    for i in range(0, USER_NUM):
        uid = i
        print('user:', uid)
        try:
            items = user_item[uid]
            # find cate for each item
            for j in range(0, len(items)):
                if j == 0:
                    first_item = items[j]
                    pos_cate = [category[first_item]]
                    num_cate = np.count_nonzero(np.sum(np.asarray(pos_cate, np.float32), axis=0))
                # the number of trials for each diversity set is the number of viewed items
                else:
                    new_item = items[j]
                    try:
                        pos_cate.append(category[new_item])
                        num_cate = np.count_nonzero(np.sum(np.asarray(pos_cate, np.float32), axis=0))
                    except KeyError:
                        pass
            user_cate.append(num_cate)
        except KeyError:
            user_cate.append(num_cate)
    user_cate = np.asarray(user_cate)
    num_users = []
    for i in range(6, CATE_NUM + 1):
        index = np.where(user_cate == i)
        num = len(index[0])
        num_users.append(num)

    x = np.arange(18-6+1)

    fig, ax = plt.subplots()
    plt.bar(x, num_users)
    # plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
    plt.show()

