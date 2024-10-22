import numpy as np
import random


def partition_noniid(dataset, num_users, alpha, sigma, cifar):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    if cifar:
        labels = dataset.targets
    else:
        labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    num_class = 10

    ##
    numdata = 40000
    K = num_users
    x = np.arange(1, K + 1)
    weights = np.float_power(x, (-sigma))
    weights /= weights.sum()
    sample = weights * numdata
    num_items = np.around(sample)
    sum = num_items.sum()
    # print('sum ' + str(sum))
    rndValues = np.random.choice(K, 10 * K, replace=True)
    t = 0
    if sum - numdata > 0:
        # deduct
        count = sum - numdata
        while t <= count:
            if num_items[rndValues[t]] - 1 > 0:
                num_items[rndValues[rndValues[t]]] -= 1
                t += 1
            else:
                count += 1
                t += 1
    else:
        # addition
        count = numdata - sum
        while t < count:
            num_items[rndValues[rndValues[t]]] += 1
            t += 1
    # print(num_items)

    # put idx into class wise idx
    classviseidx = {i: np.array([], dtype='int64') for i in range(num_class)}
    for i in range(len(dataset)):
        classviseidx[idxs_labels[1, i]] = np.append(classviseidx[idxs_labels[1, i]], idxs_labels[0, i])
    output = {'user' + str(i): np.array([], dtype='int64') for i in range(num_users)}

    # Dirichelet distributed idx distribution

    if alpha > 0:
        classViseallocationoverUsers = {i: np.array([], dtype='int64') for i in range(num_users)}
        output = {'user' + str(i): np.array([], dtype='int64') for i in range(num_users)}
        if alpha > 10000000000000:
            # consider as infinity
            for i in range(num_users):
                k = np.array([], dtype='int64')
                for j in range(num_class):
                    k = np.append(k, 1 / num_class)
                classViseallocationoverUsers[i] = np.round(k * num_items[i])
                # output['user' + str(i)] = np.round(k * totalperuser)
        else:
            for i in range(num_users):
                k = np.random.gamma(alpha, 1, num_class)
                k = k / np.sum(k)

                classViseallocationoverUsers[i] = np.round(k * num_items[i])
                # print('classwise user ' + str(i ) + ' ' + str(classViseallocationoverUsers[i]))
                # output['user' + str(i)] = np.round(k * totalperuser)
        # print('classwise allocation over users ' + str(classViseallocationoverUsers))
        # divide and assign
        for i in range(num_users):
            for j in range(num_class):

                num_choice = min(int(classViseallocationoverUsers[i][j]), len(classviseidx[j]))
                if num_choice == 0:
                    print('no items to select')
                rand_set = np.random.choice(classviseidx[j], num_choice, replace=False)
                # print('rand set ' + str(rand_set.shape))
                # remove selected list
                templst = list(classviseidx[j])
                for rem_itm in rand_set:
                    templst.remove(rem_itm)
                classviseidx[j] = np.array(templst, dtype='int64')
                dict_users[i] = np.append(dict_users[i], rand_set)

    else:
        for i in range(num_users):
            k = np.zeros(10)
            j = random.randint(0, 9)
            k[j] = 1
            rand_set = np.random.choice(classviseidx[j], int(num_items[i]), replace=False)
            templst = list(classviseidx[j])
            for rem_itm in rand_set:
                templst.remove(rem_itm)
            classviseidx[j] = np.array(templst, dtype='int64')
            dict_users[i] = np.append(dict_users[i], rand_set)
            output['user' + str(i)] = np.round(k * num_items[i])

    # number of items and variance computation
    num_items = []
    for indexterm in dict_users:
        num_items.append(len(dict_users[indexterm]))
    num_items = np.array(num_items)
    variance = np.var(num_items, ddof=0)
    # sio.savemat('classvisenoniid.mat', output)
    print('output ' + str(output))
    return [dict_users, num_items, variance]
