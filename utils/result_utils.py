import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os


def simple_read_data(alg):
    print(alg)
    hf = h5py.File("./results/ " +'{}.h5'.format(alg), 'r')
    # hf = h5py.File("./results20210714/" + '{}.h5'.format(alg), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc

def get_training_data_value(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[] ,beta=[]
                            ,algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = [], at_hyper_c = [],  at_num_of_iter = [], at_num_of_method = [], at_qkvsame = [] ,model_net = []):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])
        string_learning_rate = string_learning_rate + "_" +str(beta[i]) + "_" + str(lamb[i])
        if (algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(
                num_users[i]) + "u" + "_" + str(batch_size[i]) + "b" + "_" + str(loc_ep1[i]) + "_" + str(
                k[i]) + "_" + str(personal_learning_rate[i])

        elif (algorithms_list[i] == "pFedAT" or algorithms_list[i] == "pFedAT_p" or algorithms_list[i] == "pFedATold" or
              algorithms_list[i] == "pFedATold_p" or algorithms_list[i] == "pFedAMP" or algorithms_list[
                  i] == "pFedAMP_p" or algorithms_list[i] == "pFedAMPold" or algorithms_list[i] == "pFedAMPold_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(
                num_users[i]) + "u" + "_" + str(batch_size[i]) + "b" + "_" + str(loc_ep1[i]) + "_" + str(
                k[i]) + "_" + str(
                personal_learning_rate[i]) + "_" + "athyperc" + str(at_hyper_c[i]) + "_" + str(
                at_num_of_iter[i]) + "_" + str(
                at_num_of_method[i]) + "_" + str(at_qkvsame[i])

        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(
                num_users[i]) + "u" + "_" + str(batch_size[i]) + "b"  "_" + str(loc_ep1[i])

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset + "_" + model_net[i] + "_" + algorithms_list[i] + "_avg"))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]
    return glob_acc, train_acc, train_loss


def get_all_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0, beta=0,
                                algorithms="", batch_size=0, dataset="", k=0, personal_learning_rate=0, times=5,
                                model_net="", at_hyper_c=1000, at_num_of_iter=2, at_num_of_method=2, at_qkvsame=0):
    train_acc = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    glob_acc = np.zeros((times, Numb_Glob_Iters))
    algorithms_list = [algorithms] * times
    for i in range(times):
        string_learning_rate = str(learning_rate)
        string_learning_rate = string_learning_rate + "_" + str(beta) + "_" + str(lamb)
        if (algorithms == "pFedMe" or algorithms == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(
                num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1) + "_" + str(k) + "_" + str(
                personal_learning_rate) + "_" + str(i)
            # algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" +str(loc_ep1) + "_" + str(k) + "_" + str(personal_learning_rate) + "_" + "athyperc" + str(at_hyper_c) + "_" + str(at_num_of_iter) + "_" + str(at_num_of_method) + "_" + str(at_qkvsame) + "_" + str(i)
        elif (
                algorithms == "FedAvgAdd" or algorithms == "FedAvgAdd_p" or algorithms == "pFedAT" or algorithms == "pFedAT_p" or algorithms == "pFedATold" or algorithms == "pFedATold_p" or algorithms == "pFedAMP" or algorithms == "pFedAMP_p" or algorithms == "pFedAMPold" or algorithms == "pFedAMPold_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(
                num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1) + "_" + str(k) + "_" + str(
                personal_learning_rate) + "_" + "athyperc" + str(at_hyper_c) + "_" + str(at_num_of_iter) + "_" + str(
                at_num_of_method) + "_" + str(at_qkvsame) + "_" + str(i)
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(
                num_users) + "u" + "_" + str(batch_size) + "b"  "_" + str(loc_ep1) + "_" + str(i)

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset + "_" + model_net + "_" + algorithms_list[i]))[:, :Numb_Glob_Iters]
    return glob_acc, train_acc, train_loss


def get_data_label_style(input_data=[], linestyles=[], algs_lbl=[], lamb=[], loc_ep1=0, batch_size=0):
    data, lstyles, labels = [], [], []
    for i in range(len(algs_lbl)):
        data.append(input_data[i, ::])
        lstyles.append(linestyles[i])
        labels.append(algs_lbl[i] + str(lamb[i]) + "_" +
                      str(loc_ep1[i]) + "e" + "_" + str(batch_size[i]) + "b")

    return data, lstyles, labels


def average_data(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb="", learning_rate="", beta="", algorithms="",
                 batch_size=0, dataset="", k="", personal_learning_rate="", times=5, model_net="", at_hyper_c=1000,
                 at_num_of_iter=2, at_num_of_method=2, at_qkvsame=0):
    if (algorithms == "PerAvg"):
        algorithms = "PerAvg_p"
    glob_acc, train_acc, train_loss = get_all_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                  learning_rate, beta, algorithms, batch_size, dataset,
                                                                  k, personal_learning_rate, times, model_net,
                                                                  at_hyper_c, at_num_of_iter, at_num_of_method,
                                                                  at_qkvsame)
    glob_acc_data = np.average(glob_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    # store average value to h5 file
    max_accurancy = []
    for i in range(times):
        max_accurancy.append(glob_acc[i].max())

    print("std:", np.std(max_accurancy))
    print("Mean:", np.mean(max_accurancy))

    alg = dataset + "_" + model_net + "_" + algorithms
    alg = alg + "_" + str(learning_rate) + "_" + str(beta) + "_" + str(lamb) + "_" + str(num_users) + "u" + "_" + str(
        batch_size) + "b" + "_" + str(loc_ep1)
    if (algorithms == "pFedMe" or algorithms == "pFedMe_p"):
        alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)

    if (
            algorithms == "FedAvgAdd" or algorithms == "FedAvgAdd_p" or algorithms == "pFedAT" or algorithms == "pFedAT_p" or algorithms == "pFedATold" or algorithms == "pFedATold_p" or algorithms == "pFedAMP" or algorithms == "pFedAMP_p" or algorithms == "pFedAMPold" or algorithms == "pFedAMPold_p"):
        alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)
    # gotodo:change "pFedMe" and "pFedMe_p" to "pFedAT" "pFedAT_p"
    if (
            algorithms == "FedAvgAdd" or algorithms == "FedAvgAdd_p" or algorithms == "pFedAT" or algorithms == "pFedAT_p" or algorithms == "pFedATold" or algorithms == "pFedATold_p" or algorithms == "pFedAMP" or algorithms == "pFedAMP_p" or algorithms == "pFedAMPold" or algorithms == "pFedAMPold_p"):
        alg = alg + "_" + "athyperc" + str(at_hyper_c) + "_" + str(at_num_of_iter) + "_" + str(
            at_num_of_method) + "_" + str(at_qkvsame)

    alg = alg + "_" + "avg"
    if (len(glob_acc) != 0 & len(train_acc) & len(train_loss)):
        with h5py.File("./results/" + '{}.h5'.format(alg, loc_ep1), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            hf.close()
