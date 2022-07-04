# -*- coding: UTF-8 -*-
import random
import os
import json
import numpy as np
import pandas as pd

init = False

data_dir = "./bert_1107"
#data_dir = "./bert"
nub_labs = 6

def get_files(file_dir):
    L = []
    count = 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.find(".txt") > 0:
            #if os.path.splitext(file)[1] == '1.txt':
                # print(file)
                L.append(os.path.join(root, file))
                count += 1
        # if count == 2:
        #     break
    return L


all_file = get_files(data_dir)

all_texts = []
all_results = []
all_num = 0

for file in all_file:
    fr = open(file, "r", encoding="utf-8")

    all_line = fr.readlines()
    print("read file %s, len %d" % (file, len(all_line)))

    if init == False:
        init = True
        for i in range(len(all_line)):
            prop = []
            for j in range(nub_labs):
                prop.append(0.0)

            all_results.append(prop)

    lab_num = 0
    ttt = 1.0
    # if file.find("320") > -1:
    #     #ttt = 0.9
    #     ttt = 0.35
    #
    # if file.find("10_23") > -1:
    #     ttt = 0.5
    #
    # if file.find("nezha") > -1:
    #     ttt = 0.55
    #
    # if file.find("256") > -1:
    #     #ttt = 0.9
    #     ttt = 0.25

    print("ttt ",ttt)
    for id, line in enumerate(all_line):
        porp_list = line.strip().split(",")
        for j in range(nub_labs):
            # ppp = float(porp_list[j])
            # if ppp < 0:
            #     ppp = 0
            # if ppp > 3:
            #     ppp = 3

            all_results[id][j] += ttt * float(porp_list[j])

    fr.close()

    all_num += 1

#file_count = len(all_file) / 1
file_count = len(all_file)


print("file_count %d" % (file_count))


f_name = "sub_ccf_emo_final_bert_11_07.tsv"


fw = open(f_name, "w",encoding="utf-8")
fw.write("id\temotion\n")

data_test_fr = open("./data/test_dataset.tsv", "r")
data_test = data_test_fr.readlines()[1:]


for i, data_tmp in enumerate(data_test):
    data_tmp = data_tmp.strip("\r")
    data_tmp = data_tmp.strip("\n")
    test_id = data_tmp.split("\t")[0]

    prop_tmp = all_results[i]


    lab_all = ""


    for idx, kk in enumerate(prop_tmp):

        kk_ = kk / file_count



        # if idx == 0:
        #     kk_ = 0.8 * kk_ + 0.2 * pos_0_res
        #
        # if idx == 1:
        #     kk_ = 0.8 * kk_ + 0.2 * pos_1_res
        #
        # if idx == 2:
        #     kk_ = pos_2_res
        #
        # if idx == 3:
        #     kk_ = pos_3_res

        if kk_ < 0:
            kk_ = 0
        if kk_ > 3:
            kk_ = 3

        # if kk_ < 0.05:
        #     kk_ = 0
        #
        # if 0.95 < kk_ < 1.05:
        #     kk_ = 1
        #
        # if 1.95 < kk_ < 2.05:
        #     kk_ = 2
        #
        # if 2.95 < kk_ < 3.1:
        #     kk_ = 3

        lab_all = lab_all + str(kk_) + ","

    output_line = test_id + "\t" + lab_all[:-1] + "\n"
    fw.write(output_line)

fw.close()