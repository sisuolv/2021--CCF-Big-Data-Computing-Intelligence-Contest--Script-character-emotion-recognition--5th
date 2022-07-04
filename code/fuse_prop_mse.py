# -*- coding: UTF-8 -*-
import random
import os
import json
import numpy as np
import pandas as pd

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

def get_all_res(data_dir):
    all_results = []
    init = False
    all_file = get_files(data_dir)
    file_count = len(all_file)
    print("file_count ",file_count)


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

        for id, line in enumerate(all_line):
            porp_list = line.strip().split(",")
            for j in range(nub_labs):
                all_results[id][j] += float(porp_list[j])

        fr.close()

    all_results_new = []
    for res_tmp in all_results:
        ttmp = []

        for rr in res_tmp:
            rr_tmp = rr / file_count

            if rr_tmp < 0:
                rr_tmp = 0
            if rr_tmp > 3:
                rr_tmp = 3

            ttmp.append(rr_tmp)

        all_results_new.append(ttmp)


    return all_results_new



all_results_5 = get_all_res("./user_data/nezha_11_11")
all_results_7 = get_all_res("./user_data/bert_11_15")
all_results_10 = get_all_res("./user_data/nezha_11_20")
all_results_11 = get_all_res("./user_data/bert_11_20")



f_name = "./prediction_result/result.csv"

fw = open(f_name, "w",encoding="utf-8")
fw.write("id\temotion\n")

data_test_fr = open("./raw_data/test_dataset.tsv", "r")
data_test = data_test_fr.readlines()[1:]



for i, data_tmp in enumerate(data_test):
    data_tmp = data_tmp.strip("\r")
    data_tmp = data_tmp.strip("\n")
    test_id = data_tmp.split("\t")[0]

    prop_tmp_5 = all_results_5[i]
    prop_tmp_7 = all_results_7[i]
    prop_tmp_10 = all_results_10[i]
    prop_tmp_11 = all_results_11[i]

    lab_all = ""


    for idx, kk in enumerate(prop_tmp_7):


        #best
        kk_ = 0.45 * (0.6 * prop_tmp_7[idx] + 0.4 * prop_tmp_11[idx]) + 0.55 * (0.5* prop_tmp_10[idx]  + 0.5* prop_tmp_5[idx])



        if kk_ < 0:
            kk_ = 0
        if kk_ > 3:
            kk_ = 3


        lab_all = lab_all + str(kk_) + ","

    output_line = test_id + "\t" + lab_all[:-1] + "\n"
    fw.write(output_line)

fw.close()