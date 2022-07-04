# -*- coding: UTF-8 -*-
import random
import os
import json
import numpy as np
import pandas as pd

''''''
#加载已有结果
all_results_0 = []
fr_best = open("sub_ccf_emo_1120_macbert2_nezha2_45_55.tsv","r")
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_0.append(prop_l_tmp)

all_results_2 = []
fr_best = open("sub_ccf_emo_final_nezha_11_04.tsv","r")   #0.71007136931
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_2.append(prop_l_tmp)

all_results_4 = []
fr_best = open("sub_ccf_emo_final_nezha_11_10.tsv","r")   #0.71060107199
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_4.append(prop_l_tmp)



all_results_9 = []
fr_best = open("sub_ccf_emo_final_nezha_11_18.tsv","r")  #0.71058330047
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_9.append(prop_l_tmp)


all_results_10 = []
fr_best = open("sub_ccf_emo_final_nezha_11_20.tsv","r")  #0.71188507483
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_10.append(prop_l_tmp)

# bert。。。

all_results_13 = []
fr_best = open("sub_ccf_emo_final_bert_11_07.tsv","r")  #0.71176829944
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_13.append(prop_l_tmp)

all_results_7 = []
fr_best = open("sub_ccf_emo_final_bert_11_15.tsv","r")  #0.71176829944
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_7.append(prop_l_tmp)


all_results_11 = []
fr_best = open("sub_ccf_emo_final_bert_11_20.tsv", "r")  # 0.71094230250
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_11.append(prop_l_tmp)


all_results_12 = []
fr_best = open("sub_ccf_emo_final_bert_11_13.tsv","r")  #0.70900025747
datt = fr_best.readlines()[1:]
for line in datt:
    line = line.strip("\n")
    line_prop = line.split("\t")[1]
    line_prop_l = line_prop.split(",")
    prop_l_tmp = []
    for ll in line_prop_l:
        prop_l_tmp.append(float(ll))

    all_results_12.append(prop_l_tmp)



f_name = "sub_ccf_emo_11_22_nezha3_macbert3_final_4.tsv"

fw = open(f_name, "w",encoding="utf-8")
fw.write("id\temotion\n")

data_test_fr = open("./data/test_dataset.tsv", "r")
data_test = data_test_fr.readlines()[1:]


for i, data_tmp in enumerate(data_test):
    data_tmp = data_tmp.strip("\r")
    data_tmp = data_tmp.strip("\n")
    test_id = data_tmp.split("\t")[0]


    prop_tmp_0 = all_results_0[i]               # all 0.71138006

    #nezha
    prop_tmp_2 = all_results_2[i]               # nezha_11_04 seed 72               0.71007136931   0.710268921042249
    prop_tmp_4 = all_results_4[i]               # nezha_11_10，seed 1024            0.71060107199   0.7126515230617073
    prop_tmp_9 = all_results_9[i]               # nezha_11_18，8fold， seed 12345   0.71058330047    0.710650664958913
    #prop_tmp_10 = all_results_10[i]             # nezha_11_20，10fold， seed 71     0.71188507483    0.7091948249152256


    #bert
    prop_tmp_13 = all_results_13[i]              #bert_11_07  seed 72               0.71001905381    0.7088990559571055
    prop_tmp_12 = all_results_12[i]              #bert_11_13  seed 43               0.70900025747    0.7089472616671687
    #prop_tmp_7 = all_results_7[i]                #bert_11_15  seed 2020  8fold      0.71176829944    0.7057327697828087
    prop_tmp_11 = all_results_11[i]              #bert_11_20  seed 71    10fold     0.71094230250    0.7096762063119376


    lab_all = ""

    for idx, kk in enumerate(prop_tmp_13):

        #kk_ = 0.55* ( prop_tmp_2[idx]+ prop_tmp_4[idx]+ prop_tmp_9[idx]+ prop_tmp_10[idx] ) / 4 + 0.45* (prop_tmp_13[idx]+ prop_tmp_12[idx]+ prop_tmp_7[idx]+ prop_tmp_11[idx]) / 4

        kk_ = 0.51 * ( 0.3 * prop_tmp_2[idx] + 0.3 * prop_tmp_4[idx] + 0.4 * prop_tmp_9[idx] ) + 0.49 * (
                    0.3 * prop_tmp_13[idx] + 0.3 * prop_tmp_12[idx]  + 0.4 * prop_tmp_11[idx])



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