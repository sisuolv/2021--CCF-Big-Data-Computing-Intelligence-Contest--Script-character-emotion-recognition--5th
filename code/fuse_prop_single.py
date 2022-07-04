# -*- coding: UTF-8 -*-
import random
import os
import json
import numpy as np
import pandas as pd



data_dir = "./nezha"
nub_labs = 1


def get_files(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                # print(file)
                L.append(os.path.join(root, file))
    return L

all_result_final = {}

for idx_lab in range(2):
    dir_new = data_dir + "/" + str(idx_lab) + "/"
    all_file = get_files(dir_new)

    all_results = []
    init = False

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
        # if file.find("bert_64") > -1:
        #     #ttt = 0.9
        #     ttt = 1.0

        for id, line in enumerate(all_line):
            line = line.strip("\n")
            line = line.strip("[")
            line = line.strip("]")
            porp_list = line.strip().split(",")
            for j in range(nub_labs):
                all_results[id][j] += ttt * float(porp_list[j])

        fr.close()

    file_count = len(all_file)
    print("file_count %d" % (file_count))

    #file_count = len(all_file) / 2
    file_count = len(all_file)
    print("file_count %d" % (file_count))

    f_name = "sub_pos_" + str(idx_lab) + ".tsv"

    fw = open(f_name, "w",encoding="utf-8")
    fw.write("id\temotion\n")

    data_test_fr = open("./data/test_dataset.tsv", "r")
    data_test = data_test_fr.readlines()[1:]


    for i, data_tmp in enumerate(data_test):
        data_tmp = data_tmp.strip("\r")
        data_tmp = data_tmp.strip("\n")
        test_id = data_tmp.split("\t")[0]

        res_tmp = all_results[i]
        lab_all = res_tmp[0] / file_count

        output_line = test_id + "\t" + str(lab_all) + "\n"
        fw.write(output_line)

    fw.close()