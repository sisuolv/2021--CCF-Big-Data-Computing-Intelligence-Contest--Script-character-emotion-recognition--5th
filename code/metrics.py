"""
定义各类性能指标
created by syzong
2020/4
"""
from sklearn.metrics import roc_auc_score

def binary_auc(pred_y, true_y):
    """
    二类别的auc值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    auc = roc_auc_score(true_y, pred_y)
    return auc


def mean(item: list) -> float:
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc

def binary_precision(pred_y, true_y, positive=1):
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def multi_precision(pred_y, true_y, labels):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec

def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    #print("F1 score all ",f_betas)
    f_beta = mean(f_betas)
    return f_beta

def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta