
def update_f1(prediction, gold, TP, FP, FN):
    for term in prediction:
        if term in gold:
            TP += 1
        else:
            FP += 1
    for term in gold:
        if term not in prediction:
            FN += 1
    return TP, FP, FN


def compute_f1(TP, FP, FN):
    if TP == 0:
        p, r, f = 0, 0, 0
    else:
        p = 100 * TP / (TP + FP)
        r = 100 * TP / (TP + FN)
        f = 100 * 2 * TP/(2 * TP + FN + FP)
    return p, r, f
