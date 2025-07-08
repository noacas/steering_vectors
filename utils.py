def dict_subtraction(d1, d2):
    ret = dict()
    for k in d1.keys():
        ret[k] = d1[k] - d2[k]
    return ret