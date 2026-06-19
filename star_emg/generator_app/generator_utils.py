import numpy as np


def check_list(text):
    if text == "":
        return
    try:
        lst = [float(v.strip()) for v in text.strip("[]").split(",") if v.strip() != ""]
    except:
        return
    if len(lst) == 1:
        return lst[0]
    return lst


def get_from_range(range):
    min, max = range[0], range[1]
    random_value = np.round(np.random.uniform(min, max), 3)
    if int(random_value) == random_value:
        return int(random_value)
    return random_value
