import itertools
import math
from scipy.signal import convolve2d
import numpy as np
import copy
import matplotlib.pyplot as plt

def smart_sub_shit(mask, div_length):
    max_val = max(mask)
    divisor = []
    quotient = []
    subdivisor = []
    found_divisor = False
    for submask in itertools.product(range(-max_val, max_val)[::-1], repeat=div_length):
        if not submask[0]:
            continue
        divisor, remainder = deconvolve(mask, submask)
        if sum(abs(remainder)) == 0:
            quotient = submask
            found_divisor = True
            break
    if not found_divisor:
        if div_length < len(mask)-1:
            smart_sub_shit(mask, div_length+1)
        else:
            print("Cant find shit for {}".format(mask))
    else:
        if len(divisor) > 2:
            smart_sub_shit(divisor.astype("int"), 2)
    #print(divisor)
    #print(subdivisor)
    return {"Quotient": quotient, "Divisor": list(divisor), "Subdivisor": list(subdivisor)}

def deconvolve_mask(mask):
    quoti, div, subdiv = smart_sub_shit(mask)
    sol = quoti,

def get_sub_shit(mask):
    max__nr_sub_masks = len(mask)
    max_val = max(mask)
    min_val = min(mask)
    submasks = None
    all_combs = [list(itertools.product(range(-max_val-1, max_val+1), repeat=length+1))
                                for length in range(max__nr_sub_masks)]
    for nr_sub_masks in range(max__nr_sub_masks, 1, -1):
        all_combs_each_mask = all_combs[:len(all_combs)-(nr_sub_masks-1)]
        combs_of_all_submasks = list(itertools.chain.from_iterable(all_combs_each_mask))
        temp_comb_holder = []
        for i in range(nr_sub_masks):
            temp_comb_holder.append(combs_of_all_submasks)
        all_possible_combs_for_all_masks = list(product(*temp_comb_holder))

        combination_found = False
        for comb in all_possible_combs_for_all_masks:
            print(comb)
            if len(comb) == 2 and comb[0] == (1,0,1) and comb[1] == (1,1):
                x = 1
            convolution_this_comb = do_convolution(list(comb))
            if convolution_this_comb == mask:
                submasks = comb
                combination_found = True
                break
        if combination_found:
            break
    return submasks


def product(*args):
    pools = map(tuple, args)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def do_convolution(*set_of_masks):
    convolution = None
    for masks in set_of_masks:
        if len(masks) > 1:
            convolution = masks[0]
            for mask_i in range(1, len(masks)):
                convolution = list(np.convolve(convolution, masks[mask_i]))
        else:
            convolution = masks
    return convolution

def wtf():
    print("here")
    for a in range(10):
        for b in range(2,4):
            if a==b:
                # Break the inner loop...
                break
        else:
            # Continue if the inner loop wasn't broken.
            continue
        # Inner loop was broken, break the outer.
        break



x = np.ones((100, 100))
x = np.triu(x)
x = np.flip(x,axis = 0)
plt.figure("1")
plt.imshow(x)
plt.show()
z = np.array([[1,0], [0,-1]])
y = convolve2d(x, z)
#print(y)
plt.figure("2")
plt.imshow(y)
plt.show()