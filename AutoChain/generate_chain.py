import numpy as np
import math


def generate_one_chain(h, w, type='m', shrinking_level=4):
    '''
    Input:
        - h: an integer, c_out
        - w: an integer, k*c_in
        - type: a string: m or n, which indicates the type of chain.
        - shrinking_level: an integer, which decides the shrinking speed of the chain.
        Larger the number, slower the shrinking speed. we suggest to set shrinking_level
        in the interval [4,8]. Please set this parameter properly according to the matrix
        size.

    Return:
        - sup: a list, which is the superscripts of the factors from right to left.
        - sub: a list, which is the subscripts of the factors from right to left.

    The parameters r, s can be modified as well.
    '''

    w_channel = w / (3*3)
    sup = []
    sub = []
    if type == 'm':
        if h == w_channel:
            sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), w])
            sub.append([int(math.pow(2, 3)), 3*3, 1])
            for i in range(shrinking_level-5):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(2):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                # r = 2
                # s = 4
                r = 1
                s = 2
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel / 2:
            sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), w])
            sub.append([int(math.pow(2, 3)), 3*3, 1])
            for i in range(shrinking_level-5):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(3):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel * 2:
            sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), w])
            sub.append([int(math.pow(2, 3)), 3*3, 1])
            for i in range(shrinking_level-3):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 1
                s = 2
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
    elif type == 'b':
        bulging_rate = 4/3
        sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))) * bulging_rate), w])
        rt = 1
        s = 6
        r = s * bulging_rate
        sub.append([int(r), int(s), int(rt)])
        sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))) * bulging_rate)])
        s = 6
        r = s * bulging_rate / (2*2)
        rt = sub[-1][0] * sub[-1][2]
        sub.append([int(r), int(s), int(rt)])
        if h == w_channel:
            for i in range(shrinking_level-5-1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(2):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel / 2:
            for i in range(shrinking_level-5-1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(3):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel * 2:
            for i in range(shrinking_level-3-1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
    else:
        raise NotImplementedError('Please choose either m for monotonic chain and b for bulging chain')

    n_factors = len(sub)
    # print('Chain Type: {}.'.format(type))
    # print('Shrinking Speed: {}.'.format(shrinking_speed))
    # print('Number of factors: {}.'.format(n_factors))
    # print('Superscript: {}.'.format(sup))
    # print('Subscript: {}.'.format(sub))
    return sup, sub, n_factors


def generate_debut_chains(cfg, type='m', shrinking_level=4):
    all_chains = []
    for h, w, in cfg:
        sup, sub, n_factors = generate_one_chain(h, w, type=type, shrinking_level=shrinking_level)
        debut_chain = []
        for i in range(n_factors):
            debut_chain.append(sup[i] + sub[i])
        all_chains.append(debut_chain)
    return all_chains


if __name__ == "__main__":
    cfg_vgg_like = [[64, 32*3*3], [128, 64*3*3], [256, 128*3*3]]
    # shrinking_level 3 -> 5
    cfg_resnet = [[16, 144], [32, 144], [32, 288], [64, 288], [64, 576]]
    # shrinking_level 3 -> 7
    cfg_vgg = [[64, 576], [128, 576], [128, 1152], [256, 1152], [256, 2304], [512, 2304], [512, 4608]]
    r_shapes = generate_debut_chains(cfg_vgg, type='m', shrinking_level=7)
    for chain in r_shapes:
        print(chain)
    print("total number of factors: ", sum(len(chain) for chain in r_shapes))
    # print(r_shapes)