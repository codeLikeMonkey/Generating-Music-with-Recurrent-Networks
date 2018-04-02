import re
import numpy as np

def get_music():
    file = "input.txt"
    with open(file,"r") as f:
        file_content = f.read()

    res = re.findall(r"<start>(.*?)<end>",file_content,re.S)

    return res

def input_vector(ch_set,music):
    vectors = np.zeros((len(music) - 1,len(ch_set)))
    for i in range(0,len(music) - 1):
        # print(music[i],i)
        vectors[i,ch_set.index(music[i])] = 1
    return vectors

def target_vector(ch_set,music):
    targets = np.zeros((len(music) - 1 ,1))
    # print(targets)
    for i in range(1,len(music)):
        # print(music[i],i-1)
        targets[i-1] = ch_set.index(music[i])
    return targets


def get_ch_set():
    file = "input.txt"
    ch_set = []
    with open(file,"r") as f:
        for ch in f.read():
            if ch not in ch_set:
                ch_set.append(ch)
    ch_set.sort()
    return ch_set
if __name__ == "__main__":
    # print(get_music()[0])
    ch_set = get_ch_set()
    print(ch_set)
    print(input_vector(ch_set,"129"))
    # print
    # print(target_vector(ch_set,"1269"))


