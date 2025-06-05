def bpe(corpus_str):
    # 准备基础词表,获取所有字符和特殊符号
    base_w = set()
    for s in corpus_str:
        if s not in base_w:
                base_w.update(s)
    print('base_w:',base_w)
    #拆分语料为最小单元
    smallest_w = []
    for s in corpus_str:
        smallest_w.append(s)
    while 1:
        # 统计相邻单元对的频率
        adjoin_dic = get_adjoin_freq(smallest_w)
        # 找到频率最高的组合
        highest_co ,largest= find_max(adjoin_dic)

        if largest<=2:break

        #用新组合合并，生成新的base_w
        base_w = get_new_base_w(base_w,highest_co)
        # 同理更新smallest_w
        smallest_w = get_new_smallest_w(smallest_w, highest_co)

        # if len(base_w)<=10:
        #     break
    return base_w,smallest_w
def get_adjoin_freq(smallest_w):
    adjoin_dic = {}
    for i in range(len(smallest_w)):
        if i == len(smallest_w) - 1: break
        s = smallest_w[i] + smallest_w[i + 1]
        if s not in adjoin_dic:
            adjoin_dic[s] = 1
        else:
            adjoin_dic[s] += 1
    return adjoin_dic
def find_max(adjoin_dic):
    largest=0
    finded=''
    for k,v in adjoin_dic.items():
        if v>largest:
            largest,v= v,largest
            finded=k
    return finded,largest

def get_new_base_w(base_w,highest_co):
    remove_res= []
    for s in base_w:
        if s in highest_co:
            remove_res.append(s)
    for s in remove_res:
        base_w.remove(s)
    base_w.update({highest_co})
    return base_w

def get_new_smallest_w(smallest_w,highest_co):
    new_smallest_w=[]
    i=0
    while 1:
        if i >= len(smallest_w)-1:
            if i ==len(smallest_w)-1:
                new_smallest_w.append(smallest_w[i])
            break
        if smallest_w[i]+smallest_w[i+1]==highest_co:
            new_smallest_w.append(highest_co)
            i+=2
        else:
            new_smallest_w.append(smallest_w[i])
            i+=1
    return new_smallest_w

if __name__ == '__main__':
    base_w,smallest_w = bpe("low lower lowest largest biggest big biger smaller")
    print('finall base_w:',base_w)
    print('finall smallest_w:', smallest_w)