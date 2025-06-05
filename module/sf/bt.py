class Nd(object):
    def __init__(self,value):
        self.val=value
        self.left=None
        self.right=None
class Bt(object):
    def __init__(self):
        self.rt=None
    def add(self,nd):
        if self.rt is None:
            self.rt=nd
        else:
            queue=[self.rt]
            while queue:
                cur_nd= queue.pop(0)
                if cur_nd.left:
                    queue.append(cur_nd.left)
                else:
                    cur_nd.left=nd
                    return
                if cur_nd.right:
                    queue.append(cur_nd.right)
                else:
                    cur_nd.right=nd
                    return
    def width_search(self):
        queue = [self.rt]
        result=[]
        while queue:
            nd=queue.pop(0)
            result.append(nd.val)
            if nd.left:
                queue.append(nd.left)
            if nd.right:
                queue.append(nd.right)
        print(result)

    def pre_search(self,rt,result):
        if rt is None:
            return
        result.append(rt.val)
        self.pre_search(rt.left,result)
        self.pre_search(rt.right,result)


    def pre_search_ndg(self):
        stack=[self.rt]
        result=[]
        while stack:
            nd = stack.pop()
            result.append(nd.val)
            if nd.right:
                stack.append(nd.right)
            if nd.left:
                stack.append(nd.left)
        print(result)













if __name__ == '__main__':
    bt=Bt()
    for i in range(10):
        nd=Nd(i)
        bt.add(nd)
    bt.width_search()
    res=[]
    bt.pre_search(bt.rt,res)
    print(res)
    bt.pre_search_ndg()
    
