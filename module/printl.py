#!/usr/bin/env python
def printl(*args):
    strig=[str(i) for i in args]
    string=' '.join(strig)
    print(string)


print('dict:',[1,23,3],'ccc:',[1,23,3])
printl('dict:',[1,23,3],'ccc:',[1,23,3])