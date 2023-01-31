#!/usr/bin/env python

class Node():
    def __init__(self,val):
        self.next=None
        self.val=val

class Link():
    def __init__(self):
        self.head=None
    def add_node(self,val):
        p = self.head
        if self.head is None:
            self.head=Node(val)
            return
        while p.next:
            p=p.next
        p.next=Node(val)
    def print_link(self):
        p=self.head
        while p:
            print(p.val)
            p=p.next

if __name__ == '__main__':
    link=Link()
    link.add_node(1)
    link.add_node(2)
    link.add_node(3)
    link.add_node(6)
    link.add_node(5)
    link.print_link()

