# 前序遍历：根结点->左子树->右子树
# 中序遍历：左子树->根结点->右子树
# 后序遍历：左子树->右子树->根结点

class Node(object):
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None


class BinaryTree(object):
    def __init__(self):
        self.root = None

    def add(self, node):
        if self.root == None:
            self.root = node
        else:
            queue = [self.root]
            while queue:
                cur_node = queue.pop(0)
                if cur_node.left:
                    queue.append(cur_node.left)
                else:
                    cur_node.left = node
                    return
                if cur_node.right:
                    queue.append(cur_node.right)
                else:
                    cur_node.right = node
                    return
    #层次遍历
    def breadth_traversal(self):
        queue=[self.root]
        res=[]
        while queue:
            cur_node=queue.pop(0)
            res.append(cur_node.val)
            if cur_node.left:
                queue.append(cur_node.left)
            if cur_node.right:
                queue.append(cur_node.right)
        return res
    #先序遍历 根左右
    def pre_traversal(self):
        stack=[self.root]
        res=[]
        while stack:
            cur_node=stack.pop()
            res.append(cur_node.val)
            if cur_node.right:
                stack.append(cur_node.right)
            if cur_node.left:
                stack.append(cur_node.left)
        return res
    #中序遍历 左根右 递归法
    def mid_traversal_recursion(self,root):
        #终止条件
        if not root:
            return []
        if not root.right and not root.left:
            return [root.val]
        #分治+递归
        left_tree =root.left
        right_tree =root.right
        left=self.mid_traversal_recursion(left_tree)
        right=self.mid_traversal_recursion(right_tree)
        return left+[root.val]+right








if __name__ == '__main__':
    bt = BinaryTree()
    bt.add(Node(1))
    bt.add(Node(2))
    bt.add(Node(3))
    bt.add(Node(4))
    bt.add(Node(5))
    bt.add(Node(6))
    print(bt.breadth_traversal())
    print(bt.pre_traversal()) # 1,2,4,5,3,6
    print(bt.mid_traversal_recursion(bt.root)) # 4,2,5,1,6,3

