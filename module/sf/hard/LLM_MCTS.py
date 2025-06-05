import math
import random

class MockLLM:
    def generate_actions(self, state):
        # 修复1：放宽条件匹配逻辑
        if "订单" in state and ("未收到" in state or "没收到" in state):
            return [
                "请提供订单号",
                "我们会补偿优惠券",
                "物流延迟请等待"
            ]
        return []

    def evaluate_value(self, action, state):
        if "提供订单号" in action:
            return random.uniform(0.1, 0.2)
        elif "补偿" in action:
            return random.uniform(0.4, 0.7)
        else:
            return random.uniform(0.1, 0.3)

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value = 0.0
        self.action = action

    def is_fully_expanded(self):
        legal_actions = self.get_legal_actions()
        return len(self.children) >= len(legal_actions)

    def get_legal_actions(self):
        # 修复2：确保LLM实例一致
        return MockLLM().generate_actions(self.state)

class MCTS:
    def __init__(self, exploration_weight=1.414):
        self.llm = MockLLM()
        self.exploration_weight = exploration_weight

    def run(self, root_state, iterations=100):
        root = Node(root_state)
        for _ in range(iterations):
            selected_node = self.select(root)
            if selected_node is None:
                continue

            expanded_node = self.expand(selected_node)
            if expanded_node is None:
                # 修复3：叶节点直接模拟自身
                value = self.simulate(selected_node)
                self.backpropagate(selected_node, value)
                continue

            value = self.simulate(expanded_node)
            self.backpropagate(expanded_node, value)
        return self.get_best_action(root)

    def select(self, node):
        while node.is_fully_expanded():
            if not node.children:
                return None
            node = self.ucb_select(node)
            if node is None:
                return None
        return node

    def ucb_select(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            if child.visit_count == 0:
                score = float('inf')
            else:
                score = (child.value / child.visit_count) + \
                        self.exploration_weight * \
                        math.sqrt(math.log(node.visit_count + 1) / (child.visit_count + 1e-6))  # 修复4：避免除零
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def expand(self, node):
        legal_actions = node.get_legal_actions()
        # 修复5：确保完全扩展所有合法动作
        for action in legal_actions:
            if action not in [c.action for c in node.children]:
                new_state = f"{node.state} -> 客服回复: {action}"
                new_node = Node(new_state, parent=node, action=action)
                node.children.append(new_node)
                return new_node
        return None

    def simulate(self, node):
        return self.llm.evaluate_value(node.action, node.state)

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value += value
            node = node.parent

    def get_best_action(self, node):
        if not node.children:
            return "无可用动作"
        # 修复6：优先选择有访问次数的节点
        # return max(node.children, key=lambda x: x.visit_count).action
        return  max(node.children, key=lambda x: x.value / (x.visit_count + 1e-6)).action

if __name__ == "__main__":
    initial_state = "用户: 我的订单已经一周了还没收到！"  # 现在能触发动作生成
    mcts = MCTS()
    best_action = mcts.run(initial_state, iterations=50)
    print(f"推荐回复: {best_action}")