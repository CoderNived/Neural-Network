# engine/autograd_clone.py

"""
Autograd engine for Minigrad.

Responsible for:
- Graph construction
- Topological sorting
- Backpropagation execution
- Gradient reset

Value/Tensor objects register:
    _parents
    _backward

Autograd handles traversal only.
"""

import numpy as np


class AutogradEngine:
    """
    Generic reverse-mode automatic differentiation engine.
    """

    @staticmethod
    def build_topology(root):

        topo = []
        visited = set()

        def dfs(node):

            if id(node) in visited:
                return

            visited.add(id(node))

            for parent in node._parents:
                dfs(parent)

            topo.append(node)

        dfs(root)

        return topo

    @staticmethod
    def backward(root):

        topo = AutogradEngine.build_topology(root)

        root.grad = np.ones_like(root.data)

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def zero_grad(root):

        visited = set()

        def dfs(node):

            if id(node) in visited:
                return

            visited.add(id(node))

            node.grad = np.zeros_like(node.data)

            for parent in node._parents:
                dfs(parent)

        dfs(root)