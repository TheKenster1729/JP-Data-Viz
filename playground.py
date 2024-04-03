from sklearn.tree import DecisionTreeClassifier
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Assuming `X` and `y` are your features and target variable respectively
data = np.random.rand(100, 10)
X = pd.DataFrame(data, columns = ['x'+str(i) for i in range(1, 11)])
y = np.random.randint(0, 2, size = 100)

# Train a simple CART model
clf = DecisionTreeClassifier(max_depth=3)  # Limit depth for simplicity
clf.fit(X, y)

# Access the tree properties
tree_ = clf.tree_
features = [f"{i}" if i == -2 else clf.feature_names_in_[i] for i in tree_.feature]

class TreeNode:
    def __init__(self, id, feature=None, threshold=None, left=None, right=None, value=None):
        self.id = id
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.x = 0  # x-coordinate in the plot
        self.y = 0  # y-coordinate in the plot

def layout_binary_tree(root, x=0, y=0, level_height=1, node_spacing=2):
    if root is None:
        return 0
    
    left_width = layout_binary_tree(root.left, x, y-1, level_height, node_spacing) if root.left else 0
    root.x = x + left_width
    root.y = y
    right_width = layout_binary_tree(root.right, root.x + node_spacing, y-1, level_height, node_spacing) if root.right else 0
    
    return left_width + node_spacing + right_width

def build_tree_from_CART(tree_, node_id=0, depth=0):
    if tree_.children_left[node_id] == tree_.children_right[node_id]:  # Leaf node
        value = tree_.value[node_id]
        return TreeNode(node_id, value=value, feature="leaf", threshold=0, left=None, right=None)
    left_child = build_tree_from_CART(tree_, tree_.children_left[node_id], depth + 1)
    right_child = build_tree_from_CART(tree_, tree_.children_right[node_id], depth + 1)
    feature = features[node_id]
    threshold = tree_.threshold[node_id]
    return TreeNode(node_id, feature=feature, threshold=threshold, left=left_child, right=right_child)

def add_annotations(fig, node):
    if node is not None:
        # Add annotation with feature and threshold or leaf value
        if node.feature == "leaf":
            text = f"Leaf\nSamples: {np.sum(node.value)}"
        else:
            text = f"{node.feature}\n< {node.threshold:.2f}"
        fig.add_annotation(x=node.x, y=node.y, text=text, showarrow=False, font=dict(color="blue", size=12))
        if node.left:
            add_annotations(fig, node.left)
        if node.right:
            add_annotations(fig, node.right)

def draw_tree_with_data(root):
    node_x, node_y, edge_x, edge_y = [], [], [], []
    
    def traverse(node):
        if node:
            node_x.append(node.x)
            node_y.append(node.y)
            if node.left:
                edge_x.extend([node.x, node.left.x, None])  # None to stop drawing the line
                edge_y.extend([node.y, node.left.y, None])
                traverse(node.left)
            if node.right:
                edge_x.extend([node.x, node.right.x, None])
                edge_y.extend([node.y, node.right.y, None])
                traverse(node.right)
                
    traverse(root)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', name='Edges'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', name='Nodes'))
    
    # Add annotations for each node
    add_annotations(fig, root)
    
    fig.update_layout(showlegend=False)
    fig.show()

# Build the binary tree from the sklearn CART model
root = build_tree_from_CART(clf.tree_)

# Layout and visualize the binary tree with sklearn CART data
layout_binary_tree(root)
draw_tree_with_data(root)
