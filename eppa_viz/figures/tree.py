"""Dashboard figure builders (plotly)."""

import json
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors

from analysis import (
    FilteredInputOutputMapping,
    FilteredOutputOutputMapping,
    InputOutputMapping,
    OutputOutputMapping,
    TimeSeriesClustering,
)
from eppa_viz.figures.base import DashboardFigure, OUTPUT_TIMESERIES
from eppa_viz.figures.utils import sanitize_uid, TraceInfo
from sql_utils import DataRetrieval, SQLConnection
from styling import Color, Options, Readability



def _scenario_subplot_title(scenario):
    opts = Options()
    if scenario in opts.publication_scenario_display_names:
        return opts.publication_scenario_display_names[scenario]
    return opts.scenario_display_names[scenario]

class TreeNode:
    def __init__(self, id, feature=None, threshold=None, left=None, right=None, value=None, density = None, coverage = None):
        self.id = id
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.x = 0  # x-coordinate in the plot
        self.y = 0  # y-coordinate in the plot
        self.density = density
        self.coverage = coverage

class PlotTree(DashboardFigure):
    def __init__(self, fit_model, y=None):
        super().__init__("cart-tree-diagram")
        self.fit_model = fit_model
        self.fit_tree_model = self.fit_model.tree_
        self.y = y
        # Class 1 is the interest label from preprocess_for_classification / in_constraint_range.
        # Metrics are derived from tree_.value so they don't depend on whatever y the caller passed
        # (some call sites pass continuous values, which broke sum(y)-based coverage).
        self.interest_idx = int(np.where(self.fit_model.classes_ == 1)[0][0])
        self.non_interest_idx = 1 - self.interest_idx if len(self.fit_model.classes_) == 2 else 0
        self.total_interest = self.fit_tree_model.value[0][0][self.interest_idx]

    def layout_binary_tree(self, root, depth=0, x=0, y=0, level_height=10, node_spacing=5):
        """
        Layout binary tree with dynamic spacing based on depth to avoid label overlap.
        :param root: TreeNode, the root of the binary tree.
        :param depth: int, current depth of the node (root is 0).
        :param x: float, x-coordinate of the current node.
        :param y: float, y-coordinate of the current node.
        :param level_height: float, the vertical spacing between levels of the tree.
        :param base_spacing: float, the base horizontal spacing between nodes.
        :param depth_factor: float, the factor by which the base_spacing is increased at each depth level.
        :return: float, the total width of the subtree rooted at the current node.
        """
        if root is None:
            return 0
        
        left_width = self.layout_binary_tree(root.left, depth + 1, x, y - level_height, level_height, node_spacing) if root.left else 0
        root.x = x + left_width
        root.y = y
        right_width = self.layout_binary_tree(root.right, depth + 1, root.x + node_spacing, y - level_height, level_height, node_spacing) if root.right else 0
        
        return left_width + node_spacing + right_width

    def build_tree_from_CART(self, tree_, node_id=0, depth=0):
        class_counts = tree_.value[node_id][0]
        node_interest = class_counts[self.interest_idx]
        node_non_interest = class_counts[self.non_interest_idx] if len(class_counts) > 1 else 0
        node_total = class_counts.sum()

        # density: interest cases in this node / all cases in this node
        density = node_interest / node_total if node_total else 0.0
        # coverage: interest cases in this node / all interest cases in the tree
        coverage = node_interest / self.total_interest if self.total_interest else 0.0

        # Always store as [[non-interest, interest]] for hover text
        value = np.array([[node_non_interest, node_interest]])

        if tree_.children_left[node_id] == tree_.children_right[node_id]:  # Leaf node
            return TreeNode(node_id, value=value, feature="leaf", threshold=0, left=None, right=None, density = density, coverage = coverage)
        left_child = self.build_tree_from_CART(tree_, tree_.children_left[node_id], depth + 1)
        right_child = self.build_tree_from_CART(tree_, tree_.children_right[node_id], depth + 1)
        feature_id = self.fit_tree_model.feature[node_id]
        feature = self.fit_model.feature_names_in_[feature_id]
        threshold = tree_.threshold[node_id]
        return TreeNode(node_id, feature=feature, threshold=threshold, left=left_child, right=right_child, value = value, density = density, coverage = coverage)

    def add_annotations(self, fig, node):
        if node is not None:
            # Add annotation with feature and threshold or leaf value
            if not node.feature == "leaf":
                text = "{}<br><={:.2f}</br>".format(node.feature, node.threshold)
            else:
                text = "Leaf"
            # value is stored as [[non-interest, interest]]
            hover_text = "Samples: {}<br>Non-interest Cases: {}, Interest Cases: {}<br>Density: {}%, Coverage: {}%".format(int(node.value[0][0] + node.value[0][1]), int(node.value[0][0]), int(node.value[0][1]), int(node.density*100), int(node.coverage*100))
            fig.add_annotation(x=node.x, y=node.y, text=text, showarrow=False, font=dict(size=10), hovertext = hover_text)
            if node.left:
                self.add_annotations(fig, node.left)
            if node.right:
                self.add_annotations(fig, node.right)

    def draw_tree_with_data(self, root):
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
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', hoverinfo = "skip"))
        
        # Replace markers scatterplot with circle shapes
        radius = 4
        for x, y in zip(node_x, node_y):
            fig.add_shape(type="circle",
                        x0=x - radius, y0=y - radius,
                        x1=x + radius, y1=y + radius,
                        # xref = "paper", yref = "paper",
                        line_color="#a185ff",
                        fillcolor="#a185ff")

        # Add annotations for each node
        self.add_annotations(fig, root)
        
        fig.update_layout(showlegend=False)
        fig.update_layout(
            xaxis=dict(
                constrain='domain',  # This could also help in some cases
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            plot_bgcolor="white",  # Makes grid more visible
        )
        fig.update_yaxes(showticklabels = False)
        fig.update_xaxes(showticklabels = False)

        return fig

    def get_params(self):
        return self.fit_tree_model.node_count, self.fit_tree_model.children_left, self.fit_tree_model.children_right, self.fit_tree_model.feature, self.fit_tree_model.threshold, self.fit_tree_model.value

    def create_tree_information(self):
        # adopted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
        self.n_nodes, self.children_left, self.children_right, self.feature, self.threshold, self.values = self.get_params()
        self.node_depth = np.zeros(shape = self.n_nodes, dtype = np.int64)
        self.is_leaves = np.zeros(shape = self.n_nodes, dtype = bool)
        stack = [(0, 0)]
        while len(stack) > 0:
            node_id, depth = stack.pop()
            self.node_depth[node_id] = depth

            is_split_node = self.children_left[node_id] != self.children_right[node_id]
            if is_split_node:
                stack.append((self.children_left[node_id], depth + 1))
                stack.append((self.children_right[node_id], depth + 1))
            else:
                self.is_leaves[node_id] = True

    def draw_child_nodes(self, fig, id, spacing = 2, radius = 1):
        node_depth = self.node_depth[id]
        left_child_x = self.pos_x[id] - 2**(self.max_depth - node_depth)
        right_child_x = self.pos_x[id] + 2**(self.max_depth - node_depth)

        # edges
        # fig.add_trace(go.Scatter(x = [self.pos_x[id], center_x], y = [self.node_depth[id], center_y],
        #                          mode = "lines"))
        # left child
        fig.add_trace(go.Scatter(x = [left_child_x], y = [-node_depth], 
                                mode = 'markers',
                                name = id,
                                marker = dict(symbol = "circle",
                                            color = '#6175c1'),
                                opacity = 0)
                            )
        fig.add_shape(type = "circle", x0 = left_child_x - radius, y0 = -node_depth - radius, x1 = left_child_x + radius, y1 = -node_depth + radius)

        # right child
        fig.add_trace(go.Scatter(x = [right_child_x], y = [-node_depth], 
                                mode = 'markers',
                                name = id,
                                marker = dict(symbol = "circle",
                                            color = '#6175c1'),
                                opacity = 0)
                            )
        fig.add_shape(type = "circle", x0 = right_child_x - radius, y0 = -node_depth - radius, x1 = right_child_x + radius, y1 = -node_depth + radius)
        # fig.add_annotation(ax = center_x, axref = 'x', ay = center_y, ayref = 'y', x = 1, arrowcolor = 'red', xref = 'x', y = 1, yref='y', arrowwidth = 2.5, arrowside = 'end', arrowsize = 1, arrowhead = 4)

    def make_plot(self, show = False):
        # Build the binary tree from the sklearn CART model
        root = self.build_tree_from_CART(self.fit_tree_model)

        # Layout and visualize the binary tree with sklearn CART data
        self.layout_binary_tree(root)
        fig = self.draw_tree_with_data(root)

        if show:
            fig.show()

        return fig

        # self.create_tree_information()
        # # find max number of nodes
        
        # fig = go.Figure()
        # feature_names = self.fit_model.feature_names_in_
        # self.max_depth = max(self.node_depth)
        # print(self.max_depth)
        # self.pos_x = {}

        # for i in range(self.n_nodes):
        #     left_child = self.children_left[i]
        #     right_child = self.children_right[i]
        #     depth = self.node_depth[i]
        #     self.pos_x[left_child] = -2**(self.max_depth - 1)
        #     self.pos_x[right_child] = 2**(self.max_depth - 1)

        #     if i == 0:
        #         fig.add_trace(go.Scatter(x = [0], y = [0], 
        #                                 mode = 'markers',
        #                                 marker = dict(symbol = "circle",
        #                                             color = '#6175c1'))
        #                             )
        #         # left child
        #         fig.add_trace(go.Scatter(x = [-2**(self.max_depth - 1)], y = [-1], 
        #                                 mode = 'markers',
        #                                 marker = dict(symbol = "circle",
        #                                             color = '#6175c1'))
        #                             )
        #         # right child
        #         fig.add_trace(go.Scatter(x = [2**(self.max_depth - 1)], y = [-1], 
        #                                 mode = 'markers',
        #                                 marker = dict(symbol = "circle",
        #                                             color = '#6175c1'))
        #                             )
        #     else:
        #         if self.is_leaves[i]:
        #             # Leaf node
        #             continue
        #         else:
        #             # Decision node
        #             self.draw_child_nodes(fig, i)


        # # Customize layout
        # fig.update_layout(title = 'Decision Tree Visualization',
        #                 showlegend = False)
        # fig.update_yaxes(
        #     scaleanchor="x",
        #     scaleratio=1,
        # )
        # fig.show()
        # return fig

