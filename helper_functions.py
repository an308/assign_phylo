"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
from Bio.Align import substitution_matrices

GAP_PENALTY = -8

class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    n = len(labels)
    nodes = [Node(label, None, 0.0, None, 0.0) for label in labels]

    if n == 1:
        return nodes[0]
    
    active = list(range(n))

    while len(active) > 2:
        m = len(active)

        # matrix for calculating current storing row sums
        D = distances[active, :][:, active]

        # calculate Q
        row_sums = np.sum(D, axis=1)
        Q = (m - 2) * D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)

        # find nodes for which Q is min
        i_pos, j_pos = np.unravel_index(np.argmin(Q), Q.shape)
        i = active[i_pos]
        j = active[j_pos]
        
        # make new new node ij & calculate branch lengths from i,j to new node
        branch_i = distances[i,j] / 2 + (row_sums[i_pos] - row_sums[j_pos]) / (2 * (m - 2))
        branch_j = distances[i,j] - branch_i

        new_node = Node(f"{len(nodes) + 1}", nodes[i], branch_i, nodes[j], branch_j)

        # calculate distance from new node to every other node (except i,j)
        remaining = [k for k in active if k != i and k != j]
        
        new_distances = {}
        for k in remaining:
            new_distances[k] = (distances[i,k] + distances[j,k] - distances[i,j]) / 2

        # add new node + add new distances to distance matrix
        nodes.append(new_node)

        prev_size = distances.shape[0]
        expanded = np.zeros((prev_size + 1, prev_size + 1))
        expanded[:prev_size, :prev_size] = distances

        for k, dist in new_distances.items():
            expanded[len(nodes) - 1, k] = dist
            expanded[k, len(nodes) - 1] = dist

        distances = expanded

        active = remaining + [len(nodes)- 1]

    # join final two nodes
    i, j = active
    final_dist = distances[i,j]
    root = Node("root", nodes[i], final_dist / 2, nodes[j], final_dist / 2)

    return root
   

def plot_nj_tree(tree: Node, ax: Axes = None, group_by_label = None, colours = None) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    y_positions = {}
    leaf_num = [0]

    def assign_y(node):
        if node.left is None and node.right is None:
            y_positions[node] = leaf_num[0]
            leaf_num[0] += 1
        else:
            assign_y(node.left)
            assign_y(node.right)
            y_positions[node] = (y_positions[node.left] + y_positions[node.right]) / 2

    def draw(node, x):
        if node.left is None and node.right is None:
            colour = "black"

            # for 3d
            if group_by_label is not None and colours is not None:
                group = group_by_label.get(node.name)
                colour = colours.get(group, "black")

            ax.text(x + 0.5, y_positions[node], node.name, va="center", color=colour)
            return
        
        left = node.left
        right = node.right

        x_left = x + node.left_distance
        x_right = x + node.right_distance

        y_left = y_positions[left]
        y_right = y_positions[right]

        ax.plot([x, x_left], [y_left, y_left], color="black")
        ax.plot([x, x_right], [y_right, y_right], color="black")
        ax.plot([x, x], [y_left, y_right], color="black")

        draw(left, x_left)
        draw(right, x_right)

    assign_y(tree)
    draw(tree, 0)

    ax.plot([-0.5, 0], [y_positions[tree], y_positions[tree]], color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """

    def count_leaves(node):
        if node is None:
            return 0
        
        if node.left is None and node.right is None:
            return 1

        left_count = count_leaves(node.left)
        right_count = count_leaves(node.right)

        return left_count + right_count

    if tree is None or (tree.left is None and tree.right is None):
        return None

    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)

    if count_leaves(tree.left) > count_leaves(tree.right):
        tree.left, tree.right = tree.right, tree.left
        tree.left_distance, tree.right_distance = tree.right_distance, tree.left_distance



def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()

def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """
    num_rows = len(seq2) + 1
    num_cols = len(seq1) + 1
    nw_matrix = np.zeros((num_rows, num_cols))

    nw_matrix[0, :] = np.arange(num_cols) * GAP_PENALTY
    nw_matrix[:, 0] = np.arange(num_rows) * GAP_PENALTY

    traceback = np.zeros((num_rows, num_cols), dtype=object)

    # populate nw_matrix with scores & traceback matrix with predecessor's coordinates

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            match = nw_matrix[i-1, j-1] + scoring_function(seq2[i-1], seq1[j-1])
            insertion = nw_matrix[i-1, j] + GAP_PENALTY
            deletion = nw_matrix[i, j-1] + GAP_PENALTY

            max_score = max(match, insertion, deletion)
            nw_matrix[i, j] = max_score
            
            if deletion == max_score:
                traceback[i, j] = (i, j-1)  
            elif insertion == max_score:
                traceback[i, j] = (i-1, j)
            elif match == max_score:
                traceback[i, j] = (i-1, j-1)   

    i = num_rows - 1
    j = num_cols - 1
    aligned_seq1 = []
    aligned_seq2 = []

    # traceback to get alignemnts

    while (i > 0 or j > 0):

        if i == 0: # reached top row, need to move left (deletion) to get to 0,0
            aligned_seq1.append(seq1[j-1])
            aligned_seq2.append("-")
            j -= 1
        elif j == 0: # reached leftmost (1st) col, need to move up (insertion) to get to 0,0
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[i-1])
            i -= 1

        else:
            ip, jp = traceback[i, j]

            if i == ip: # deletion --> move left (i didn't change)
                aligned_seq1.append(seq1[j-1])
                aligned_seq2.append("-")
                j = jp
            elif j == jp: # insertion --> move up (j didn't change)
                aligned_seq1.append("-")
                aligned_seq2.append(seq2[i-1])
                i = ip
            else: # match/mismatch --> move diagonally (both i & j changed)
                aligned_seq1.append(seq1[j-1])
                aligned_seq2.append(seq2[i-1])
                i = ip
                j = jp

    aligned_seq1.reverse()
    aligned_seq2.reverse()
    
    aligned_seq1 = "".join(aligned_seq1)
    aligned_seq2 = "".join(aligned_seq2)

    return (aligned_seq1, aligned_seq2, nw_matrix[-1, -1])

sub_matrix = substitution_matrices.load("BLOSUM62")

def scoring_function(aa_i,aa_j):
    return(sub_matrix[aa_i.upper(), aa_j.upper()])