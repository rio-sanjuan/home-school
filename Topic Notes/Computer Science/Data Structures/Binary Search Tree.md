A Binary Search Tree (BST) is a type of binary tree used in computer science for storing data in a way that allows for efficient searching, insertion, and deletion operations. Each node in a BST contains a key, and the keys in the left subtree of a node are smaller than the key of a node, while the keys in the right subtree are greater. This property makes BSTs useful for tasks that require sorted data or efficient searching.

## Key properties
1. **Binary Tree Structure**: Each node has at most two children (left and right)
2. **Node Order**:
	1. Left child: the key of any node's left child is less than the key of the node itself
	2. Right child: the key of any node's right child is greater than the key of the node itself
3. **No Duplicate Values**: A well-formed BST usually does not contain duplicate keys, but some variations may allow duplicates.

## Advantages
1. **Efficient Search**: Searching, insertion, and deletion can all be done in `O(log n)` time on average.
2. **Dynamic Data Structure**: Unlike arrays, BSTs allow dynamic growth and shrinkage without needing the reallocate memory or reorganize elements.
3. **In-order Traversal**: BSTs can be used to retrieve elements in sorted order by performing an in-order traversal.

## Disadvantages
1. **Can be unbalanced**: If the BST becomes unbalanced (e.g., when elements are inserted in sorted order), the tree may degrade into a linked list, with all nodes having only one child. This results in a time complexity of `O(n)` for search, insertion, and deletion. To avoid the worst-case scenario, self-balancing BSTs have been developed, such as:
	1. [[AVL Tree]]
	2. [[Red-Black Tree]]

