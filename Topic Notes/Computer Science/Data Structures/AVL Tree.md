An AVL tree (Adelson-Velsky and Landis tree) is a type of self-balancing [[Binary Search Tree]]. It was the first data structure to maintain a balance condition for a binary search tree, ensuring that the height of the tree is kept logarithmic in relation to the number of nodes. This provides efficient operations for searching, insertion, and deletion.

## Properties
1. **Binary Search Tree Property**: Like any binary tree, for each node in the AVL tree:
	1. The key of the left child node is smaller than the key of the parent node
	2. The key of the right child node is larger than the key of the parent node
2. **Balance Factor**: An AVL tree maintains a balance factor for every node, which is defined as the difference between the heights of the left subtree and the right subtree: `Balance Factor = Height of Left Subtree - Height of Right Subtree`. For each node, the balance factor must in the range of -1 to 1. If the balance factor goes outside this range, the tree is unbalanced, and it needs to be rebalanced through rotations. 
	1. `Balance Factor = 0`: The left and right subtrees are of equal height
	2. `Balance Factor = +1`: The left subtree is one level taller than the right
	3. `Balance Factor = -1`: The right subtree is one level taller than the left
3. **Height of the AVL Tree**: The height of an AVL tree is kept logarithmic with respect to the number of nodes. Specifically, the height of an AVL tree with `n` nodes is `O(log n)`, which ensures that the time complexity for search, insertion, and deletion remains `O(log n)`.

## Operations on an AVL Tree:

### Insertion
- Insertions follow the same process as in a regular binary search tree (BST).
- After insertion, the **balance factor** of each ancestor node is checked. If any node has a balance factor outside the range of -1 to 1, a rotation is required to restore balance.

There are **4 types of rotations** that can be used to rebalance the tree:
- **Left Rotation (Single Rotation)**: Used when the right subtree is taller than the left (right-heavy).
- **Right Rotation (Single Rotation)**: Used when the left subtree is taller than the right (left-heavy).
- **Left-Right Rotation (Double Rotation)**: A combination of a left rotation followed by a right rotation. This is used when the left child of the right subtree is taller (right-left imbalance).
- **Right-Left Rotation (Double Rotation)**: A combination of a right rotation followed by a left rotation. This is used when the right child of the left subtree is taller (left-right imbalance).
### Deletion
- Deletion involves removing a node, followed by checking the balance factors of each ancestor node.
- If any ancestor node has a balance factor outside the acceptable range (-1 to +1), a rotation is performed to restore balance.
### Searching
- Searching in an AVL tree is the same as in a regular binary search tree. Since the height is logarithmic, search operations are efficient with a time complexity of **O(log n)**.

## Implementation

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # Height of the node (initially 1 for new node)

class AVLTree:
    def __init__(self):
        self.root = None

    # Utility function to get the height of a node
    def height(self, node):
        if not node:
            return 0
        return node.height

    # Utility function to get the balance factor of a node
    def balance_factor(self, node):
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)

    # Right rotation (Single Rotation)
    def right_rotate(self, y):
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        y.height = max(self.height(y.left), self.height(y.right)) + 1
        x.height = max(self.height(x.left), self.height(x.right)) + 1
        
        # Return new root
        return x

    # Left rotation (Single Rotation)
    def left_rotate(self, x):
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        x.height = max(self.height(x.left), self.height(x.right)) + 1
        y.height = max(self.height(y.left), self.height(y.right)) + 1
        
        # Return new root
        return y

    # Insert a node
    def insert(self, node, key):
        # 1. Perform the normal BST insert
        if not node:
            return Node(key)

        if key < node.key:
            node.left = self.insert(node.left, key)
        else:
            node.right = self.insert(node.right, key)

        # 2. Update the height of the ancestor node
        node.height = max(self.height(node.left), self.height(node.right)) + 1

        # 3. Get the balance factor of this ancestor node to check whether it became unbalanced
        balance = self.balance_factor(node)

        # If the node becomes unbalanced, then there are 4 cases

        # Left-Left Case (Right Rotation)
        if balance > 1 and key < node.left.key:
            return self.right_rotate(node)

        # Right-Right Case (Left Rotation)
        if balance < -1 and key > node.right.key:
            return self.left_rotate(node)

        # Left-Right Case (Left Rotation followed by Right Rotation)
        if balance > 1 and key > node.left.key:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # Right-Left Case (Right Rotation followed by Left Rotation)
        if balance < -1 and key < node.right.key:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        # Return the (unchanged) node pointer
        return node

    # Function to perform the insertion of a key in the AVL tree
    def insert_key(self, key):
        self.root = self.insert(self.root, key)

    # Utility function to find the node with the minimum key value
    def min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    # Delete a node
    def delete(self, node, key):
        # STEP 1: Perform the normal BST delete
        if not node:
            return node

        if key < node.key:
            node.left = self.delete(node.left, key)
        elif key > node.key:
            node.right = self.delete(node.right, key)
        else:
            # Node to be deleted found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # Node with two children, get the inorder successor (smallest in the right subtree)
                temp = self.min_value_node(node.right)
                node.key = temp.key
                node.right = self.delete(node.right, temp.key)

        # STEP 2: Update height of the current node
        if not node:
            return node

        node.height = max(self.height(node.left), self.height(node.right)) + 1

        # STEP 3: Get the balance factor of this node to check whether it became unbalanced
        balance = self.balance_factor(node)

        # If the node becomes unbalanced, then there are 4 cases

        # Left-Left Case (Right Rotation)
        if balance > 1 and self.balance_factor(node.left) >= 0:
            return self.right_rotate(node)

        # Right-Right Case (Left Rotation)
        if balance < -1 and self.balance_factor(node.right) <= 0:
            return self.left_rotate(node)

        # Left-Right Case (Left Rotation followed by Right Rotation)
        if balance > 1 and self.balance_factor(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # Right-Left Case (Right Rotation followed by Left Rotation)
        if balance < -1 and self.balance_factor(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    # Function to delete a key in the AVL tree
    def delete_key(self, key):
        self.root = self.delete(self.root, key)

    # Function to perform an inorder traversal of the tree
    def inorder(self, node):
        if not node:
            return []
        return self.inorder(node.left) + [node.key] + self.inorder(node.right)

    # Function to search a node
    def search(self, node, key):
        # Base case: root is null or key is present at the root
        if not node or node.key == key:
            return node

        # Key is greater than root's key
        if key > node.key:
            return self.search(node.right, key)

        # Key is smaller than root's key
        return self.search(node.left, key)

    # Function to print the tree in order
    def print_tree(self):
        print("In-order Traversal:", self.inorder(self.root))

# Example usage:
if __name__ == "__main__":
    tree = AVLTree()
    
    # Insert elements into the AVL tree
    elements = [20, 15, 25, 10, 5, 1, 7, 30]
    for element in elements:
        tree.insert_key(element)

    # Print in-order traversal of the AVL tree
    tree.print_tree()  # Should be sorted: [1, 5, 7, 10, 15, 20, 25, 30]

    # Delete a node
    tree.delete_key(15)

    # Print in-order traversal after deletion
    tree.print_tree()  # Should show the tree after deletion
```