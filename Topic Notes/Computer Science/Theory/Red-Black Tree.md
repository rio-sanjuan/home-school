A Red-Black Tree is a type of *self-balancing* [[Binary Search Tree]] that ensures the tree remains balanced by enforcing a set of properties on the tree's nodes. These properties help guarantee that the tree's height is kept in check, which in turn ensures efficient performance for operations like searching, insertion, and deletion.

Red-Black Trees are used to maintain a balanced BST, but unlike [[AVL Tree]] (another self-balancing BST), Red-Black Trees are easier to maintain during insertion and deletion because the balancing operations (like rotations) are less restrictive and can be performed more efficiently.

By enforcing these properties, Red-Black Trees can provide `O(log n)` time complexity for the operations of searching, insertion, and deletion in the worst case, which makes them highly suitable for applications that require ordered data, such as database indexing and associative containers in libraries (e.g. `std::map` in [[C++]] or `TreeMap` in [[Java]]).
## Properties
1. **Node Color**: Each node in the tree is either red or black
2. **Root Property**: The root node is always black
3. **Red Node Property**: Red nodes cannot have a red child (i.e., no two red nodes can appear consecutively on any path from the root to a leaf)
4. **Black Height Property**: Every path from a node to its descendant leaves must have the same number of black nodes. This number is called the *black height*
5. **Leaf Property**: All leaves (null nodes or `NIL` nodes, representing the absence of a child) are considered black

## Operations

### Insertion

When inserting a new node into a Red-Black Tree:
* The node is initially inserted as a *red* node.
* After insertion, the tree may violate one or more of the Red-Black properties, particularly the **Red Node Property** (i.e., two consecutive red nodes). To fix this, the tree may require a series of color flips and rotations to restore the Red-Black properties.

### Deletion

When deleting a node from a Red-Black Tree:
* After the deletion, the tree may violate the **Black Height Property** or the **Red Node Property**.
* To restore the tree's properties, a sequence of color flips and rotations is performed to maintain balance and ensure that the Red-Black Tree properties are preserved.

### Searching

Searching in a Red-Black Tree follows the same procedure as in a regular BST. You traverse the tree from the root, comparing the target key to the current node's key and moving to the left or right subtree accordingly. The search operation has a time complexity of `O(log n)`.

## Balancing

Red-Black Trees ensure that the height of the tree remains within a logarithmic bound. Specifically, the height of a Red-Black tree is at most `2 * log(n + 1)` where `n` is the number of nodes in the tree. This guarantees that the tree does not degenerate into a linear structure, as can happen in an unbalanced BST.

When inserting or deleting a node, balancing a tree may involve *rotations* and *color flips*. The two primary types of rotations used are:
1. **Left Rotation**: A rotation that moves a right child up to the parent position and makes the parent the left child of the rotated node.
2. **Right Rotation**: The opposite of a left rotation, where a left child is moved up to the parent position.

## Implementation

```python
class Node:
    def __init__(self, key, color='red'):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None


class RedBlackTree:
    def __init__(self):
        self.TNULL = Node(key=None, color='black')  # Sentinel leaf node
        self.root = self.TNULL

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert_fix(self, k):
        while k.parent.color == 'red':
            if k.parent == k.parent.parent.left:
                u = k.parent.parent.right  # Uncle
                if u.color == 'red':
                    # Case 1: Uncle is red
                    u.color = 'black'
                    k.parent.color = 'black'
                    k.parent.parent.color = 'red'
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # Case 2: k is right child
                        k = k.parent
                        self.left_rotate(k)
                    # Case 3: k is left child
                    k.parent.color = 'black'
                    k.parent.parent.color = 'red'
                    self.right_rotate(k.parent.parent)
            else:
                # Symmetric to the "if" block above
                u = k.parent.parent.left
                if u.color == 'red':
                    u.color = 'black'
                    k.parent.color = 'black'
                    k.parent.parent.color = 'red'
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 'black'
                    k.parent.parent.color = 'red'
                    self.left_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 'black'

    def insert(self, key):
        node = Node(key)
        node.left = self.TNULL
        node.right = self.TNULL
        node.parent = None

        y = None
        x = self.root

        # Find the correct position for the new node
        while x != self.TNULL:
            y = x
            if node.key < x.key:
                x = x.left
            else:
                x = x.right

        node.parent = y
        if y == None:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node

        # Fix the tree after insertion
        self.insert_fix(node)

    def transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def delete_fix(self, x):
        while x != self.root and x.color == 'black':
            if x == x.parent.left:
                w = x.parent.right
                if w.color == 'red':
                    w.color = 'black'
                    x.parent.color = 'red'
                    self.left_rotate(x.parent)
                    w = x.parent.right

                if w.left.color == 'black' and w.right.color == 'black':
                    w.color = 'red'
                    x = x.parent
                else:
                    if w.right.color == 'black':
                        w.left.color = 'black'
                        w.color = 'red'
                        self.right_rotate(w)
                        w = x.parent.right

                    w.color = x.parent.color
                    x.parent.color = 'black'
                    w.right.color = 'black'
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                # Symmetric to the "if" block above
                w = x.parent.left
                if w.color == 'red':
                    w.color = 'black'
                    x.parent.color = 'red'
                    self.right_rotate(x.parent)
                    w = x.parent.left

                if w.right.color == 'black' and w.left.color == 'black':
                    w.color = 'red'
                    x = x.parent
                else:
                    if w.left.color == 'black':
                        w.right.color = 'black'
                        w.color = 'red'
                        self.left_rotate(w)
                        w = x.parent.left

                    w.color = x.parent.color
                    x.parent.color = 'black'
                    w.left.color = 'black'
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = 'black'

    def delete(self, key):
        z = self.root
        while z != self.TNULL:
            if z.key == key:
                break
            elif key < z.key:
                z = z.left
            else:
                z = z.right

        if z == self.TNULL:
            print("Key not found in the tree")
            return

        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.transplant(z, z.right)
        elif z.right == self.TNULL:
            x = z.left
            self.transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self.transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        if y_original_color == 'black':
            self.delete_fix(x)

    def minimum(self, node):
        while node.left != self.TNULL:
            node = node.left
        return node

    def print_tree(self):
        self.print_helper(self.root, "", True)

    def print_helper(self, node, indent, last):
        if node != self.TNULL:
            print(indent, end="")
            if last:
                print("R----", end="")
                indent += "     "
            else:
                print("L----", end="")
                indent += "|    "

            print(f"{node.key}({node.color})")
            self.print_helper(node.left, indent, False)
            self.print_helper(node.right, indent, True)


# Example usage
if __name__ == "__main__":
    rbt = RedBlackTree()
    keys = [20, 15, 25, 30, 10, 5, 1, 7, 12]

    for key in keys:
        rbt.insert(key)

    print("Red-Black Tree:")
    rbt.print_tree()

    print("\nAfter deleting 15:")
    rbt.delete(15)
    rbt.print_tree()
```


