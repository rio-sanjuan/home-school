You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.


## My Solution

```python
class Solution(object):
	def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
		n1 = len(l1)
		n2 = len(l2)
		num1 = 0
		num2 = 0
		carry = 0
		i = 0
		out = []
		while i < max(n1, n2):
			if i < n1:
				num1 = l1[i]
			if i < n2:
				num2 = l2[i]
			tally = num1 + num2
			carry = tally % 10
			out.append(tally)
			

			i = i + 1
		
```