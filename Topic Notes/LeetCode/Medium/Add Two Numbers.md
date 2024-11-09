#AmazonLeetCode

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.
## Solution

```python
class Solution(object):
	def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
		dummy = ListNode()
		res = dummy
		tally = carry = 0
		while l1 or l2 or carry:
			tally = carry
			if l1:
				tally += l1.val
				l1 = l1.next
			if l2:
				tally += l2.val
				l2 = l2.next

			num = tally % 10
			carry = tally // 10
			dummy.next = ListNode(num)
			dummy = dummy.next
		return res.next		
```



```python
dummy = ListNode()
ret = dummy
total = carry = 0
while l1 or l2 or carry:
	total = carry
	if l1:
		total += l1.val
		l1 = l1.next
	if l2:
		total += l2.val
		l2 = l2.next

	tally = total % 10
	carry = total // 10
	dummy.next = ListNode(tally)
	dummy = dummy.next
return ret.next
```

