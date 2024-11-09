#AmazonLeetCode

Given a string `s`, find the length of the longest substring without repeating characters.

## Solution 1: Sliding Window & Set

```python
class Solution(object):
	def lengthOfLongestSubstring(self, s):
		"""
		:type s: str
		:rtype: int
		"""
		char_set = {}
		left = max_len = 0

		for right in range(len(s)):
			while s[right] in char_set:
				char_set.remove(s[left])
				left += 1
			char_set.add(s[right])
			max_len = max(max_len, right - left + 1)
		return max_len
```


```python
left = max_len = 0
char_set = {}

for right in range(len(s)):
	
```

## Solution 2: Sliding Window and Hashing

```python
class Solution(object):
	def lengthOfLongestSubstring(self, s):
		max_len = left = 0
		count = {}
		for right, c in enumerate(s):
			count[c] = 1 + count.get(c, 0)
```


```python
class Solution(object):
	def lengthOfLongestSubstring(self, s):
		"""
		:type s: str
		:rtype: int
		"""
```