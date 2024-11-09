Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice.

You can return the answer in any order.

### Example 1

**Input:** nums = [2,7,11,15], target = 9
**Output:** [0,1]
**Explanation:** Because nums[0] + nums[1] == 9, we return [0, 1].

## Solution 1: Brute Force

Brute force solution, iterate through array twice, looking for solutions (`O(n^2)`).

```python
class Solution(object):
	def twoSum(self, nums, target):
		"""
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)
        for i in range(n):
	        for j in range(i + 1, n):
		        if nums[i] + nums[j] = target:
			        return [i,j]
		return []
```

## Solution 2: Two-pass Hash Table

Iterate through the array once, and for each element check if the target minus the current element exists in the hash table.

```python
class Solution(object):
	def twoSum(self, nums, target):
		"""
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)
        numMap = {nums[i]: i for i in range(n)}
        
	    for i in range(n):
		    complement = target - nums[i]
		    if complement in numMap and numMap[complement] != i:
			    return [i, numMap[complement]]
		
		return []
```
## Solution 3: One-pass Hash Table

```python
class Solution(object):
	def twoSum(self, nums, target):
		"""
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        num_map = {}
        n = len(nums)

		for i in range(n):
			complement = target - nums[i]
			if complement in nums:
				return [i, num_map[complement]]
			num_map[nums[i]] = i

		return []
```
