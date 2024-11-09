Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice.

You can return the answer in any order.

### Example 1

**Input:** nums = [2,7,11,15], target = 9
**Output:** [0,1]
**Explanation:** Because nums[0] + nums[1] == 9, we return [0, 1].

## Solution

```python
class Solution(object):
	# Brute force approach
	def twoSum1(self, nums, target):
		"""
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
	    n = len(nums)
		for i in range(n - 1):		
			for j in range(i + 1, n):
				if nums[i] + nums[j] == target:
					return [i, j]
		return []

	def twoSum2(self, nums: List[int], target: int) -> List[int]:
		"""
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        numMap = {}
        n = len(nums)
        
        for i in range(n):
	        numMap[nums[i]] = i
        

```

