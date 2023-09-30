# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

# You may assume that each input would have exactly one solution, and you may not use the same element twice.

# You can return the answer in any order.


# Example 1:

# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]
# Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

# Example 2:

# Input: nums = [3,2,4], target = 6
# Output: [1,2]

# Example 3:

# Input: nums = [3,3], target = 6
# Output: [0,1]

from typing import List
import random


class Solution:
    def threeSum(self, nums: List[int], target: int) -> List[int]:
        lookup = {}
        for i, num in enumerate(nums):
            nums2 = nums.copy()
            nums2[i] = None
            temp = self.twoSum(nums2, target - num)
            if temp:
                return [i] + temp
        return False

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        lookup = {}
        for i, num in enumerate(nums):
            if num is None:
                continue
            diff = target - num
            if diff in lookup:
                return [lookup[diff], i]
            lookup[num] = i
        return False

    def test(self):
        assert self.twoSum([2, 7, 11, 15], 9) == [0, 1]
        assert self.threeSum([2, 7, 11, 15], 20) == [0, 1, 2]

        print("All test cases pass")


if __name__ == "__main__":
    sol = Solution()
    # sol.test()

    nums = random.sample(range(1, 21), 10)
    idxs = random.sample(range(10), 3)
    target = nums[idxs[0]] + nums[idxs[1]] + nums[idxs[2]]

    print(f"Target: {target}")
    print(f"Nums: {nums}")
    print(f"Output: {sol.threeSum(nums, target)}")
