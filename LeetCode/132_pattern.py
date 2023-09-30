# Given an array of n integers nums, a 132 pattern is a subsequence of three integers nums[i], nums[j] and nums[k] such that i < j < k and nums[i] < nums[k] < nums[j].
# Return true if there is a 132 pattern in nums, otherwise, return false.

# Example 1:

# Input: nums = [1,2,3,4]
# Output: false
# Explanation: There is no 132 pattern in the sequence.

# Example 2:

# Input: nums = [3,1,4,2]
# Output: true
# Explanation: There is a 132 pattern in the sequence: [1, 4, 2].

# Example 3:

# Input: nums = [-1,3,2,0]
# Output: true
# Explanation: There are three 132 patterns in the sequence: [-1, 3, 2], [-1, 3, 0] and [-1, 2, 0].

from typing import List


class Solution:
    def has132pattern(self, nums: List[int]) -> bool:
        third = float("-inf")
        stack = []
        for num in reversed(nums):
            if num < third:
                return True
            while stack and stack[-1] < num:
                third = stack.pop()
            stack.append(num)
        return False

    # TODO: this function seems a bit buggy?
    # find132pattern([1,2,5,4,3,6]) returns [1,2,4] as a match?
    def find132pattern(self, nums: List[int]) -> List[List[int]]:
        third = float("-inf")
        stack = []
        patterns = []
        for num in reversed(nums):
            if num < third:
                patterns.append([num, stack[-1], third])
            while stack and stack[-1] < num:
                third = stack.pop()
            stack.append(num)
        return patterns

    def test(self):
        assert self.has132pattern([3, 1, 4, 2])  # 132 pattern exists: 1, 4, 2.
        assert not self.has132pattern([1, 2, 3, 4])  # No 132 pattern exists.
        assert not self.has132pattern([1])  # Single element, no 132 pattern.

        print("All test cases pass")


if __name__ == "__main__":
    sol = Solution()
    sol.test()

    nums = [1, 2, 5, 4, 3, 6]
    print(f"Input: {nums}")
    print(f"132 Patterns: {sol.find132pattern(nums)}")

    nums = [3, 1, 4, 2]
    print(f"Input: {nums}")
    print(f"132 Patterns: {sol.find132pattern(nums)}")

    nums = [-1, 3, 2, 0]
    print(f"Input: {nums}")
    print(f"132 Patterns: {sol.find132pattern(nums)}")
