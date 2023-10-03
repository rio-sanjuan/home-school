# Problem


# Example 1:

# Example 2:

# Example 3:

from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """Do not return anything, modify nums1 in-place instead"""
        p = m + n - 1
        p1 = m - 1
        p2 = n - 1

        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -= 1

        nums1[: p2 + 1] = nums2[: p2 + 1]


if __name__ == "__main__":
    sol = Solution()

    nums1 = [1, 2, 3, 0, 0, 0]
    m = 3
    nums2 = [2, 5, 6]
    n = 3

    print(nums1)
    sol.merge(nums1, m, nums2, n)
    print(nums1)
