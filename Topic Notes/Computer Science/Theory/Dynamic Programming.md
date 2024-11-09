Dynamic Programming (DP) is a programming paradigm that can systematically and efficiently explore all possible solutions to a problem. It is capable of solving a wide variety of problems that often have the following characteristics:

1. The problem can be broken down into "overlapping subproblems" - smaller versions of the original problem that are re-used multiple times.
2. The problem has an "optimal substructure" - an optimal solution can be formed from optimal solutions to the overlapping subproblems of the original problem.

A classic example used to explain DP is the Fibonacci sequence. If you want to find the `nth` Fibonacci number `F(n)`, you can break it down into smaller **subproblems** - find `F(n-1)` and `F(n-2)` instead. Then, adding the solutions to these subproblems together gives the answer to the original question, `F(n) = F(n-1) + F(n-2)`, which means the problem has **optimal substructure**, since a solution `F(n)` to to the original problem can be formed from the solutions to the subproblems. These subproblems are **overlapping**.

Greedy problems have optimal substructure, but not overlapping subproblems. Dive and conquer algorithms break a problem into subproblems, but these subproblems are not **overlapping** (which is why DP and divide and conquer are commonly mistaken for one another).

## Top-down and Bottom-up

There are two ways to implement a DP algorithm:

1. Bottom-up, also known as *tabulation*
2. Top-down, also known as *memoization*

### Bottom-up (Tabulation)



### Top-down (Memoization)