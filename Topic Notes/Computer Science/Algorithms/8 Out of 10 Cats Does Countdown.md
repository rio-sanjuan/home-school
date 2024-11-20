```python
# Number Game from 8 out of 10 Cats Does Countdown!
# https://www.youtube.com/watch?v=bWNyBEicRjI

import operator
import itertools


def solve_countdown(numbers, target):
    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }

    for p in itertools.permutations(numbers):
        for comb in itertools.product(ops.keys(), repeat=len(numbers) - 1):
            expression = [str(p[0])]
            for i in range(1, len(numbers)):
                expression.append(comb[i - 1])
                expression.append(str(p[i]))
            expr_str = " ".join(expression)
            try:
                if eval(expr_str) == target:
                    return expr_str
            except ZeroDivisionError:
                continue
    return None


if __name__ == "__main__":
    numbers = [100, 50, 8, 8, 7, 9]
    target = 690
    solution = solve_countdown(numbers, target)

    print(f"Target: {target}")
    print(f"Numbers: {numbers}")
    if solution:
        print(f"Solution: {solution}")
    else:
        print("No solution found!")
```