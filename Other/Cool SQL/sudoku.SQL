-- command solves sudoku puzzles using recursive subquery factoring!
-- https://technology.amis.nl/it/solving-a-sudoku-with-one-sql-statement/
SET
    sudoku_str = '53  7    6  195    98    6 8   6   34  8 3  17   2   6 6    28    419  5    8  79';

WITH
    x (s, ind) AS (
        SELECT
            sud
          , instr (sud, ' ')
        FROM
            (
                SELECT
                    $sudoku_str sud
                FROM
                    dual
            )
        UNION ALL
        SELECT
            SUBSTR(s, 1, ind - 1) || z || SUBSTR(s, ind + 1)
          , instr (s, ' ', ind + 1)
        FROM
            x
          , (
                SELECT
                    TO_CHAR(rownum) z
                FROM
                    dual CONNECT BY rownum <= 9
            ) z
        WHERE
            ind > 0
            AND NOT EXISTS (
                SELECT
                    NULL
                FROM
                    (
                        SELECT
                            rownum lp
                        FROM
                            dual CONNECT BY rownum <= 9
                    )
                WHERE
                    z = SUBSTR(s, TRUNC((ind - 1) / 9) * 9 + lp, 1)
                    OR z = SUBSTR(s, MOD(ind - 1, 9) - 8 + lp * 9, 1)
                    OR z = SUBSTR(
                        s
                      , MOD(TRUNC((ind - 1) / 3), 3) * 3 + TRUNC((ind - 1) / 27) * 27 + lp + TRUNC((lp - 1) / 3) * 6
                      , 1
                    )
            )
    )
SELECT
    s
FROM
    x
WHERE
    ind = 0