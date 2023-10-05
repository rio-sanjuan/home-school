-- COALESCE: takes a list of columns (or other info) and returns the first non-`NULL` one
-- i.e. if you have an email address return that, otherwise return the phone-number
SELECT
  id
, name
, COALESCE(email, phone) AS contact_method
FROM
  clients