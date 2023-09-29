-- USING: helpful for joining two tables on columns with the same name
SELECT
  fcst.forecast_date
, fcst.SKU
, fcst.forecast
, act.actuals
FROM
  forecast_tbl fcst
  JOIN actuals_tbl act USING (forecast_date, SKU);