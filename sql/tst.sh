
bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item_reordered_30" --flatten_results --replace "
SELECT
  user_id,
  product_id,
  CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
  CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
  CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
FROM
(
SELECT
  a.user_id as user_id,
  a.product_id as product_id,
  LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
  b.last_buy as last_buy
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  b.last_buy <= 30 and reordered = 1
) as s
GROUP BY
  user_id, product_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item_reordered" --flatten_results --replace "
SELECT
  user_id,
  product_id,
  CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
  CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
  CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
FROM
(
SELECT
  a.user_id as user_id,
  a.product_id as product_id,
  LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
  b.last_buy as last_buy
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  reordered = 1
) as s
GROUP BY
  user_id, product_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_aisle_reordered_30" --flatten_results --replace "
SELECT
  user_id,
  aisle_id,
  CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
  CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
  CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
FROM
(
SELECT
  a.user_id as user_id,
  a.aisle_id as aisle_id,
  LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.aisle_id ORDER BY a.order_number) as diffs,
  b.last_buy as last_buy
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  b.last_buy <= 30 and reordered = 1
) as s
GROUP BY
  user_id, aisle_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_aisle_reordered" --flatten_results --replace "
SELECT
  user_id,
  aisle_id,
  CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
  CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
  CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
FROM
(
SELECT
  a.user_id as user_id,
  a.aisle_id as aisle_id,
  LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.aisle_id ORDER BY a.order_number) as diffs,
  b.last_buy as last_buy
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  reordered = 1
) as s
GROUP BY
  user_id, aisle_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_depart_reordered_30" --flatten_results --replace "
SELECT
  user_id,
  department_id,
  CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
  CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
  CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
FROM
(
SELECT
  a.user_id as user_id,
  a.department_id as department_id,
  LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.department_id ORDER BY a.order_number) as diffs,
  b.last_buy as last_buy
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  b.last_buy <= 30 and reordered = 1
) as s
GROUP BY
  user_id, department_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_depart_reordered" --flatten_results --replace "
SELECT
  user_id,
  department_id,
  CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
  CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
  CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
FROM
(
SELECT
  a.user_id as user_id,
  a.department_id as department_id,
  LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.department_id ORDER BY a.order_number) as diffs,
  b.last_buy as last_buy
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  reordered = 1
) as s
GROUP BY
  user_id, department_id
"

bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_reordered_30" --flatten_results --replace "
SELECT
  user_id,
  CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
  CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
  CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
FROM
(
SELECT
  a.user_id as user_id,
  a.product_id as product_id,
  LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
  b.last_buy as last_buy
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  b.last_buy <= 30 and reordered = 1
) as s
GROUP BY
  user_id
"

###
