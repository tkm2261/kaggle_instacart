
###
bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item_30" --flatten_results --replace "
SELECT
  user_id,
  product_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
  b.last_buy <= 30
) as s
GROUP BY
  user_id, product_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item" --flatten_results --replace "
SELECT
  user_id,
  product_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
) as s
GROUP BY
  user_id, product_id
"
###

###
bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_aisle_30" --flatten_results --replace "
SELECT
  user_id,
  aisle_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
  b.last_buy <= 30
) as s
GROUP BY
  user_id, aisle_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_aisle" --flatten_results --replace "
SELECT
  user_id,
  aisle_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(51, QUANTILES(diffs - last_buy, 101)) is not NULL THEN NTH(51, QUANTILES(diffs - last_buy, 101)) ELSE -1 END as med_diffs,
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
) as s
GROUP BY
  user_id, aisle_id
"
###

###
bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_depart_30" --flatten_results --replace "
SELECT
  user_id,
  department_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
  b.last_buy <= 30
) as s
GROUP BY
  user_id, department_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_depart" --flatten_results --replace "
SELECT
  user_id,
  department_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
) as s
GROUP BY
  user_id, department_id
"
###

bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_30" --flatten_results --replace "
SELECT
  user_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
  b.last_buy <= 30
) as s
GROUP BY
  user_id
"

bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user" --flatten_results --replace "
SELECT
  user_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
) as s
GROUP BY
  user_id
"

###
bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_item_30" --flatten_results --replace "
SELECT
  product_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
  b.last_buy <= 30
) as s
GROUP BY
  product_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_item" --flatten_results --replace "
SELECT
  product_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
) as s
GROUP BY
  product_id
"
######

bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_aisle_30" --flatten_results --replace "
SELECT
  aisle_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
  b.last_buy <= 30
) as s
GROUP BY
  aisle_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_depart_30" --flatten_results --replace "
SELECT
  department_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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
  b.last_buy <= 30
) as s
GROUP BY
   department_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item_reordered_30" --flatten_results --replace "
SELECT
  user_id,
  product_id,
  CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
  CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
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

###

bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_item_recent_reordered" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  a.product_id as product_id,
  AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
  AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
  AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
  AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
  AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  user_id, product_id
"
####
bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_aisle_recent_reordered" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  a.aisle_id as aisle_id,
  AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
  AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
  AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
  AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
  AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  user_id, aisle_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_depart_recent_reordered" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  a.department_id as department_id,
  AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
  AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
  AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
  AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
  AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  user_id, department_id
"

bq query --max_rows 20  --allow_large_results --destination_table "instacart.aisle_recent_reordered" --flatten_results --replace "
SELECT
  a.aisle_id as aisle_id,
  AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
  AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
  AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
  AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
  AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  aisle_id
"


bq query --max_rows 20  --allow_large_results --destination_table "instacart.depart_recent_reordered" --flatten_results --replace "
SELECT
  a.department_id as department_id,
  AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
  AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
  AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
  AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
  AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  department_id
"



####


bq query --max_rows 20  --allow_large_results --destination_table "instacart.item_recent_reordered" --flatten_results --replace "
SELECT
  a.product_id as product_id,
  AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as i_under7,
  AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as i_under14,
  AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as i_under21,
  AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as i_under28,
  AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as i_over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  product_id
"

bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_recent_reordered" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as u_under7,
  AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as u_under14,
  AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as u_under21,
  AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as u_under28,
  AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as u_over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  user_id
"


bq query --max_rows 1  --maximum_billing_tier 3 --allow_large_results --destination_table "instacart.dmt_train_only_rebuy" --flatten_results --replace "
SELECT
  CASE WHEN tr.reordered is not null THEN tr.reordered ELSE 0 END as target,
  o.user_id,
  p.aisle_id,
  p.department_id,
  o.product_id,
  o.order_id,
  o.order_number,
  o.order_dow,
  o.order_hour_of_day,
  o.days_since_prior_order,
  o.cum_days,
  du.*,
  dui3.*,
  dd.*,
  du3.*,
  da3.*,
  dd3.*,
  udd.*,
  udd3.*,
  di.*,
  di3.*,
  ddi3.*,
  ddd3.*,
  dai3.*,
  dddi3.*,
  u.*,
  u2.*,
  uc.*,
  i.*,
  i2.*,
  l.*,
  la.*,
  ld.*,
  l2.*,
  la2.*,
  ld2.*,
  ui.*,
  ua.*,
  ud.*,
  ui3.*,
  ua3.*,
  ud3.*,
  rui.*,
  ru.*,
  ri.*
FROM
  [instacart.only_rebuy_train] as o
LEFT OUTER JOIN
  [instacart.products] as p
ON
  o.product_id = p.product_id
LEFT OUTER JOIN
  [instacart.diff_user_item] as du
ON
  du.user_id = o.user_id AND  o.product_id = du.product_id
LEFT OUTER JOIN
  [instacart.diff_user_item_reordered_30] as dui3
ON
  dui3.user_id = o.user_id AND  o.product_id = dui3.product_id
LEFT OUTER JOIN
  [instacart.diff_user_aisle_reordered_30] as dai3
ON
  dai3.user_id = o.user_id AND p.aisle_id = dai3.aisle_id
LEFT OUTER JOIN
  [instacart.diff_user_depart_reordered_30] as dddi3
ON
  dddi3.user_id = o.user_id AND  p.department_id = dddi3.department_id
LEFT OUTER JOIN
  [instacart.diff_user_depart] as dd
ON
  dd.user_id = o.user_id AND  p.product_id = dd.department_id
LEFT OUTER JOIN
  [instacart.diff_user_item_30] as du3
ON
  du3.user_id = o.user_id AND  o.product_id = du3.product_id
LEFT OUTER JOIN
  [instacart.diff_user_aisle_30] as da3
ON
  da3.user_id = o.user_id AND  p.aisle_id = da3.aisle_id
LEFT OUTER JOIN
  [instacart.diff_user_depart_30] as dd3
ON
  dd3.user_id = o.user_id AND  p.product_id = dd3.department_id
LEFT OUTER JOIN
  [instacart.diff_user] as udd
ON
  udd.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.diff_user_30] as udd3
ON
  udd3.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.diff_item] as di
ON
  di.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.diff_item_30] as di3
ON
  di3.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.diff_aisle_30] as ddi3
ON
  ddi3.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.diff_depart_30] as ddd3
ON
  ddd3.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.dmt_user] as u
ON
  u.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_user2_30] as u2
ON
  u2.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.user_cart_30] as uc
ON
  uc.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_item] as i
ON
  i.i1_product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_item2_30] as i2
ON
  i2.i1_product_id = o.product_id
LEFT OUTER JOIN
  [instacart.last_buy] as l
ON
  l.user_id = o.user_id AND l.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.last_buy_aisle] as la
ON
  la.user_id = o.user_id AND la.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.last_buy_depart] as ld
ON
  ld.user_id = o.user_id AND ld.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.last_buy_2] as l2
ON
  l2.user_id = o.user_id AND l2.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.last_buy_aisle_2] as la2
ON
  la2.user_id = o.user_id AND la2.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.last_buy_depart_2] as ld2
ON
  ld2.user_id = o.user_id AND ld2.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.dmt_user_item] as ui
ON
  ui.user_id = o.user_id AND ui.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user_aisle] as ua
ON
  ua.user_id = o.user_id AND ua.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.dmt_user_depart] as ud
ON
  ud.user_id = o.user_id AND ud.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.dmt_user_item_30] as ui3
ON
  ui3.user_id = o.user_id AND ui3.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user_aisle_30] as ua3
ON
  ua3.user_id = o.user_id AND ua3.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.dmt_user_depart_30] as ud3
ON
  ud3.user_id = o.user_id AND ud3.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.user_item_recent_reordered] as rui
ON
  rui.user_id = o.user_id AND rui.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.user_recent_reordered] as ru
ON
  ru.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.item_recent_reordered] as ri
ON
  ri.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.df_train] as tr
ON
  tr.user_id = o.user_id AND tr.product_id = o.product_id AND tr.order_id = o.order_id
"



bq query --maximum_billing_tier 2 --max_rows 1  --allow_large_results --destination_table "instacart.dmt_test_only_rebuy" --flatten_results --replace "
SELECT
  o.user_id,
  p.aisle_id,
  p.department_id,
  o.product_id,
  o.order_id,
  o.order_number,
  o.order_dow,
  o.order_hour_of_day,
  o.days_since_prior_order,
  o.cum_days,
  du.*,
  dui3.*,
  dd.*,
  du3.*,
  da3.*,
  dd3.*,
  udd.*,
  udd3.*,
  di.*,
  di3.*,
  ddi3.*,
  ddd3.*,
  dai3.*,
  dddi3.*,
  u.*,
  u2.*,
  uc.*,
  i.*,
  i2.*,
  l.*,
  la.*,
  ld.*,
  l2.*,
  la2.*,
  ld2.*,
  ui.*,
  ua.*,
  ud.*,
  ui3.*,
  ua3.*,
  ud3.*,
  rui.*,
  ru.*,
  ri.*
FROM
  [instacart.only_rebuy_test] as o
LEFT OUTER JOIN
  [instacart.products] as p
ON
  o.product_id = p.product_id
LEFT OUTER JOIN
  [instacart.diff_user_item] as du
ON
  du.user_id = o.user_id AND  o.product_id = du.product_id
LEFT OUTER JOIN
  [instacart.diff_user_item_reordered_30] as dui3
ON
  dui3.user_id = o.user_id AND  o.product_id = dui3.product_id
LEFT OUTER JOIN
  [instacart.diff_user_aisle_reordered_30] as dai3
ON
  dai3.user_id = o.user_id AND p.aisle_id = dai3.aisle_id
LEFT OUTER JOIN
  [instacart.diff_user_depart_reordered_30] as dddi3
ON
  dddi3.user_id = o.user_id AND  p.department_id = dddi3.department_id
LEFT OUTER JOIN
  [instacart.diff_user_depart] as dd
ON
  dd.user_id = o.user_id AND  p.product_id = dd.department_id
LEFT OUTER JOIN
  [instacart.diff_user_item_30] as du3
ON
  du3.user_id = o.user_id AND  o.product_id = du3.product_id
LEFT OUTER JOIN
  [instacart.diff_user_aisle_30] as da3
ON
  da3.user_id = o.user_id AND  p.aisle_id = da3.aisle_id
LEFT OUTER JOIN
  [instacart.diff_user_depart_30] as dd3
ON
  dd3.user_id = o.user_id AND  p.product_id = dd3.department_id
LEFT OUTER JOIN
  [instacart.diff_user] as udd
ON
  udd.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.diff_user_30] as udd3
ON
  udd3.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.diff_item] as di
ON
  di.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.diff_item_30] as di3
ON
  di3.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.diff_aisle_30] as ddi3
ON
  ddi3.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.diff_depart_30] as ddd3
ON
  ddd3.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.dmt_user] as u
ON
  u.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_user2_30] as u2
ON
  u2.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.user_cart_30] as uc
ON
  uc.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_item] as i
ON
  i.i1_product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_item2_30] as i2
ON
  i2.i1_product_id = o.product_id
LEFT OUTER JOIN
  [instacart.last_buy] as l
ON
  l.user_id = o.user_id AND l.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.last_buy_aisle] as la
ON
  la.user_id = o.user_id AND la.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.last_buy_depart] as ld
ON
  ld.user_id = o.user_id AND ld.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.last_buy_2] as l2
ON
  l2.user_id = o.user_id AND l2.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.last_buy_aisle_2] as la2
ON
  la2.user_id = o.user_id AND la2.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.last_buy_depart_2] as ld2
ON
  ld2.user_id = o.user_id AND ld2.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.dmt_user_item] as ui
ON
  ui.user_id = o.user_id AND ui.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user_aisle] as ua
ON
  ua.user_id = o.user_id AND ua.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.dmt_user_depart] as ud
ON
  ud.user_id = o.user_id AND ud.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.dmt_user_item_30] as ui3
ON
  ui3.user_id = o.user_id AND ui3.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user_aisle_30] as ua3
ON
  ua3.user_id = o.user_id AND ua3.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.dmt_user_depart_30] as ud3
ON
  ud3.user_id = o.user_id AND ud3.department_id = p.department_id
LEFT OUTER JOIN
  [instacart.df_train] as tr
ON
  tr.user_id = o.user_id AND tr.product_id = o.product_id AND tr.order_id = o.order_id
LEFT OUTER JOIN
  [instacart.user_item_recent_reordered] as rui
ON
  rui.user_id = o.user_id AND rui.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.user_recent_reordered] as ru
ON
  ru.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.item_recent_reordered] as ri
ON
  ri.product_id = o.product_id
"

gsutil -m rm -r gs://kaggle-instacart-takami/data/

bq extract --compression GZIP instacart.dmt_train_only_rebuy gs://kaggle-instacart-takami/data/dmt_train_only_rebuy/data*.csv.gz
bq extract --compression GZIP instacart.dmt_test_only_rebuy gs://kaggle-instacart-takami/data/dmt_test_only_rebuy/data*.csv.gz

rm -rf ../data/
gsutil -m cp -r gs://kaggle-instacart-takami/data/ ../

