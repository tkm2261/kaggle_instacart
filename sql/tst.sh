
bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_item_recent" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  a.product_id as product_id,
  sum(CASE WHEN b.last_buy <=7 THEN 1 ELSE 0 END) as under7,
  sum(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN 1 ELSE 0 END) as under14,
  sum(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN 1 ELSE 0 END) as under21,
  sum(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN 1 ELSE 0 END) as under28,
  sum(CASE WHEN b.last_buy > 28 THEN 1 ELSE 0 END) as over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  user_id, product_id
"

bq query --max_rows 20  --allow_large_results --destination_table "instacart.item_recent" --flatten_results --replace "
SELECT
  a.product_id as product_id,
  sum(CASE WHEN b.last_buy <=7 THEN 1 ELSE 0 END) as under7,
  sum(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN 1 ELSE 0 END) as under14,
  sum(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN 1 ELSE 0 END) as under21,
  sum(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN 1 ELSE 0 END) as under28,
  sum(CASE WHEN b.last_buy > 28 THEN 1 ELSE 0 END) as over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  product_id
"

bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_recent" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  sum(CASE WHEN b.last_buy <=7 THEN 1 ELSE 0 END) as under7,
  sum(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN 1 ELSE 0 END) as under14,
  sum(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN 1 ELSE 0 END) as under21,
  sum(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN 1 ELSE 0 END) as under28,
  sum(CASE WHEN b.last_buy > 28 THEN 1 ELSE 0 END) as over28
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
GROUP BY
  user_id
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

bq query --max_rows 1  --allow_large_results --destination_table "instacart.order_30days" --flatten_results --replace "
SELECT
a.order_id order_id,
FROM
  [instacart.df_prior] as a
LEFT OUTER JOIN
  [instacart.cum_orders] as b
ON
  a.order_id = b.order_id
WHERE
  b.last_buy <= 30
GROUP BY
order_id
"


bq query --max_rows 1  --allow_large_results --destination_table "instacart.aisle_fund" --flatten_results --replace "  
SELECT
  aisle_id,
  count(1) as aisle_user_cnt,
  EXACT_COUNT_DISTINCT( user_id) as aisle_usr_cnt,
  EXACT_COUNT_DISTINCT( department_id) as aisle_depart_cnt,
  EXACT_COUNT_DISTINCT( aisle_id) as aisle_aisle_cnt,
  EXACT_COUNT_DISTINCT( order_id) as aisle_order_cnt,
  EXACT_COUNT_DISTINCT( order_id) / count(1) as aisle_order_rate,
  AVG(days_since_prior_order) as avg_aisle_days_since_prior_order,
  MIN(days_since_prior_order) as min_aisle_days_since_prior_order,
  MAX(days_since_prior_order) as max_aisle_days_since_prior_order,
  MAX(order_hour_of_day) as max_order_hour_of_day,
  MIN(order_hour_of_day) as min_order_hour_of_day,
  AVG(order_hour_of_day) as avg_order_hour_of_day,
  AVG(reordered) as avg_aisle_reordered,
  SUM(reordered) as sum_aisle_reordered,
  AVG(order_dow) as avg_order_dow
FROM
  [instacart.df_prior]
GROUP BY
  aisle_id
"

bq query --max_rows 1  --allow_large_results --destination_table "instacart.department_fund" --flatten_results --replace "  
SELECT
  department_id,
  count(1) as department_user_cnt,
  EXACT_COUNT_DISTINCT( user_id) as department_usr_cnt,
  EXACT_COUNT_DISTINCT( department_id) as department_depart_cnt,
  EXACT_COUNT_DISTINCT( department_id) as department_department_cnt,
  EXACT_COUNT_DISTINCT( order_id) as department_order_cnt,
  EXACT_COUNT_DISTINCT( order_id) / count(1) as department_order_rate,
  AVG(days_since_prior_order) as avg_department_days_since_prior_order,
  MIN(days_since_prior_order) as min_department_days_since_prior_order,
  MAX(days_since_prior_order) as max_department_days_since_prior_order,
  MAX(order_hour_of_day) as max_order_hour_of_day,
  MIN(order_hour_of_day) as min_order_hour_of_day,
  AVG(order_hour_of_day) as avg_order_hour_of_day,
  AVG(reordered) as avg_department_reordered,
  SUM(reordered) as sum_department_reordered,
  AVG(order_dow) as avg_order_dow
FROM
  [instacart.df_prior]
GROUP BY
  department_id
"

bq extract --compression GZIP instacart.aisle_fund gs://kaggle-instacart-takami/tmp/aisle_fund.csv.gz
bq extract --compression GZIP instacart.department_fund gs://kaggle-instacart-takami/tmp/department.csv.gz

bq extract --compression GZIP instacart.aisle_fund2 gs://kaggle-instacart-takami/tmp/aisle_fund2.csv.gz
bq extract --compression GZIP instacart.department_fund2 gs://kaggle-instacart-takami/tmp/department_fund2.csv.gz


bq query --max_rows 1  --allow_large_results --destination_table "instacart.aisle_fund2" --flatten_results --replace "  
SELECT
  aisle_id,
  count(1) as aisle_user_cnt_aisle2,
  EXACT_COUNT_DISTINCT( user_id) as aisle_usr_cnt_aisle2,
  EXACT_COUNT_DISTINCT( department_id) as aisle_depart_cnt_aisle2,
  EXACT_COUNT_DISTINCT( aisle_id) as aisle_aisle_cnt_aisle2,
  EXACT_COUNT_DISTINCT( order_id) as aisle_order_cnt_aisle2,
  EXACT_COUNT_DISTINCT( order_id) / count(1) as aisle_order_rate_aisle2,
  AVG(days_since_prior_order) as avg_aisle_days_since_prior_order_aisle2,
  MIN(days_since_prior_order) as min_aisle_days_since_prior_order_aisle2,
  MAX(days_since_prior_order) as max_aisle_days_since_prior_order_aisle2,
  MAX(order_hour_of_day) as max_order_hour_of_day_aisle2,
  MIN(order_hour_of_day) as min_order_hour_of_day_aisle2,
  AVG(order_hour_of_day) as avg_order_hour_of_day_aisle2,
  AVG(reordered) as avg_aisle_reordered_aisle2,
  SUM(reordered) as sum_aisle_reordered_aisle2,
  AVG(order_dow) as avg_order_dow_aisle2
FROM
  [instacart.df_prior2]
GROUP BY
  aisle_id
"

bq query --max_rows 1  --allow_large_results --destination_table "instacart.department_fund2" --flatten_results --replace "  
SELECT
  department_id,
  count(1) as department_user_cnt_depart2,
  EXACT_COUNT_DISTINCT( user_id) as department_usr_cnt_depart2,
  EXACT_COUNT_DISTINCT( department_id) as department_depart_cnt_depart2,
  EXACT_COUNT_DISTINCT( department_id) as department_department_cnt_depart2,
  EXACT_COUNT_DISTINCT( order_id) as department_order_cnt_depart2,
  EXACT_COUNT_DISTINCT( order_id) / count(1) as department_order_rate_depart2,
  AVG(days_since_prior_order) as avg_department_days_since_prior_order_depart2,
  MIN(days_since_prior_order) as min_department_days_since_prior_order_depart2,
  MAX(days_since_prior_order) as max_department_days_since_prior_order_depart2,
  MAX(order_hour_of_day) as max_order_hour_of_day_depart2,
  MIN(order_hour_of_day) as min_order_hour_of_day_depart2,
  AVG(order_hour_of_day) as avg_order_hour_of_day_depart2,
  AVG(reordered) as avg_department_reordered_depart2,
  SUM(reordered) as sum_department_reordered_depart2,
  AVG(order_dow) as avg_order_dow_depart2
FROM
  [instacart.df_prior]
GROUP BY
  department_id
"

bq extract --compression GZIP instacart.user_aisle_recent_reordered gs://kaggle-instacart-takami/tmp/user_aisle_recent_reordered.csv.gz
bq extract --compression GZIP instacart.user_depart_recent_reordered gs://kaggle-instacart-takami/tmp/user_depart_recent_reordered.csv.gz

bq extract --compression GZIP instacart.aisle_recent_reordered gs://kaggle-instacart-takami/tmp/aisle_recent_reordered.csv.gz
bq extract --compression GZIP instacart.depart_recent_reordered gs://kaggle-instacart-takami/tmp/depart_recent_reordered.csv.gz
