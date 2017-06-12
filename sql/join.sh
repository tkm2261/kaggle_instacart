bq query --allow_large_results --destination_table "instacart.cum_orders" --flatten_results --replace "
SELECT
  user_id,
  order_id,
  eval_set,
  order_number,
  days_since_prior_order,
  SUM(days_since_prior_order) OVER (PARTITION BY user_id ORDER BY order_number ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_days
FROM
  [instacart.orders]
order by
  user_id, order_number
"



bq query --allow_large_results --destination_table "instacart.only_rebuy" --flatten_results --replace "
SELECT
  user_id,
  product_id
FROM
  [instacart.df_prior]
GROUP BY
  user_id, product_id
"

bq query --allow_large_results --destination_table "instacart.only_rebuy_train" --flatten_results --replace "
SELECT
  r.user_id as user_id,
  r.product_id as product_id,
  o.order_id as order_id,
  o.order_number as order_number,
  o.order_dow as order_dow,
  o.order_hour_of_day as order_hour_of_day,
  o.days_since_prior_order as days_since_prior_order,
  c.cum_days as cum_days
FROM
  [instacart.only_rebuy] as r
INNER JOIN
  (SELECT * FROM [instacart.orders] WHERE eval_set='train') as o
ON
  r.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.cum_orders] as c
ON
  c.order_id = o.order_id
"

bq query --allow_large_results --destination_table "instacart.only_rebuy_test" --flatten_results --replace "
SELECT
  r.user_id as user_id,
  r.product_id as product_id,
  o.order_id as order_id,
  o.order_number as order_number,
  o.order_dow as order_dow,
  o.order_hour_of_day as order_hour_of_day,
  o.days_since_prior_order as days_since_prior_order,
  c.cum_days as cum_days
FROM
  [instacart.only_rebuy] as r
INNER JOIN
  (SELECT * FROM [instacart.orders] WHERE eval_set='test') as o
ON
  r.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.cum_orders] as c
ON
  c.order_id = o.order_id
"


bq query --allow_large_results --destination_table "instacart.last_buy" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  a.product_id as product_id,
  a.max_order_number as max_order_number,
  o.order_id,
  c.cum_days as cum_days
FROM
  (
  SELECT
    user_id,
    product_id,
    MAX(order_number) as max_order_number
  FROM
    [instacart.df_prior]
  GROUP BY
    user_id,
    product_id
  ) as a
LEFT OUTER JOIN
  [instacart.orders] as o
ON
  a.user_id = o.user_id AND
  a.max_order_number = o.order_number
LEFT OUTER JOIN
  [instacart.cum_orders] as c
ON
  c.order_id = o.order_id
"


bq query --allow_large_results --destination_table "instacart.last_buy_aisle" --flatten_results --replace "
SELECT
  a.user_id as user_id,
  a.aisle_id as aisle_id,
  a.max_order_number as max_order_number,
  o.order_id,
  c.cum_days as cum_days
FROM
  (
  SELECT
    user_id,
    aisle_id,
    MAX(order_number) as max_order_number
  FROM
    [instacart.df_prior]
  GROUP BY
    user_id,
    aisle_id
  ) as a
LEFT OUTER JOIN
  [instacart.orders] as o
ON
  a.user_id = o.user_id AND
  a.max_order_number = o.order_number
LEFT OUTER JOIN
  [instacart.cum_orders] as c
ON
  c.order_id = o.order_id
"

bq query --allow_large_results --destination_table "instacart.dmt_user_item" --flatten_results --replace "
SELECT
  user_id,
  product_id,
  count(1) cnt_user_item,
  count(distinct order_id) cnt_user_order,
  max(reordered) max_reordered,
  sum(reordered) sum_reordered,
  avg(reordered) avg_reordered,
  AVG(days_since_prior_order) as avg_days_since_prior_order,
  MAX(days_since_prior_order) as max_days_since_prior_order,
  MIN(days_since_prior_order) as min_days_since_prior_order
FROM
  [instacart.df_prior]
GROUP BY
  user_id, product_id
"

bq query --allow_large_results --destination_table "instacart.dmt_user_aisle" --flatten_results --replace "
SELECT
  user_id,
  aisle_id,
  count(1) cnt_user_aisle,
  count(distinct order_id) cnt_aisle_order,
  max(reordered) max_reordered,
  sum(reordered) sum_reordered,
  avg(reordered) avg_reordered
FROM
  [instacart.df_prior]
GROUP BY
  user_id, aisle_id
"


bq query --maximum_billing_tier 2 --allow_large_results --destination_table "instacart.dmt_train_only_rebuy" --flatten_results --replace "
SELECT
  CASE WHEN tr.order_number is not null THEN 1 ELSE 0 END as target,
  o.user_id,
  p.aisle_id,
  o.product_id,
  o.order_id,
  o.order_number,
  o.order_dow,
  o.order_hour_of_day,
  o.days_since_prior_order,
  o.cum_days,
  l.cum_days,
  u.*,
  i.*,
  ui.*,
  la.*,
  ua.*
FROM
  [instacart.only_rebuy_train] as o
LEFT OUTER JOIN
  [instacart.last_buy] as l
ON
  l.user_id = o.user_id AND l.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user] as u
ON
  u.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_item] as i
ON
  i.i1_product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user_item] as ui
ON
  ui.user_id = o.user_id AND ui.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.products] as p
ON
  o.product_id = p.product_id
LEFT OUTER JOIN
  [instacart.last_buy_aisle] as la
ON
  la.user_id = o.user_id AND la.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.dmt_user_aisle] as ua
ON
  ua.user_id = o.user_id AND ua.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.df_train] as tr
ON
  tr.user_id = o.user_id AND tr.product_id = o.product_id
"


bq query --allow_large_results --destination_table "instacart.dmt_test_only_rebuy" --flatten_results --replace "
SELECT
  o.user_id,
  p.aisle_id,
  o.product_id,
  o.order_id,
  o.order_number,
  o.order_dow,
  o.order_hour_of_day,
  o.days_since_prior_order,
  o.cum_days,
  l.cum_days,
  u.*,
  i.*,
  ui.*,
  la.*,
  ua.*
FROM
  [instacart.only_rebuy_test] as o
LEFT OUTER JOIN
  [instacart.last_buy] as l
ON
  l.user_id = o.user_id AND l.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user] as u
ON
  u.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_item] as i
ON
  i.i1_product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user_item] as ui
ON
  ui.user_id = o.user_id AND ui.product_id = o.product_id
LEFT OUTER JOIN
  [instacart.products] as p
ON
  o.product_id = p.product_id
LEFT OUTER JOIN
  [instacart.last_buy_aisle] as la
ON
  la.user_id = o.user_id AND la.aisle_id = p.aisle_id
LEFT OUTER JOIN
  [instacart.dmt_user_aisle] as ua
ON
  ua.user_id = o.user_id AND ua.aisle_id = p.aisle_id
"

gsutil -m rm -r gs://kaggle_quora/data/

bq extract --compression GZIP instacart.dmt_train_only_rebuy gs://kaggle_quora/data/dmt_train_only_rebuy/data*.csv.gz
bq extract --compression GZIP instacart.dmt_test_only_rebuy gs://kaggle_quora/data/dmt_test_only_rebuy/data*.csv.gz

rm -rf ../data/
gsutil -m cp -r gs://kaggle_quora/data/ ../
