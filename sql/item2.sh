

bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_fund" --flatten_results --replace "  
SELECT
  product_id,
  count(1) as item_user_cnt,
  count(distinct user_id) as item_usr_cnt,
  count(distinct department_id) as item_depart_cnt,
  count(distinct aisle_id) as item_aisle_cnt,
  count(distinct order_id) as item_order_cnt,
  count(distinct order_id) / count(1) as item_order_rate,
  AVG(days_since_prior_order) as avg_item_days_since_prior_order,
  MIN(days_since_prior_order) as min_item_days_since_prior_order,
  MAX(days_since_prior_order) as max_item_days_since_prior_order,
  MAX(order_hour_of_day) as max_order_hour_of_day,
  MIN(order_hour_of_day) as min_order_hour_of_day,
  AVG(order_hour_of_day) as avg_order_hour_of_day,
  AVG(reordered) as avg_item_reordered,
  SUM(reordered) as sum_item_reordered
FROM
  [instacart.df_prior2]
GROUP BY
  product_id
"

bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_dow" --flatten_results --replace "  
SELECT
  product_id,
  sum(CASE WHEN order_dow = 0  THEN 1 ELSE 0 END) AS  order_dow_0,
  sum(CASE WHEN order_dow = 1  THEN 1 ELSE 0 END) AS  order_dow_1,
  sum(CASE WHEN order_dow = 2  THEN 1 ELSE 0 END) AS  order_dow_2,
  sum(CASE WHEN order_dow = 3  THEN 1 ELSE 0 END) AS  order_dow_3,
  sum(CASE WHEN order_dow = 4  THEN 1 ELSE 0 END) AS  order_dow_4,
  sum(CASE WHEN order_dow = 5  THEN 1 ELSE 0 END) AS  order_dow_5,
  sum(CASE WHEN order_dow = 6  THEN 1 ELSE 0 END) AS  order_dow_6
FROM
  [instacart.df_prior2]
GROUP BY
  product_id
"

bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_hour" --flatten_results --replace "  
SELECT
  product_id,
  sum(CASE WHEN order_hour_of_day = 0  THEN 1 ELSE 0 END) AS order_hour_of_day_0,
  sum(CASE WHEN order_hour_of_day = 1  THEN 1 ELSE 0 END) AS order_hour_of_day_1,
  sum(CASE WHEN order_hour_of_day = 2  THEN 1 ELSE 0 END) AS order_hour_of_day_2,
  sum(CASE WHEN order_hour_of_day = 3  THEN 1 ELSE 0 END) AS order_hour_of_day_3,
  sum(CASE WHEN order_hour_of_day = 4  THEN 1 ELSE 0 END) AS order_hour_of_day_4,
  sum(CASE WHEN order_hour_of_day = 5  THEN 1 ELSE 0 END) AS order_hour_of_day_5,
  sum(CASE WHEN order_hour_of_day = 6  THEN 1 ELSE 0 END) AS order_hour_of_day_6,
  sum(CASE WHEN order_hour_of_day = 7  THEN 1 ELSE 0 END) AS order_hour_of_day_7,
  sum(CASE WHEN order_hour_of_day = 8  THEN 1 ELSE 0 END) AS order_hour_of_day_8,
  sum(CASE WHEN order_hour_of_day = 9  THEN 1 ELSE 0 END) AS order_hour_of_day_9,
  sum(CASE WHEN order_hour_of_day = 10  THEN 1 ELSE 0 END) AS order_hour_of_day_10,
  sum(CASE WHEN order_hour_of_day = 11  THEN 1 ELSE 0 END) AS order_hour_of_day_11,
  sum(CASE WHEN order_hour_of_day = 12  THEN 1 ELSE 0 END) AS order_hour_of_day_12,
  sum(CASE WHEN order_hour_of_day = 13  THEN 1 ELSE 0 END) AS order_hour_of_day_13,
  sum(CASE WHEN order_hour_of_day = 14  THEN 1 ELSE 0 END) AS order_hour_of_day_14,
  sum(CASE WHEN order_hour_of_day = 15  THEN 1 ELSE 0 END) AS order_hour_of_day_15,
  sum(CASE WHEN order_hour_of_day = 16  THEN 1 ELSE 0 END) AS order_hour_of_day_16,
  sum(CASE WHEN order_hour_of_day = 17  THEN 1 ELSE 0 END) AS order_hour_of_day_17,
  sum(CASE WHEN order_hour_of_day = 18  THEN 1 ELSE 0 END) AS order_hour_of_day_18,
  sum(CASE WHEN order_hour_of_day = 19  THEN 1 ELSE 0 END) AS order_hour_of_day_19,
  sum(CASE WHEN order_hour_of_day = 20  THEN 1 ELSE 0 END) AS order_hour_of_day_20,
  sum(CASE WHEN order_hour_of_day = 21  THEN 1 ELSE 0 END) AS order_hour_of_day_21,
  sum(CASE WHEN order_hour_of_day = 22  THEN 1 ELSE 0 END) AS order_hour_of_day_22,
  sum(CASE WHEN order_hour_of_day = 23  THEN 1 ELSE 0 END) AS order_hour_of_day_23
FROM
  [instacart.df_prior2]
GROUP BY
  product_id
"


bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_depart" --flatten_results --replace "  
SELECT
  product_id,
  sum(CASE WHEN department_id = 1  THEN 1 ELSE 0 END) AS department_id_1,
  sum(CASE WHEN department_id = 2  THEN 1 ELSE 0 END) AS department_id_2,
  sum(CASE WHEN department_id = 3  THEN 1 ELSE 0 END) AS department_id_3,
  sum(CASE WHEN department_id = 4  THEN 1 ELSE 0 END) AS department_id_4,
  sum(CASE WHEN department_id = 5  THEN 1 ELSE 0 END) AS department_id_5,
  sum(CASE WHEN department_id = 6  THEN 1 ELSE 0 END) AS department_id_6,
  sum(CASE WHEN department_id = 7  THEN 1 ELSE 0 END) AS department_id_7,
  sum(CASE WHEN department_id = 8  THEN 1 ELSE 0 END) AS department_id_8,
  sum(CASE WHEN department_id = 9  THEN 1 ELSE 0 END) AS department_id_9,
  sum(CASE WHEN department_id = 10  THEN 1 ELSE 0 END) AS department_id_10,
  sum(CASE WHEN department_id = 11  THEN 1 ELSE 0 END) AS department_id_11,
  sum(CASE WHEN department_id = 12  THEN 1 ELSE 0 END) AS department_id_12,
  sum(CASE WHEN department_id = 13  THEN 1 ELSE 0 END) AS department_id_13,
  sum(CASE WHEN department_id = 14  THEN 1 ELSE 0 END) AS department_id_14,
  sum(CASE WHEN department_id = 15  THEN 1 ELSE 0 END) AS department_id_15,
  sum(CASE WHEN department_id = 16  THEN 1 ELSE 0 END) AS department_id_16,
  sum(CASE WHEN department_id = 17  THEN 1 ELSE 0 END) AS department_id_17,
  sum(CASE WHEN department_id = 18  THEN 1 ELSE 0 END) AS department_id_18,
  sum(CASE WHEN department_id = 19  THEN 1 ELSE 0 END) AS department_id_19,
  sum(CASE WHEN department_id = 20  THEN 1 ELSE 0 END) AS department_id_20,
  sum(CASE WHEN department_id = 21  THEN 1 ELSE 0 END) AS department_id_21
FROM
  [instacart.df_prior2]
GROUP BY
  product_id
"

bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_item2_30" --flatten_results --replace "  
SELECT
  *
FROM
  [instacart.item_fund] as i1
LEFT OUTER JOIN
  [instacart.item_dow] as i2
ON
  i1.product_id = i2.product_id
LEFT OUTER JOIN
  [instacart.item_hour] as i3
ON
  i1.product_id = i3.product_id
LEFT OUTER JOIN
  [instacart.item_depart] as i4
ON
  i1.product_id = i4.product_id
"
