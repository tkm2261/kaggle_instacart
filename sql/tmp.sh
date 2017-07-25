
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
  dd.*,
  du3.*,
  dd3.*,
  udd.*,
  udd3.*,
  di.*,
  di3.*,
  u.*,
  uc.*,
  i.*,
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
  ud3.*
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
  [instacart.diff_user_depart] as dd
ON
  dd.user_id = o.user_id AND  p.product_id = dd.department_id
LEFT OUTER JOIN
  [instacart.diff_user_item_30] as du3
ON
  du3.user_id = o.user_id AND  o.product_id = du3.product_id
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
  [instacart.dmt_user] as u
ON
  u.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.user_cart_30] as uc
ON
  uc.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_item] as i
ON
  i.i1_product_id = o.product_id
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
  dd.*,
  du3.*,
  dd3.*,
  udd.*,
  udd3.*,
  di.*,
  di3.*,
  u.*,
  uc.*,
  i.*,
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
  ud3.*
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
  [instacart.diff_user_depart] as dd
ON
  dd.user_id = o.user_id AND  p.product_id = dd.department_id
LEFT OUTER JOIN
  [instacart.diff_user_item_30] as du3
ON
  du3.user_id = o.user_id AND  o.product_id = du3.product_id
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
  [instacart.dmt_user] as u
ON
  u.u1_user_id = o.user_id
LEFT OUTER JOIN
  [instacart.user_cart_30] as uc
ON
  uc.user_id = o.user_id
LEFT OUTER JOIN
  [instacart.dmt_item] as i
ON
  i.i1_product_id = o.product_id
LEFT OUTER JOIN
  [instacart.dmt_user_item] as ui
ON
  ui.user_id = o.user_id AND ui.product_id = o.product_id
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
"

gsutil -m rm -r gs://kaggle-instacart-takami/data/

bq extract --compression GZIP instacart.dmt_train_only_rebuy gs://kaggle-instacart-takami/data/dmt_train_only_rebuy/data*.csv.gz
bq extract --compression GZIP instacart.dmt_test_only_rebuy gs://kaggle-instacart-takami/data/dmt_test_only_rebuy/data*.csv.gz


