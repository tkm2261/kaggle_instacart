

bq query --max_rows 1  --allow_large_results --destination_table "instacart.user_aisle" --flatten_results --replace "
SELECT
  user_id,
  aisle_id,
  count(1) as cnt,
  sum(reordered) as reordered
FROM
  [instacart.df_prior2]
GROUP BY
  user_id, aisle_id
"
bq extract --compression GZIP instacart.user_aisle gs://kaggle-instacart-takami/tmp/user_aisle.csv.gz

gsutil -m cp -r gs://kaggle-instacart-takami/tmp/user_aisle.csv.gz .
