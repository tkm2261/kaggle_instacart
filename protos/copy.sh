mkdir $1/
cp *_idx.csv $1/
cp train_cv_tmp.pkl  $1/
cp test_tmp.pkl  $1/
aws s3 sync $1/ s3://takami-kaggle-dr/kaggle_instacart/protos/$1/
