# attention_pairwise_ranking
## 環境
docker : nvcr.io/nvidia/pytorch:21.03-py3


## ファイル構造  
Dataset/  
&ensp;    |--apply_eyeliner/  
&ensp;   |--braid_hair/  
&ensp;    |--origami/  
&ensp;    |--scrambled_eggs/  
&ensp;    |--tie_tie/  
APR_demo  
&ensp;    |--Datalist/  
&ensp;    |  |--apply_eyeliner/  
&ensp;    |  |  |--test.txt  
&ensp;    |  |  |--train.txt  
&ensp;    |  |--braid_hair/  
&ensp;    |  |  |--test.txt  
&ensp;    |  |  |--train.txt  
&ensp;    |  |--origami/  
&ensp;    |  |  |--test.txt  
&ensp;    |  |  |--train.txt  
&ensp;    |  |--scrambled_eggs/  
&ensp;    |  |  |--test.txt  
&ensp;    |  |  |--train.txt  
&ensp;    |  |--tie_tie/  
&ensp;    |  |  |--test.txt  
&ensp;    |  |  |--train.txt  
&ensp;    |--README.txt  
&ensp;    |--best_model/  
&ensp;    |  |--apply_eyeliner/  
&ensp;    |  |--braid_hair/  
&ensp;    |  |--origami/  
&ensp;    |  |--scrambled_eggs/  
&ensp;    |  |--tie_tie/  
&ensp;    |--dataset.py  
&ensp;    |--image/  
&ensp;    |  |--apply_eyeliner/  
&ensp;    |  |--braid_hair/  
&ensp;    |  |--origami/  
&ensp;    |  |--scrambled_eggs/  
&ensp;    |  |--tie_tie/  
&ensp;    |--main.py  
&ensp;    |--ops/  
&ensp;    |--opts.py  
&ensp;    |--resnet.py  
&ensp;    |--run.sh  
&ensp;    |--tf_model_zoo/  
&ensp;    |--transforms.py  

## train&test
bash run.sh 

## 注意事項
- Datalistディレクトリ内にある各タスクのpathはデータセットの場所によって変更して下さい．
- データセットはここからダウンロードして下さい．https://github.com/hazeld/rank-aware-attention-network
- このコードはBEST datasetを前提として作成してあります．EPICK Skills datasetを使用する場合は書き換えが必要になります．
- attention mapの可視化は評価時のみです．可視化結果はimageディレクトリの中に保存されます．
