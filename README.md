# attention_pairwise_ranking
環境
docker : nvcr.io/nvidia/pytorch:21.03-py3


ファイル構造
Dataset/
    |--apply_eyeliner/
    |--braid_hair/
    |--origami/
    |--scrambled_eggs/
    |--tie_tie/
APR_demo
    |--Datalist/
    |  |--apply_eyeliner/
    |  |  |--test.txt
    |  |  |--train.txt
    |  |--braid_hair/
    |  |  |--test.txt
    |  |  |--train.txt
    |  |--origami/
    |  |  |--test.txt
    |  |  |--train.txt
    |  |--scrambled_eggs/
    |  |  |--test.txt
    |  |  |--train.txt
    |  |--tie_tie/
    |  |  |--test.txt
    |  |  |--train.txt
    |--README.txt
    |--best_model/
    |  |--apply_eyeliner/
    |  |--braid_hair/
    |  |--origami/
    |  |--scrambled_eggs/
    |  |--tie_tie/
    |--dataset.py
    |--image/
    |  |--apply_eyeliner/
    |  |--braid_hair/
    |  |--origami/
    |  |--scrambled_eggs/
    |  |--tie_tie/
    |--main.py
    |--ops/
    |--opts.py
    |--resnet.py
    |--run.sh
    |--tf_model_zoo/
    |--transforms.py


code
bash run.sh 

注意事項
- Datalistディレクトリ内にある各タスクのpathはデータセットの場所によって変更して下さい．
- データセットはここからダウンロードして下さい．https://github.com/hazeld/rank-aware-attention-network
- このコードはBEST datasetを前提として作成してあります．EPICK Skills datasetを使用する場合は書き換えが必要になります．
- attention mapの可視化は評価時のみです．可視化結果はimageディレクトリの中に保存されます．
