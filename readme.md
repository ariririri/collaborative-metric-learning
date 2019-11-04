# Collbaorative Metric Learning


This is pytorch implementation  of http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf


DataFlow

1. preprocess
2. train
3. predict


## preprocess
1. 以下を作成
  - user_item_matrix
  - item_feature
2. user_itemからposiとnegをサンプリング
  - featureは該当するitemごとに設定する?


Datasetの段階で


- posi_indexの作成
iterateの仕方
- batch_size: バッチ数
- n_negative: negativeとの比率
- posi_indexからバッチ数取得
- n_negativeかつ

Samplerで
- posi_index
- nega_index
- feature_index(まず後回し)
[posi, ]


## Model
- user_embed
- pos_item_embed
- neg_item_embed

## Loss
- userとposの距離
- posとnegの距離
- rank loss

        loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0,

                                   name="pair_loss")