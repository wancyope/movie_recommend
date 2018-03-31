# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#ピアソンの相関係数を計算する関数
def sim_pearson(x,y):
    x_norm = x - np.mean(x)
    y_norm = y - np.mean(y)
    num = np.dot(x_norm.T,y_norm)
    den = np.sqrt(np.dot(x_norm.T,x_norm) * np.dot(y_norm.T,y_norm))
    if den == 0: return 0
    return (num / den)

#レーティングデータを読み込みレーティング行列を作成
df = pd.read_csv('u.data', sep = '\t', names = ['user_id','item_id', 'rating', 'timestamp'])
shape = (df.max().ix['user_id'] + 1, df.max().ix['item_id'] + 1)
R = np.zeros(shape) 
for i in df.index:
    row = df.ix[i]
    R[row['user_id'], row['item_id']] = row['rating']

#映画のタイトルをロード    
movies = {}
for line in open("u.item"):
    (id,title)=line.split("|")[0:2]
    movies[id]=title
    
#対象のユーザと他のすべてのユーザとの類似度を計算し、上位30人を取りだす
person = 100
scores = [(sim_pearson(R[person],R[other]),other) for other in range(len(R)) if other != person]
scores.sort()
scores.reverse()
knn = [scores[i][1] for i in range(30)]

#上位30人のレーティングに類似度を掛けて重み付けし、それを類似度の合計で割りランキングリストを作成
weighted_ratings = {}
simSums = {}
for i in knn:
    for j in range(len(R[0])):
        if R[person,j] == 0:
            weighted_ratings.setdefault(j,0)
            weighted_ratings[j] += (R[i,j] * scores[i][0])
            simSums.setdefault(j,0)
            simSums[j] += scores[i][0]
rankings=[(weighted_rating / simSums[movie],movie) for movie,weighted_rating in weighted_ratings.items()]
rankings.sort()
rankings.reverse()

#推薦する映画の上位10を表示
for i in range(10):
    print(movies[str(rankings[i][1] + 1)] + ":" + str(rankings[i][0]))





