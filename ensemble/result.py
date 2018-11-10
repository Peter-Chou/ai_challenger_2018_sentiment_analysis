import multiprocessing
import os
import random
from collections import Counter
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(10)
tables = []
WORKERS = 8


for file in os.listdir():
  if file.split(sep='.')[-1] == "csv":
    df = pd.read_csv(file)
    tables.append(df)


df_new = (pd.concat(tables)
          .groupby(level=0)
          .apply(
    lambda g: pd.Series({i: np.hstack(g[i].values) for i in df.columns})))
df_new.iloc[:, :2] = df.iloc[:, :2]


def vote_func(data):
  (data, i) = data
  for row in range(data.shape[0]):
    for col in range(2, data.shape[1]):
      counter = Counter(data.iloc[row, col])
      max_count = counter.most_common(1)[0][1]
      candidates = []
      for res, count in counter.items():
        if count == max_count:
          candidates.append(res)
      data.iloc[row, col] = random.choice(candidates)
  return i, data


if __name__ == "__main__":
  start_time = time()
  pool = multiprocessing.Pool(processes=WORKERS)
  result = pool.map(vote_func, [(d, i) for i, d in enumerate(np.array_split(df_new, WORKERS))])
  pool.close()
  result = sorted(result, key=lambda x: x[0])
  df_ensemble = pd.concat([i[1] for i in result])
  print("--- {:0.2f} seconds ---".format(time() - start_time))

  print(df_ensemble.iloc[0, 2:])
  df_ensemble.to_csv("ensembles.csv", index=False)
