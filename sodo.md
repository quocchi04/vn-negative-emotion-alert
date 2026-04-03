### Sơ đồ luồn chức năng
~~~

data/raw/*.csv
   |
   v
preprocessing.py
   |
   v
data/processed/*.csv
   |
   v
dataset.py
   |
   v
tokenized_dataset
   |
   v
train.py  ---> trainer_utils.py
   |
   v
models/phobert_final/
   |
   v
inference.py
   |
   v
app.py / predict.py
        |
        v
     utils.py

~~~