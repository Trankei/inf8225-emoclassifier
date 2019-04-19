import os
from fastText import train_supervised

train_data = os.path.join("../../processed_data", 'text.train')
valid_data = os.path.join("../../processed_data", 'text.valid')

model = train_supervised(
    input=train_data, epoch=200, lr=0.1, wordNgrams=3, verbose=2, minCount=1, dim=200, neg=3
)
print(*model.test(valid_data))
model.save_model("../../models/text_classfier.bin")