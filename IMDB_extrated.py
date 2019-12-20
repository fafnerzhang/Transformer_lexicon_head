"""import pandas as pd
def sentiment_to_label(sentiment):
    return (sentiment["sentiment"]).replace("negative","__label__1").replace("positive",'__label__0')
imdb = pd.read_csv("IMDB Dataset.csv")
imdb = imdb[['sentiment','review']]
imdb['sentiment'] = imdb.apply(sentiment_to_label,axis=1)
imdb.to_csv("IMDB_extracted.csv",header=False,index=False)"""

import pandas as pd
def label_transform(label):
    return f'__label__{label["label"]}'
def text_transform(text):
    return text['text'].replace("'","").replace("[","").replace("]","").replace(",","")

train_data = pd.read_csv("test_dataset.csv")

train_data['label'] = train_data.apply(label_transform,axis=1)
train_data['text'] = train_data.apply(text_transform,axis=1)
print(train_data)
train_data.to_csv("test_data_extracted.csv",header=False,index=False)