import pandas as pd
from arabert.preprocess import ArabertPreprocessor
from transformers import TFBertModel, BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split
import tensorflow as tf

MAX_LENGHT = 32
MODEL_NAME = 'aubmindlab/bert-base-arabertv2'
arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME, keep_emojis=False)
config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=False)
pre = BertTokenizer.from_pretrained(MODEL_NAME)
bert = TFBertModel.from_pretrained(MODEL_NAME, config=config, from_pt=True, trainable=True)

df = pd.read_csv("filtered_data.csv")

df['text'] = df['sentence'].apply(lambda x: arabert_prep.preprocess(x))


X_train, X_valid, y_train, y_valid = train_test_split(df["text"], df["intent"],
                                                      stratify=df["intent"],
                                                      test_size=0.1)
classes = df.intent.unique().tolist()

y_train = y_train.apply(lambda x: classes.index(x))
y_valid = y_valid.apply(lambda x: classes.index(x))

X_train = pre(X_train.tolist(), return_tensors="tf", padding='max_length', max_length=MAX_LENGHT, truncation=True)
X_valid = pre(X_valid.tolist(), return_tensors="tf", padding='max_length', max_length=MAX_LENGHT, truncation=True)


input_text = tf.keras.Input(shape=(MAX_LENGHT,), dtype=tf.int32, name="input")

bert_output = bert(input_text)

net = tf.keras.layers.Dropout(0.5, name='DropOut1')(bert_output['pooler_output'])
net = tf.keras.layers.Dense(units=768, activation='tanh', name='classifier')(net)
net = tf.keras.layers.Dropout(0.5, name='DropOut2')(net)
net = tf.keras.layers.Dense(units=len(classes), activation='softmax', name='output')(net)

model = tf.keras.Model(input_text, net)

loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
              loss=loss,
              metrics=metrics
            )


history = model.fit(
                    x = X_train['input_ids'],
                    y = y_train,
                    validation_data = (X_valid['input_ids'], y_valid),
                    epochs=10,
                   shuffle=True
                   )

