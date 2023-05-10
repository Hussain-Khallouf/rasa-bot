import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

MODEL_NAME = 'aubmindlab/bert-base-arabertv02-twitter'

pre = BertTokenizer.from_pretrained(MODEL_NAME)

classes = [
    'booking_availablity',
    'booking_methods',
    'change_book_cost_inquiry',
    'change_book_inquiry',
    'flight_schedules_inquiry',
    'ticket_price_inquiry',
    'book_flight',
    'flight_info_inquiry'
]


def predict_m(model_m, example):
    ids = pre(example, return_tensors="tf", padding='max_length', max_length=35, truncation=True)['input_ids']
    print(ids)
    pred = model_m.predict(ids)
    index = tf.math.argmax(pred[0])
    print(index)
    print(classes[index] + " with percentage: " + str(pred[0][index]) + "\n")

    scores = {}
    for i in range(len(classes)):
        scores[classes[i]] = pred[0][i]

    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    for k, v in scores.items():
        print(k + " with percentage: " + "{:.2f}".format(v))


loaded_model = tf.keras.models.load_model(
    'models/9_acc.h5',
    compile=False,
    custom_objects={'TFBertModel': TFBertModel})
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
              loss=loss,
              metrics=metrics
            )


example = "تعديل الحجز من دمشق الى الشارقة"
print(example)
predict_m(loaded_model, example)

example = "هل يوجد شحن جوي من قطر الى سوريا"
print(example)
predict_m(loaded_model, example)

example = " متى تصل رحلتكم اليوم من دمشق ؟"
print(example)
predict_m(loaded_model, example)
