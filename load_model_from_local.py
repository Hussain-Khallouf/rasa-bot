import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

MODEL_NAME = 'aubmindlab/bert-base-arabertv2'

pre = BertTokenizer.from_pretrained(MODEL_NAME)
classes = ['agency_contact_inquiry', 'book_flight', 'change_book_inquiry', 'flight_info_inquiry', 'flight_schedules_inquiry', 'ticket_price_inquiry', 'shipping_inquiry', 'weight_inquiry']

def predict_m(model_m, example):
    ids = pre(example, return_tensors="tf", padding='max_length', max_length=32, truncation=True)['input_ids']
    print(ids)
    pred = model_m.predict(ids)
    index = tf.math.argmax(pred[0])

    print(classes[index] + " with percentage: " + str(pred[0][index] ) + "\n")

    scores = {}
    for i in range(len(classes)):
        scores[classes[i]] = pred[0][i]

    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    for k,v in scores.items():
        print(k + " with percentage: " + "{:.2f}".format(v))


loaded_model = tf.keras.models.load_model('models/nlu-20230430-135555-largo-soldier/components/train_ArabertIntentClassifierComponent_old.CustomArabertClassifier3/train_ArabertIntentClassifierComponent_old.CustomArabertClassifier3.h5',
                                          custom_objects = {'TFBertModel': TFBertModel})
example = "تعديل الحجز من دمشق الى الشارقة"
print(example)
predict_m(loaded_model,example)

example = "هل يوجد شحن جوي من قطر الى سوريا"
print(example)
predict_m(loaded_model,example)

example = " متى تصل رحلتكم اليوم من دمشق ؟"
print(example)
predict_m(loaded_model,example)

