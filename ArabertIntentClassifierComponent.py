import logging
from typing import Any, Text, Dict, List, Type

import numpy as np
import tensorflow as tf
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from scipy.sparse import csr_matrix
from transformers import TFBertModel, BertTokenizer, BertConfig

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class CustomArabertClassifier(IntentClassifier, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Featurizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn", "tensorflow"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "epochs": 10,
            "class_num": 8,
            "model_name": 'aubmindlab/bert-base-arabertv2'
        }

    def __init__(
            self,
            config: Dict[Text, Any],
            name: Text,
            model_storage: ModelStorage,
            resource: Resource,
    ) -> None:
        self.name = name
        self.CLASS_NUM = config.get("class_num", 8)
        self.EPOCHS = config.get("epochs", 10)
        self.MODEL_NAME = config.get("model_name", 'aubmindlab/bert-base-arabertv2')

        self._model_storage = model_storage
        self._resource = resource

        self.MAX_LENGHT = 35

        self.intents = [
            'booking_availablity',
            'booking_methods',
            'change_book_cost_inquiry',
            'change_book_inquiry',
            'flight_schedules_inquiry',
            'ticket_price_inquiry',
            'book_flight',
            'flight_info_inquiry'
        ]

        # tf.random.set_seed(42)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        config = BertConfig.from_pretrained(self.MODEL_NAME, output_hidden_states=False)
        bert = TFBertModel.from_pretrained(self.MODEL_NAME, config=config, from_pt=True, trainable=True)
        input_text = tf.keras.Input(shape=(self.MAX_LENGHT,), dtype=tf.int32, name="input")

        bert_output = bert(input_text)

        net = tf.keras.layers.Dropout(0.5, name='DropOut1')(bert_output['pooler_output'])
        net = tf.keras.layers.Dense(units=768, activation='tanh', name='classifier')(net)
        net = tf.keras.layers.Dropout(0.5, name='DropOut2')(net)
        net = tf.keras.layers.Dense(units=self.CLASS_NUM, activation='softmax', name='output')(net)

        model = tf.keras.Model(input_text, net)

        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
                      loss=loss,
                      metrics=metrics
                      )
        self.model = model

    def _get_bert_x(self, messages: List[Message]) -> csr_matrix:
        msgs = [e.get(TEXT) for e in messages]
        msgs = self.bert_tokenizer(
            msgs,
            return_tensors="tf",
            padding='max_length',
            max_length=self.MAX_LENGHT,
            truncation=True
        )
        return msgs['input_ids']

    def _get_bert_training_matrix(self, training_data: TrainingData):
        X = self._get_bert_x(training_data.training_examples)
        y = [self.intents.index(e.get(INTENT)) for e in training_data.training_examples]
        return np.array(X), np.array(y)

    def train(self, training_data: TrainingData) -> Resource:
        bert_X, bert_y = self._get_bert_training_matrix(training_data)
        self.model.fit(x=bert_X, y=bert_y, epochs=self.EPOCHS, shuffle=True)

        self.persist()

        return self._resource

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)

    def process(self, messages: List[Message]) -> List[Message]:
        for idx, message in enumerate(messages):
            ids = self.bert_tokenizer(
                message.get(TEXT),
                return_tensors="tf",
                padding='max_length',
                max_length=self.MAX_LENGHT,
                truncation=True
            )['input_ids']
            pred = self.model.predict(ids)
            index = tf.math.argmax(pred[0])
            scores = {}
            for i in range(len(self.intents)):
                scores[self.intents[i]] = pred[0][i]
            scores = {k: v for k, v in
                      sorted(scores.items(), key=lambda item: item[1], reverse=True)[:LABEL_RANKING_LENGTH]}

            intent = {"name": self.intents[index], "confidence": float(pred[0][index])}

            intent_ranking = [
                {"name": k, "confidence": float(v)} for k, v in scores.items()
            ]

            message.set("intent", intent, add_to_output=True)
            message.set("intent_ranking", intent_ranking, add_to_output=True)

        return messages

    def persist(self) -> None:
        with self._model_storage.write_to(self._resource) as model_dir:
            self.model.save(f'{model_dir}/{self.name}.h5', save_format='h5')

    @classmethod
    def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:

        with model_storage.read_from(resource) as model_dir:
            loaded_model = tf.keras.models.load_model(model_dir / f"{resource.name}.h5",
                                                      custom_objects={'TFBertModel': TFBertModel})

            # loaded_model = tf.keras.models.load_model(
            #     'models/9_acc.h5',
            #     compile=False,
            #     custom_objects={'TFBertModel': TFBertModel})
            # loss = tf.keras.losses.SparseCategoricalCrossentropy()
            # metrics = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
            #
            # loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
            #                      loss=loss,
            #                      metrics=metrics
            #                      )

            component = cls(
                config, execution_context.node_name, model_storage, resource
            )

            component.model = loaded_model
            component.test_val = "hussain"
        return component

    def parse_msg(self, model, example):
        ids = self.bert_tokenizer(example, return_tensors="tf", padding='max_length', max_length=32, truncation=True)[
            'input_ids']
        pred = model.predict(ids)
        index = tf.math.argmax(pred[0])

        print(self.intents[index] + " with percentage: " + str(pred[0][index]) + "\n")

        scores = {}
        for i in range(len(self.intents)):
            scores[self.intents[i]] = pred[0][i]

        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        for k, v in scores.items():
            print(k + " with percentage: " + "{:.2f}".format(v))

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
