from arabert import ArabertPreprocessor
from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.nlu import utils


@DefaultV1Recipe.register("arabert_preprocessor", is_trainable=False)
class CustomArabertPreprocessor(GraphComponent):
    name = "arabert_preprocessor"


    def __init__(self, component_config=None):
        super().__init__(component_config)
        model_name = component_config.get("model_name", "aubmindlab/bert-base-arabertv2")
        self.arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)

    def preprocess_text(self, text):
        return self.arabert_prep.preprocess(text)

    def process(self, message, **kwargs):
        text = message.text
        preprocessed_text = self.preprocess_text(text)
        message.text = preprocessed_text

    def persist(self, model_dir):
        pass

    @classmethod
    def load(cls, component_meta=None, model_dir=None, model_metadata=None, **kwargs):
        if component_meta:
            return cls(component_meta)
        else:
            utils.raise_warning(
                f"Failed to load {cls.__name__} from model archive. "
                f"No component data was found."
            )
            return cls(component_meta={})
