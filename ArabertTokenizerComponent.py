from __future__ import annotations
from __future__ import annotations

from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
from transformers import BertTokenizer


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class CustomAraBertTokenizer(Tokenizer):
    """ A custom Rasa NLU component that preprocesses text using BertTokenizer. """

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """The languages that are not supported."""
        return ["zh", "ja", "th"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            "intent_tokenization_flag": False,
            # This *must* be added due to the parent class.
            "intent_split_symbol": "_",

            "model_name": "aubmindlab/bert-base-arabertv2",

            "max_length": 32,
        }

    def __init__(self, component_config: Dict[Text, Any]) -> None:
        super().__init__(component_config)
        model_name = component_config.get("model_name", "aubmindlab/bert-base-arabertv2")
        self.max_length = component_config.get("max_length", 32)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> CustomAraBertTokenizer:
        return cls(config)

    def parse_string(self, s):
        return "".join([c for c in s if (((c == " ") or str.isalnum(c)) and c not in "UNK")])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = self.parse_string(message.get(attribute))
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.parse_string(token) for token in tokens]

        if not tokens:
            tokens = [text]
        tokens = self._convert_words_to_tokens(tokens, text)
        pattern = self._apply_token_pattern(tokens)
        return pattern

