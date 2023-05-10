import pandas as pd

agreed_data = pd.read_csv("agreed_data.csv")

intents = agreed_data.intent.unique()
with open("data/train_nlu.yml", "w") as nlu:
    nlu.write('version: "3.1"')
    nlu.write("\n")
    nlu.write("nlu:")
    for intent in intents:
        nlu.write("\n")
        nlu.write(f"- intent: {intent}\n")
        nlu.write("  examples: |\n")
        for sentence in agreed_data["sentence"][agreed_data.intent == intent].tolist():
            nlu.write(f"    - {sentence}\n")
