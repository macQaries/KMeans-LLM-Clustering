import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Read csv file with pandas
path = "C:/Users/USER/Documents/macQaries/bank cluster/data/wrangled_data.csv"
df = pd.read_csv(path)
df.head()

def compile_exp(x):
    text = f"""Age: {x["age"]},
               Job: {x["job"]},
               Marital: {x["marital"]},
               Education: {x["education"]},
               Default: {x["default"]},
               Balance: {x["balance"]},
               Housing: {x["housing"]},
               Loan: {x["loan"]}
            """
    return text

sentences = df.apply(lambda x: compile_exp(x), axis = 1).tolist()

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

output = model.encode(sentences = sentences, show_progress_bar = True, normalize_embeddings = True)

data_embed = pd.DataFrame(output)

data_embed

data_embed.to_csv("C:/Users/USER/Documents/macQaries/bank cluster/data/embedded_data.csv", index = False)
