import transformers
model = transformers.AutoModel.from_pretrained("colorfulscoop/gpt2-small-ja", revision="20210820.1.0")
tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja", revision="20210820.1.0")
import os
path = "./model/gpt2-small-jae"

os.makedirs(path , exist_ok=True)

tokenizer.save_pretrained(path)
model.save_pretrained(path)