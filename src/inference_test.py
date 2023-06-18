from inference import test_df
from inference_old import summarize

model_summaries = []

from tqdm import tqdm

for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    type = row['type']
    text = row['text']
    summary = row['summary']
    output = summarize(text=text)
    model_summaries.append(output)


test_df['model_summaries'] = model_summaries
test_df.to_csv("inference_test_old.csv", index=False)

