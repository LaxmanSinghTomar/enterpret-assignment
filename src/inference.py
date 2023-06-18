# Inference

import pandas as pd
from model import ReviewSummaryModel, tokenizer
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

pl.seed_everything(42)


trained_model = ReviewSummaryModel.load_from_checkpoint("checkpoints_updated/best-checkpoint.ckpt")
trained_model.freeze()

df = pd.read_csv("data/final_df_analysis.csv", encoding='latin-1')
  
df = df[['type', 'text', 'summary']]
print(f"File's shape before: {df.shape}")
  
filtered_df = df.dropna()
def is_empty_string(value):
  return value.strip() == ""

empty_string_check = filtered_df[['type', 'text', 'summary']].applymap(is_empty_string)
filtered_df = filtered_df[~empty_string_check.any(axis=1)]
train_df, test_df = train_test_split(filtered_df, test_size=0.1)


def summarize(text, record_type):
  text_with_record_type = f"<{record_type}> {text}"
  text_encoding = tokenizer(
      text_with_record_type,
      max_length=512,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt",
      )
  generated_ids = trained_model.model.generate(
      input_ids = text_encoding['input_ids'],
      attention_mask=text_encoding['attention_mask'],
      max_length=128,
      num_beams=2,
      repetition_penalty=2.5,
      length_penalty=1.0,
      early_stopping=True
    )

  preds = [
      tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_specs=True)
      for gen_id in generated_ids
    ]

  return "".join(preds)



if __name__ == "__main__":
    while True:
        # Ask the user to enter a number
        user_input = input("Enter a number: ")

        # Convert the user input to an integer
        index = int(user_input)

        # Use the integer as an index for df.iloc
        sample_row = filtered_df.iloc[index]

        # The rest of the code remains the same
        text = sample_row["text"]
        record_type = sample_row["type"]
        actual_summary = sample_row["summary"]


        model_summary = summarize(text, record_type=record_type)

        print(f"text: {text}")
        print(f"actual summary: {actual_summary}")
        print(f"model_summary: {model_summary}")

        # Ask the user if they want to continue
        continue_input = input("Do you want to continue? (yes/no): ")

        # Break the loop if the user chooses not to continue
        if continue_input.lower() == "no":
            break