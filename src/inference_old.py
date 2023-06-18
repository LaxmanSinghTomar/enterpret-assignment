
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

pl.seed_everything(42)

df = pd.read_csv("data/Enterpret Summary Generation Assignment - final_public_train.csv", encoding='latin-1')

df = df[['Text', 'Summary']]
filtered_df = df.copy()

class ReviewSummaryDataset(Dataset):
  def __init__(
      self,
      data: pd.DataFrame,
      tokenizer: T5Tokenizer,
      text_max_token_len: int = 512,
      summary_max_token_len: int = 128
    ):
    self.tokenizer = tokenizer
    self.data = data
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    text = data_row["Text"]

    text_encoding = self.tokenizer(
        text,
        max_length = self.text_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    summary_encoding = self.tokenizer(
        data_row["Summary"],
        max_length = self.summary_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    labels = summary_encoding["input_ids"]
    labels[labels == 0] = -100

    return dict(
        text=text,
        summary=data_row["Summary"],
        text_input_ids=text_encoding["input_ids"].flatten(),
        text_attention_mask=text_encoding["attention_mask"].flatten(),
        labels=labels.flatten(),
        labels_input_ids = summary_encoding["input_ids"].flatten(),
        labels_attention_mask=summary_encoding["attention_mask"].flatten()
    )

class ReviewSummaryDataModule(pl.LightningDataModule):
  
  def __init__(
      self,
      train_df: pd.DataFrame,
      test_df: pd.DataFrame,
      tokenizer: T5Tokenizer,
      batch_size: int = 8,
      text_max_token_len: int = 512,
      summary_max_token_len: int = 128,
    ):

    super().__init__()
    
    self.train_df = train_df
    self.test_df = test_df

    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len

  def setup(self, stage=None):
    self.train_dataset = ReviewSummaryDataset(
          self.train_df,
          self.tokenizer,
          self.text_max_token_len,
          self.summary_max_token_len,
      )

    self.test_dataset = ReviewSummaryDataset(
          self.test_df,
          self.tokenizer,
          self.text_max_token_len,
          self.summary_max_token_len,
      )

  def train_dataloader(self):
    return DataLoader(
          self.train_dataset,
          batch_size = self.batch_size,
          shuffle=True,
          num_workers=4
      )

  def val_dataloader(self):
    return DataLoader(
          self.test_dataset,
          batch_size = self.batch_size,
          shuffle=False,
          num_workers=4
      )

  def test_dataloader(self):
    return DataLoader(
          self.test_dataset,
          batch_size = self.batch_size,
          shuffle=False,
          num_workers=4
      )

MODEL_NAME = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

N_EPOCHS = 3
BATCH_SIZE = 8


class ReviewSummaryModel(pl.LightningModule):
 
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

    output = self.model(
        input_ids, 
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask
    )

    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels=batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]
        
    loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels=batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels
      )
    
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"]
    labels=batch["labels"]
    labels_attention_mask = batch["labels_attention_mask"]

    loss, outputs = self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels
      )
    
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr=0.0001)

model = ReviewSummaryModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="reviews-summary")

trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=N_EPOCHS,
    )

trained_model = ReviewSummaryModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")

trained_model.freeze()

def summarize(text):
  text_encoding = tokenizer(
      text,
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
        text = sample_row["Text"]
        actual_summary = sample_row["Summary"]
        model_summary = summarize(text)

        print(f"text: {text}")
        print(f"actual summary: {actual_summary}")
        print(f"model_summary: {model_summary}")

        # Ask the user if they want to continue
        continue_input = input("Do you want to continue? (yes/no): ")

        # Break the loop if the user chooses not to continue
        if continue_input.lower() == "no":
            break
