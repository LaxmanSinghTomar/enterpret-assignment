# Dataset

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5TokenizerFast as T5Tokenizer
)

# Custom Dataset class for handling review summary data
class ReviewSummaryDataset(Dataset):
  def __init__(
      self,
      data: pd.DataFrame,
      tokenizer: T5Tokenizer,
      record_types: pd.Series,
      text_max_token_len: int = 512,
      summary_max_token_len: int = 128,
    ):
    """
    Initializes the ReviewSummaryDataset with the given data, tokenizer, and token length limits.

    Args:
      data (pd.DataFrame): The input data as a DataFrame.
      tokenizer (T5Tokenizer): The tokenizer to use for encoding the text and summary.
      record_types (pd.Series): The record types for each data row.
      text_max_token_len (int, optional): The maximum token length for the text. Defaults to 512.
      summary_max_token_len (int, optional): The maximum token length for the summary. Defaults to 128.
    """
    self.tokenizer = tokenizer
    self.data = data
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len
    self.record_types = record_types
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    text = data_row["text"]
    record_type = self.record_types.iloc[index]
    record_type_token = f"<{record_type}>"
    text_with_record_type = f"{record_type_token} {text}"

    # Encode the text with the tokenizer
    text_encoding = self.tokenizer(
        text_with_record_type,
        max_length = self.text_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # Encode the summary with the tokenizer
    summary_encoding = self.tokenizer(
        data_row["summary"],
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
        summary=data_row["summary"],
        text_input_ids=text_encoding["input_ids"].flatten(),
        text_attention_mask=text_encoding["attention_mask"].flatten(),
        labels=labels.flatten(),
        labels_input_ids = summary_encoding["input_ids"].flatten(),
        labels_attention_mask=summary_encoding["attention_mask"].flatten()
    )

# Custom DataModule class for handling review summary data
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
    """
    Initializes the ReviewSummaryDataModule with the given train and test data, tokenizer, and token length limits.

    Args:
      train_df (pd.DataFrame): The training data as a DataFrame.
      test_df (pd.DataFrame): The testing data as a DataFrame.
      tokenizer (T5Tokenizer): The tokenizer to use for encoding the text and summary.
      batch_size (int, optional): The batch size for the DataLoader. Defaults to 8.
      text_max_token_len (int, optional): The maximum token length for the text. Defaults to 512.
      summary_max_token_len (int, optional): The maximum token length for the summary. Defaults to 128.
    """
    super().__init__()
    
    self.train_df = train_df
    self.test_df = test_df

    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len

  def setup(self, stage=None):
    # Create the train and test datasets
    self.train_dataset = ReviewSummaryDataset(
          self.train_df,
          self.tokenizer,
          self.train_df['type'],
          self.text_max_token_len,
          self.summary_max_token_len,
      )

    self.test_dataset = ReviewSummaryDataset(
          self.test_df,
          self.tokenizer,
          self.test_df['type'],
          self.text_max_token_len,
          self.summary_max_token_len,
      )

  def train_dataloader(self):
    # Create the train DataLoader
    return DataLoader(
          self.train_dataset,
          batch_size = self.batch_size,
          shuffle=True,
          num_workers=4
      )

  def val_dataloader(self):
    # Create the validation DataLoader
    return DataLoader(
          self.test_dataset,
          batch_size = self.batch_size,
          shuffle=False,
          num_workers=4
      )

  def test_dataloader(self):
    # Create the test DataLoader
    return DataLoader(
          self.test_dataset,
          batch_size = self.batch_size,
          shuffle=False,
          num_workers=4
      )
