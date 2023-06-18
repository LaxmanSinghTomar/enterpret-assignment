# Import Packages

import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from dataset import ReviewSummaryDataModule
from model import ReviewSummaryModel, tokenizer
from config import BATCH_SIZE, N_EPOCHS


pl.seed_everything(42)


def train():
  df = pd.read_csv("data/final_df_analysis.csv", encoding='latin-1')
  
  df = df[['type', 'text', 'summary']]
  print(f"File's shape before: {df.shape}")
  
  filtered_df = df.dropna()
  def is_empty_string(value):
    return value.strip() == ""

  empty_string_check = filtered_df[['type', 'text', 'summary']].applymap(is_empty_string)
  filtered_df = filtered_df[~empty_string_check.any(axis=1)]
  print(f"File's shape after: {filtered_df.shape}")

  train_df, test_df = train_test_split(filtered_df, test_size=0.1)
  print(f"Train Dataset: {train_df.shape}, Test Dataset: {test_df.shape}")


  data_module = ReviewSummaryDataModule(train_df, test_df, tokenizer, BATCH_SIZE)
  model = ReviewSummaryModel()
  print(f"Data and Model is Prepared!")

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
      #gpus=1
      )
    
  print(f"Starting Training...")
  trainer.fit(model, data_module)


if __name__=="__main__":
  train()