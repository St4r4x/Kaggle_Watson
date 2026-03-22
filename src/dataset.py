import pandas as pd
from torch.utils.data import Dataset


LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_train_data(data_dir: str, language_filter: str = "english") -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/train.csv")
    if language_filter != "all":
        df = df[df["language"].str.lower() == language_filter.lower()].reset_index(
            drop=True
        )
    return df


def load_test_data(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/test.csv")


class NLIDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, tokenizer, max_length: int = 256, is_test: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["premise"],
            row["hypothesis"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if not self.is_test:
            item["labels"] = int(row["label"])
        return item
