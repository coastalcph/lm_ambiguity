from typing import Generator, List
import pandas as pd
from .utils import TemplateLoader

class TsvTextDataset:
    def __init__(
        self,
        tsv_path: str,
        template_name: str,
        batch_size: int = 1,
        text_column: str = "sentence",
        status_column: str = "ambig_status",  # New parameter for `ambig_status`
    ):
        self.file_path = tsv_path
        self.batch_size = batch_size

        self.data = pd.read_csv(tsv_path, sep="\t", quoting=3, dtype=str, na_filter=False)

        # Extract text and status columns
        self.texts = self.data[text_column].dropna()
        self.statuses = self.data[status_column].dropna().astype(int)

        self.template = TemplateLoader().load(template_name)

    def __iter__(self) -> Generator[List[str], None, None]:
        for i in range(0, len(self.texts), self.batch_size):
            yield [t for t in self.texts[i : i + self.batch_size] if t.strip()]

    def filter_indices(self, indices: List[int]) -> None:
        self.texts = self.texts.iloc[indices]
        self.statuses = self.statuses.iloc[indices]

