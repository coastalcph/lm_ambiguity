import math
from typing import Generator, List

import pandas as pd
import csv
from .utils import TemplateLoader


class TsvTextDataset:
    def __init__(
        self,
        tsv_path: str,
        template_name: str,
        batch_size: int = 1,
        text_column: str = "text",
    ):
        self.file_path = tsv_path
        self.batch_size = batch_size

        self.data = pd.read_csv(tsv_path, sep="\t", quoting=3, dtype=str, na_filter=False)[text_column].dropna()

        #self.data = pd.read_csv(tsv_path, sep='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\')[text_column].dropna()

        self.template = TemplateLoader().load(template_name)

    def __iter__(self) -> Generator[List[str], None, None]:
        for i in range(0, len(self.data), self.batch_size):
            yield [t for t in self.data[i : i + self.batch_size] if t.strip()]

    def filter_indices(self, indices: List[int]) -> None:
        self.data = self.data[indices]

    @property
    def num_batches(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def __len__(self) -> int:
        return self.num_batches
