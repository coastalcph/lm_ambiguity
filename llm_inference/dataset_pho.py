from typing import Generator, List, Tuple
import pandas as pd
from .utils import TemplateLoader

class TsvTextDataset:
    def __init__(
        self,
        tsv_path: str,
        template_name: str,
        batch_size: int = 1,
        text_column1: str = "sentence1",
        text_column2: str = "sentence2",
        status_column: str = "ambig_status",  # New parameter for `ambig_status`
    ):
        self.file_path = tsv_path
        self.batch_size = batch_size

        self.data = pd.read_csv(tsv_path, sep="\t", quoting=3, dtype=str, na_filter=False)

        # Extract text and status columns
        self.texts = list(zip(
            self.data[text_column1].dropna(), 
            self.data[text_column2].dropna()
        ))

        
        self.statuses = (
            self.data[status_column]
            .replace('', '0')  # Replace empty strings with a default value
            .dropna()          # Drop missing values
            .astype(int)       # Convert to integers
        )
        
        self.template = TemplateLoader().load(template_name)

    def __iter__(self) -> Generator[List[Tuple[str, str]], None, None]:
        for i in range(0, len(self.texts), self.batch_size):
            yield self.texts[i : i + self.batch_size]

    def filter_indices(self, indices: List[int]) -> None:
        self.texts = [self.texts[i] for i in indices]
        self.statuses = self.statuses.iloc[indices]

