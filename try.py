import logging
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
import pandas as pd

from llm_inference.dataset import TsvTextDataset  # Modify this with the actual import path
from llm_inference.utils import setup_config, setup_logging

logger = logging.getLogger(__name__)

def add_status_to_results(output_path: str, dataset: TsvTextDataset):
    # Load the output TSV
    df_results = pd.read_csv(output_path, sep="\t")
    print("opened output")
    # Ensure the lengths match
    if len(df_results) != len(dataset.statuses):
        raise ValueError("Mismatch in the number of rows between results and dataset statuses.")

    # Add the ambig_status column
    df_results["true_ambig_status"] = dataset.statuses.tolist()
    print("merged")
    print(head(df_results))
    # Save the updated TSV
    df_results.to_csv(output_path, sep="\t", index=False)

    # Calculate accuracy
    if "ambig_status" not in df_results.columns:
        raise KeyError("The results file does not contain the 'binary' column.")
    accuracy = accuracy_score(df_results["true_ambig_status"], df_results["ambig_status"])
    print(f"Accuracy: {accuracy:.2%}")

    return accuracy

@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):
    # Set up logging and configuration
    setup_logging()
    setup_config(config)

    dataset = TsvTextDataset(
        tsv_path=config.data.tsv_path,
        template_name=config.data.template_name,
        batch_size=config.data.batch_size,
        text_column="sentence",          # Ensure this matches the column name in your dataset
        status_column="ambig_status" # Ensure this matches the column name in your dataset
    )

    # Add ambig_status to results and compute accuracy

    # Add ambig_status to the results and calculate accuracy
    output_path = config.output_path
    acc = add_status_to_results(output_path, dataset)
    print(acc)

if __name__ == "__main__":
    main()
