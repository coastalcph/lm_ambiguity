import logging
from pathlib import Path

import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from llm_inference.engine import Engine
from llm_inference.schema import schema_registry
from llm_inference.utils import TsvWriter, setup_config, setup_logging
from llm_inference.dataset import TsvTextDataset  # Modify this with the actual import path

from sklearn.metrics import accuracy_score
import pandas as pd

logger = logging.getLogger(__name__)

def add_status_to_results(output_path: str, dataset: TsvTextDataset):
    df_results = pd.read_csv(output_path, sep="\t")
    # Ensure the lengths match
    if len(df_results) != len(dataset.statuses):
        raise ValueError("Mismatch in the number of rows between results and dataset statuses.")

    # Add the ambig_status column
    df_results["ambig_status"] = dataset.statuses.tolist()

    #df_results["is_ambiguous"] = df_results["is_ambiguous"].astype(int)  # Convert True/False to 1/0
    df_results["is_ambiguous"] = df_results["is_ambiguous"].apply(lambda x: 1 if x in ["yes", "right", "true", True] else 0)
    df_results["is_ambiguous"] = df_results["is_ambiguous"].astype(int)  # Convert True/False to 1/0

    # Save the updated DataFrame to file
    df_results.to_csv(output_path, sep="\t", index=False)

    true_count = (df_results["is_ambiguous"] == 1).sum()
    false_count = (df_results["is_ambiguous"] == 0).sum()

    # Calculate accuracy
    accuracy = accuracy_score(df_results["ambig_status"], df_results["is_ambiguous"])
    print(f"Accuracy: {accuracy:.2%}")
    print(f"True Count: {true_count}")
    print(f"False Count: {false_count}")

    return accuracy, true_count, false_count

@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):
    
    setup_logging()
    setup_config(config)

    engine = Engine(
        llm=instantiate(config.model),
        sampling_params=instantiate(config.generation),
        schema=schema_registry.get(config.schema_name),
        dataset=instantiate(config.data),
    )

    
    dataset = TsvTextDataset(
        tsv_path=config.data.tsv_path,
        template_name=config.data.template_name,
        batch_size=config.data.batch_size,
        text_column="sentence",
        #text_column2="sentence2",# Ensure this matches the column name in your dataset
        status_column="ambig_status" # Ensure this matches the column name in your dataset
    )

    model_name = config.model.model.split("/")[-1]
    #run_name = config.data.template_name  # Use schema name for the run name
    run_name = "type"
    template_name = config.data.template_name.split("/")[-1]
    data_name = config.data.tsv_path.split("/")[-1].split(".")[0]
    # Sanitize config for wandb
    sanitized_config = OmegaConf.to_container(config, resolve=True)

    # Initialize wandb with the run name and sanitized config
    wandb.init(
        project=f"{data_name}_{model_name}",
        name=f"{template_name}",  # Name the run after the schema name
        config=sanitized_config
    )
    
    with TsvWriter(Path(config.output_path)) as writer:
        for i, result in enumerate(engine()):
            if config.enable_logprobs:
                json_completion, log_probs = result
                if i == 0:
                    writer.writerow(json_completion.keys_as_list())
                writer.writerow(json_completion.values_as_list())
                print(f"{log_probs[:5] = }")
            else:
                json_completion = result
                if i == 0:
                    writer.writerow(json_completion.keys_as_list())
                writer.writerow(json_completion.values_as_list())

    output_path = config.output_path
    acc,true,false = add_status_to_results(output_path, dataset)
    print(acc,true,false)

    wandb.log({"accuracy": acc})
    wandb.log({"true_count": true})
    wandb.log({"false_count": false})
    wandb.finish()

if __name__ == "__main__":
    main()
