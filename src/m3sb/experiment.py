from typing import Any
import torch
from m3sb.merging import barycentric_merge, linear_merge, pairwise_slerp_merge
from m3sb.merging import build_merged_image_classifier
from m3sb.utils import load_model
from m3sb.data import get_data_loader
from m3sb.eval import evaluate_model
import pandas as pd

class Experiment:
    """Helps automating the experiments.
    """

    def __init__(
        self,
        name: str,
        model_checkpoints: list[str],
        datasets_config: list[dict[str, Any]],
        merge_configs: dict[str, dict[str, Any]],
        base_model_checkpoint: str 
    ):
        
        self.name = name
        self.model_checkpoints = model_checkpoints
        self.datasets_config = datasets_config
        self.merge_configs = merge_configs
        self.base_model_checkpoint = base_model_checkpoint

        self.results = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #map strings to actual functions
        self.merge_functions = {
            "barycenter": barycentric_merge,
            "linear": linear_merge,
            "pairwise_slerp": pairwise_slerp_merge
        }

    def run(self):
        print(f"Loading {len(self.model_checkpoints)} fine-tuned models.")
        fine_tuned_state_dicts = [
            load_model(checkpoint).state_dict() for checkpoint in self.model_checkpoints
        ]

        #iterating through each merge method
        for method_name, config in self.merge_configs.items():
            print(f"Merging with {method_name}.")

            if method_name not in self.merge_functions:
                print(f"WARNING: Merge method '{method_name}' not supported.")
                continue
            merge_function = self.merge_functions[method_name]

            #performing the merge on all the fine-tuned models' bodies
            merged_body_state_dict = merge_function(fine_tuned_state_dicts,
                                                    **config)
            
            for i, dataset_config in enumerate(self.datasets_config):
                #the assumption is that the order of the model checkpoints 
                #is the same as the dataset configs
                task_name = dataset_config["name"]
                donor_model_checkpoint = self.model_checkpoints[i]

                print("Evaluating on", task_name)

                #attaching the classification head to merged body and getting
                #approprate data loader
                eval_model = build_merged_image_classifier(
                    self.base_model_checkpoint,
                    donor_model_checkpoint,
                    merged_body_state_dict
                )

                eval_loader = get_data_loader(
                    processor_checkpoint=self.base_model_checkpoint,
                    **dataset_config
                )

                eval_output = evaluate_model(eval_model, eval_loader)

                self.results.append({
                    "experiment": self.name,
                    "merge_method": method_name,
                    "dataset": task_name,
                    **eval_output["metrics"]
                })

                del eval_model
                del eval_loader

        print(f"Experiment {self.name} completed.")

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)