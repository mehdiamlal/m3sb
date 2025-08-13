import torch
from typing import Any
import copy
from transformers import AutoModel

def get_model_parameters(model: torch.nn.Module) -> str:
    """Generates a summary string of a model's total and trainable parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to inspect.

    Returns:
        str: A formatted, multi-line string detailing the total and
            trainable parameters.
    """

    model_total_params = sum(p.numel() for p in model.parameters())
    model_trainalble_params = sum(p.numel() for p in model.parameters() 
                                  if p.requires_grad)
    return (
        f"Total parameters: {model_total_params:,}"
        f"\nTrainable parameters: {model_trainalble_params:,}"
    )

def map_key_names(dict1: dict[str, Any], 
                  dict2: dict[str, Any]) -> dict[str, str]:
    """Creates a one-to-one mapping of keys between two dictionaries based 
    on order.

    Args:
        dict1 (dict): The first dictionary. Its keys will become the keys of the 
            returned map.
        dict2 (dict): The second dictionary. Its keys will become the values
            of the returned map.

    Returns:
        dict: A dictionary that maps the keys of `dict1` to the keys of `dict2`.

    Raises:
        AssertionError: If the input dictionaries do not have the same
            number of keys.
    """
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    key_map = {}

    assert len(keys1) == len(keys2)

    for k1, k2 in zip(keys1, keys2):
        key_map[k1] = k2

    return key_map


def get_task_vector(base: dict[str, torch.Tensor], 
                    fine_tuned: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  ''''
  Computes the task vector by subtracting the base model parameters from the
  fine-tuned model parameters.

  Args:
      base (dict[str, torch.Tensor]): The base model parameters.
      fine_tuned (dict[str, torch.Tensor]): The fine-tuned model parameters.

  Returns:
      dict[str, torch.Tensor]: A dictionary representing the task vector,
          where each key corresponds to a parameter and the value is the
          difference between the fine-tuned and base parameters.

  Raises:
      AssertionError: If the input dictionaries do not have the same
          number of keys.'''
  assert len(base) == len(fine_tuned)

  task_vector = copy.deepcopy(base)
  key_map = map_key_names(base, fine_tuned)

  for k in base.keys():
    task_vector[k] = fine_tuned[key_map[k]] - base[k]

  return task_vector

def load_model(model_checkpoint: str, model_class: type = AutoModel) -> torch.nn.Module:
    """Loads a model from a specified checkpoint.

    Args:
        model_checkpoint (str): The path or identifier of the model checkpoint.
        model_class (type, optional): The class of the model to load. 
            Defaults to AutoModel.

    Returns:
        torch.nn.Module: The loaded model.
    """
    return model_class.from_pretrained(model_checkpoint)

def similar_architecture(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    """Compares the architecture of two models.

    Args:
        model1 (torch.nn.Module): The first model to compare.
        model2 (torch.nn.Module): The second model to compare.

    Returns:
        bool: True if the models have similar architectures, False otherwise.
    """
    key_map = map_key_names(model1.state_dict(), model2.state_dict())
    for k, v in model1.state_dict().items():
        if v.shape != model2.state_dict()[key_map[k]].shape:
            return False

    return True