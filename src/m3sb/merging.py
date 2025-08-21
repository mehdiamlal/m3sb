import copy
from m3sb.geometry import slerp, barycenter, weighted_average
from m3sb.utils import map_key_names
import torch
from transformers import AutoModelForImageClassification

def slerp_merge(interpolation_factor: float, 
                       params1: dict[str, torch.Tensor], 
                       params2: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merges two models applying SLERP.

    Args:
        interpolation_factor (float): The factor by which to interpolate
            between the two model's parameters. Should be between 0 and 1.
        params1 (dict[str, torch.Tensor]): The first model's parameters.
        params2 (dict[str, torch.Tensor]): The second model's parameters.

    Returns:
        dict[str, torch.Tensor]: The newly merged state dictionary.

    Raises:
        AssertionError: If the input dictionaries do not have the same
            number of keys or if the interpolation factor is not in the range [0, 1].
    """
    assert 0 <= interpolation_factor <= 1
    assert len(params1) == len(params2)

    merged_params = copy.deepcopy(params1)
    key_map = map_key_names(params1, params2)

    for k, v in params1.items():
        merged_params[k] = slerp(interpolation_factor, v, params2[key_map[k]])

    return merged_params

def pairwise_slerp_merge(parameters: list[dict[str, torch.Tensor]],
                         weights: list[float]) -> dict[str, torch.Tensor]:
    """Merges a list of models' parameters using pairwise SLERP with specified weights.
    
    Args:
        parameters (list[dict[str, torch.Tensor]]): 
            A list of state dictionaries.
        weights (list[float]): 
            A list of weights corresponding to each model's parameters, used to 
            determine interpolation factors.
    
    Returns:
        dict[str, torch.Tensor]: 
            A dictionary representing the merged state dictionary.
    
    Raises:
        ValueError: If the lengths of `parameters` and `weights` do not match.
    """

    merged_params = copy.deepcopy(parameters[0])
    cumulative_weight = weights[0]

    for i in range(1, len(parameters)):
        new_params = parameters[i]
        new_model_weight = weights[i]

        #correct interpolation factor
        interp_factor = new_model_weight / (cumulative_weight + new_model_weight)

        merged_params = slerp_merge(interp_factor, merged_params, 
                                         new_params)
        
        cumulative_weight += new_model_weight

    return merged_params



def barycentric_merge(parameters: list[dict[str, torch.Tensor]], 
                      weights: list[float], iterations: int = 20, 
                      threshold: float = 1e-5) -> dict[str, torch.Tensor]:
    
    """Merges multiple models' parameters using a barycentric approach.
    
    Args:
        parameters (list[dict[str, torch.Tensor]]): 
            A list of model's state dictionaries.
        weights (list[float]): 
            A list of weights corresponding to each state dictionary, used for 
            barycentric averaging.
        iterations (int, optional): 
            Number of iterations for the barycenter computation. Default is 20.
        threshold (float, optional): 
            Convergence threshold for the barycenter computation. Default is 1e-5.
    
    Returns:
        dict[str, torch.Tensor]: 
            A merged state dictionary.
    
    Notes:
        - Non-tensor parameters are skipped and not merged.
        - Assumes all state dictionaries have matching keys and compatible tensor 
        shapes.
    """

    merged_params = copy.deepcopy(parameters[0])
    
    for key in merged_params.keys():
        #skip non-tensor parameters 
        if not isinstance(merged_params[key], torch.Tensor):
            continue

        tensors_to_merge = [tv[key] for tv in parameters]
        merged_tensor = barycenter(tensors_to_merge, weights, iterations,
                                       threshold)
        merged_params[key] = merged_tensor
        
    return merged_params

def build_merged_image_classifier(base_model_checkpoint: str, 
                            finetuned_model_checkpoint: str, 
                            merged_state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Builds an image classifier attaching a fine-tuned model's classifier head
        to the merged weights.

    Args:
        base_model_checkpoint (str): The checkpoint of the base (pretrained) model.
        finetuned_model_checkpoint (str): The checkpoint of the fine-tuned model.
        merged_state_dict (dict[str, torch.Tensor]): The state dictionary containing
            the merged weights.

    Returns:
        torch.nn.Module: The merged image classifier model.
    """

    #full fine-tuned model to act as a "donor" for the classifier head
    finetuned_model = AutoModelForImageClassification.from_pretrained(finetuned_model_checkpoint)
    final_model_state_dict = {}

    for k, v in merged_state_dict.items():
        #'pooler' layer from the base model is not used in the classification model
        #so we skip it
        if not k.startswith("pooler."):
            #prefix vit. to match final architecture
            final_model_state_dict['vit.' + k] = v

    for k, v in finetuned_model.state_dict().items():
        #only copy the classifier weights from donor
        if k.startswith("classifier."):
            final_model_state_dict[k] = v

    #instantiate a new model with the correct classification head configuration,
    #the assembled final state dict will be used to populate the loaded model
    final_model = AutoModelForImageClassification.from_pretrained(
        base_model_checkpoint,
        num_labels=finetuned_model.config.num_labels,
        id2label=finetuned_model.config.id2label,
        label2id=finetuned_model.config.label2id
    )

    final_model.load_state_dict(final_model_state_dict)

    return final_model

def linear_merge(parameters: list[dict[str, torch.Tensor]], 
                 weights: list[float]) -> dict[str, torch.Tensor]:
    """Merges multiple models' weights a weighted linear combination.
    
    Args:
        parameters (list[dict[str, torch.Tensor]]): 
            A list of dictionaries corresponding to state dictionaries of different 
            models to be merged.
        weights (list[float]): 
            A list of weights corresponding to each state dictionary. 
            The weights are used for the linear combination.
    
    Returns:
        dict[str, torch.Tensor]: 
            The merged parameters.
    
    Raises:
        ValueError: If the number of weights does not match the number of state 
            dictionaries.
        KeyError: If a key is missing in any of the state dictionaries.
    """

    merged_parameters = copy.deepcopy(parameters[0])

    for key in merged_parameters.keys():
        #skip non-tensor params
        if not isinstance(merged_parameters[key], torch.Tensor):
            continue

        tensors_to_merge = [tv[key] for tv in parameters]
        merged_tensor = weighted_average(tensors_to_merge, weights)

        merged_parameters[key] = merged_tensor

    return merged_parameters