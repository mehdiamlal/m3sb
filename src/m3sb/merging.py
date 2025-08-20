import copy
from geometry import slerp, barycenter, weighted_average
from utils import map_key_names
import torch
from transformers import AutoModelForImageClassification

def slerp_merge(interpolation_factor: float, 
                       task_vector1: dict[str, torch.Tensor], 
                       task_vector2: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merges two task vectors by interpolating their values.

    Args:
        interpolation_factor (float): The factor by which to interpolate
            between the two task vectors. Should be between 0 and 1.
        task_vector1 (dict[str, torch.Tensor]): The first task vector.
        task_vector2 (dict[str, torch.Tensor]): The second task vector.

    Returns:
        dict[str, torch.Tensor]: A new task vector that is a linear
            interpolation of the two input task vectors.

    Raises:
        AssertionError: If the input dictionaries do not have the same
            number of keys or if the interpolation factor is not in the range [0, 1].
    """
    assert 0 <= interpolation_factor <= 1
    assert len(task_vector1) == len(task_vector2)

    task_vector3 = copy.deepcopy(task_vector1)
    key_map = map_key_names(task_vector1, task_vector2)

    for k, v in task_vector1.items():
        task_vector3[k] = slerp(interpolation_factor, v, task_vector2[key_map[k]])

    return task_vector3

def pairwise_slerp_merge(task_vectors: list[dict[str, torch.Tensor]],
                         weights: list[float]) -> dict[str, torch.Tensor]:
    """Merges a list of task vectors using pairwise SLERP with specified weights.
    
    Args:
        task_vectors (list[dict[str, torch.Tensor]]): 
            A list of dictionaries, each mapping parameter names to PyTorch 
            tensors representing task vectors.
        weights (list[float]): 
            A list of weights corresponding to each task vector, used to 
            determine interpolation factors.
    
    Returns:
        dict[str, torch.Tensor]: 
            A dictionary representing the merged task vector, with parameter names as keys and PyTorch tensors as values.
    
    Raises:
        ValueError: If the lengths of `task_vectors` and `weights` do not match.
    """

    merged_task_vector = copy.deepcopy(task_vectors[0])
    cumulative_weight = weights[0]

    for i in range(1, len(task_vectors)):
        new_task_vector = task_vectors[i]
        new_model_weight = weights[i]

        #correct interpolation factor
        interp_factor = new_model_weight / (cumulative_weight + new_model_weight)

        merged_task_vector = slerp_merge(interp_factor, merged_task_vector, 
                                         new_task_vector)
        
        cumulative_weight += new_model_weight

    return merged_task_vector



def barycentric_merge(task_vectors: list[dict[str, torch.Tensor]], 
                      weights: list[float], iterations: int = 20, 
                      threshold: float = 1e-5) -> dict[str, torch.Tensor]:
    
    """Merges multiple model task_vectors using a barycentric approach.
    
    Args:
        task_vectors (list[dict[str, torch.Tensor]]): 
            A list of dictionaries corresponding to task vectors of different 
            models to be merged.
        weights (list[float]): 
            A list of weights corresponding to each state dictionary, used for 
            barycentric averaging.
        iterations (int, optional): 
            Number of iterations for the barycenter computation. Default is 20.
        threshold (float, optional): 
            Convergence threshold for the barycenter computation. Default is 1e-5.
    
    Returns:
        dict[str, torch.Tensor]: 
            A task vector.
    
    Notes:
        - Non-tensor parameters are skipped and not merged.
        - Assumes all state dictionaries have matching keys and compatible tensor 
        shapes.
    """

    merged_task_vector = copy.deepcopy(task_vectors[0])
    
    for key in merged_task_vector.keys():
        #skip non-tensor parameters 
        if not isinstance(merged_task_vector[key], torch.Tensor):
            continue

        tensors_to_merge = [tv[key] for tv in task_vectors]
        merged_tensor = barycenter(tensors_to_merge, weights, iterations,
                                       threshold)
        merged_task_vector[key] = merged_tensor
        
    return merged_task_vector

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

#================================ BASELINES ================================

def linear_merge(task_vectors: list[dict[str, torch.Tensor]], 
                 weights: list[float]) -> dict[str, torch.Tensor]:
    """Merges multiple task vectors using a weighted linear combination.
    
    Args:
        task_vectors (list[dict[str, torch.Tensor]]): 
            A list of dictionaries corresponding to task vectors of different 
            models to be merged.
        weights (list[float]): 
            A list of weights corresponding to each state dictionary. 
            The weights are used for the linear combination.
    
    Returns:
        dict[str, torch.Tensor]: 
            A merged task vector.
    
    Raises:
        ValueError: If the number of weights does not match the number of state 
            dictionaries.
        KeyError: If a key is missing in any of the state dictionaries.
    """

    merged_task_vector = copy.deepcopy(task_vectors[0])

    for key in merged_task_vector.keys():
        #skip non-tensor params
        if not isinstance(merged_task_vector[key], torch.Tensor):
            continue

        tensors_to_merge = [tv[key] for tv in task_vectors]
        merged_tensor = weighted_average(tensors_to_merge, weights)

        merged_task_vector[key] = merged_tensor

    return merged_task_vector