import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from typing import Any

def evaluate_model(
    model: torch.nn.Module, 
    data_loader: "DataLoader", 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict[str, Any]:
    """Evaluates a model on a given task and returns the evaluation metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): The DataLoader providing the evaluation data.
        device (str): The device to run the evaluation on (cuda or cpu).

    Returns:
        dict[str, Any]: A dictionary containing:
                         - 'metrics': A dict of accuracy, precision, recall, f1.
                         - 'labels': A list of all true labels.
                         - 'predictions': A list of all model predictions.
    """

    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if batch is None:
                continue
                
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=images)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    #computing the metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    correct_predictions = sum(p == l for p, l in zip(all_predictions, all_labels))
    total_samples = len(all_labels)
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100
    }
    
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1-Score: {metrics['f1_score']:.2f}%")
    
    return {
        'metrics': metrics,
        'labels': all_labels,
        'predictions': all_predictions
    }