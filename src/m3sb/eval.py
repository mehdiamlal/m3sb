import torch
from tqdm import tqdm

def evaluate_model(model: torch.nn.Module, data_loader: "DataLoader", 
                   device: str="cuda" if torch.cuda.is_available() else "cpu") -> dict[str, float]:
    """Evaluates a given model on a dataset provided by a DataLoader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): The DataLoader providing the evaluation data.
        device (str): The device to run the evaluation on (cuda or cpu).

    Returns:
        dict[str, float]: The model's evaluation metrics.
    """
    results = {}
    model.to(device)
    model.eval()
    
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating"):
            #The collate_fn might return None if a whole batch was corrupted
            if batch is None:
                continue
                
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=images)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")
    results["accuracy"] = accuracy
    return results