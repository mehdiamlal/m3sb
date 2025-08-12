import torch

def normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalizes a tensor to unit length.
    Args:
        vector (torch.Tensor): The tensor to normalize.
        eps (float, optional): A small value to avoid division by zero. 
            Defaults to 1e-8.

    Returns:
        torch.Tensor: The normalized tensor.
    """

    norm = torch.linalg.norm(vector) + eps
    if norm > eps:
        return vector / norm
    return vector 

def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between two vectors.

    Args:        
        t (float): Interpolation factor, should be between 0 and 1.
        v0 (torch.Tensor): The first vector.
        v1 (torch.Tensor): The second vector.  

    Returns:
        torch.Tensor: The interpolated vector.

    Raises:
        ValueError: If the input vectors are not of the same shape.
    """

    return (1 - t) * v0 + t * v1

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD=0.9995) -> torch.Tensor:
    """Spherical linear interpolation between two vectors.

    Args:
        t (float): Interpolation factor, should be between 0 and 1.
        v0 (torch.Tensor): The first vector.
        v1 (torch.Tensor): The second vector.
        DOT_THRESHOLD (float, optional): Threshold for using linear interpolation.
            Defaults to 0.9995.

    Returns:
        torch.Tensor: The interpolated vector.
    
    Raises:
        ValueError: If the input vectors are not of the same shape.
    """

    original_shape = v0.shape

    #flatten the tensors for geometric operations
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    u0 = normalize(v0_flat)
    u1 = normalize(v1_flat)

    dot = torch.dot(u0, u1).clamp(-1.0, 1.0)

    #if vectors are very close, use LERP on the original vectors
    if dot > DOT_THRESHOLD:
        return lerp(t, v0, v1)
    
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)
    
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    
    interp_unit_vector = w0 * u0 + w1 * u1

    #linear interpolation of magnitudes
    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)
    interp_norm = (1 - t) * norm0 + t * norm1

    #combining and reshaping
    interp_vector = interp_unit_vector * interp_norm
    return interp_vector.reshape(original_shape)

def project_on_tangent_space(v: torch.Vector, t: torch.Vector, 
                             DOT_THRESHOLD=0.9995) -> torch.Vector:
    """Projects a vector v onto a tangent space in t.
    Assumption: the vectors are normalized and flattened into a 1D tensor.

    Args:
        v (torch.Vector): The vector to map.
        t (torch.Vector): The tangent space vector.

    Returns:
        torch.Vector: The mapped vector in the tangent space.
    """
    dot = torch.dot(v, t).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    #if the vectors are very close, return a zero vector
    if dot > DOT_THRESHOLD:
        return torch.zeros_like(t)
    #if the vectors are anti-parallel, we need to find a vector orthogonal to t
    elif dot < -DOT_THRESHOLD:
        #just pick a random direction...
        helper = torch.randn_like(t)
        #Gram-schmidt process to find a vector orthogonal to t
        tangent_v = normalize(helper - torch.dot(helper, t) * t)
        return tangent_v * theta
    
    #general case
    tangent_v = normalize(v - dot * t)
    return tangent_v * theta