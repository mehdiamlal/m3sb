import torch
import copy

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

def weighted_average(parameters: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """Computes the weighted average of a list of tensors.
    
    Args:
        parameters (list[torch.Tensor]): List of tensors to be averaged.
        weights (list[float]): List of weights corresponding to each tensor.
    
    Returns:
        torch.Tensor: The weighted average tensor, with the same shape as the 
            input tensors.
    
    Raises:
        ValueError: If `parameters` or `weights` are empty, or if their lengths 
            do not match.
    """

    original_shape = parameters[0].shape
    flat_params = [(p.flatten() * w ) for p, w in zip(parameters, weights)]
    return sum(flat_params).reshape(original_shape)

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

def project_on_tangent_space(v: torch.Tensor, t: torch.Tensor, 
                             DOT_THRESHOLD=0.9995) -> torch.Tensor:
    """Projects a vector v onto a tangent space in t.
    Assumption: the vectors are normalized and flattened into a 1D tensor.

    Args:
        v (torch.Tensor): The vector to map.
        t (torch.Tensor): The tangent space vector.

    Returns:
        torch.Tensor: The mapped vector in the tangent space.
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

def project_on_hypersphere(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Projects a vector v onto a hypersphere from a tangent space in t.
    Assumption: the vector t is normalized and both v and t are flattened into 
    a 1D tensor.

    Args:
        v (torch.Tensor): The vector to project.
        t (torch.Tensor): The tangent space vector.

    Returns:
        torch.Tensor: The projected vector on the hypersphere.
    """
    
    norm_v = torch.linalg.norm(v)
    u = normalize(v)
    sin_theta = torch.sin(norm_v)
    cos_theta = torch.cos(norm_v)

    sphere_v = cos_theta * t + sin_theta * u

    return sphere_v

def barycenter(parameters: list[torch.Tensor], weights: list[float], 
                   iterations: int, threshold: float) -> torch.Tensor:
    
    """Computes the weighted barycenter of a list of parameter tensors on a 
    hypersphere.
    
    Args:
        parameters (list[torch.Tensor]): List of tensors representing parameter 
            vectors to average.
        weights (list[float]): List of weights corresponding to each parameter 
            tensor.
        iterations (int): Maximum number of iterations for the barycenter 
            computation.
        threshold (float): Convergence threshold for the change in barycenter 
            (measured as the angle between successive estimates).
    
    Returns:
        torch.Tensor: The barycenter tensor, reshaped to match the original parameter shape.
    """
    

    original_shape = parameters[0].shape
    norms = [torch.linalg.norm(p.flatten()) for p in parameters]
    u_vectors = [normalize(p.flatten()) for p in parameters]
    barycenter = copy.deepcopy(u_vectors[0])

    for i in range(iterations):
        tangent_vectors = [project_on_tangent_space(u, barycenter) for u in u_vectors]
        avg_tangent = torch.stack([w * t for w, t in zip(weights, tangent_vectors)]).sum(dim=0)
        
        new_barycenter = project_on_hypersphere(avg_tangent, barycenter)

        #the distance between the new barycenter and the old one is the angle
        #between the two vectors
        distance = torch.acos(torch.dot(barycenter, new_barycenter).clamp(-1.0, 1.0))

        barycenter = new_barycenter

        if distance < threshold:
            #converged
            break

    #linearly interpolate the norms of the parameter vectors to resclale the
    #found barycenter
    interp_norm = sum(w * n for w, n in zip(weights, norms))

    final_vector = barycenter * interp_norm

    return final_vector.reshape(original_shape)