import torch

def compute_symmetrised_cross_covariance_eigenvectors(
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    """
    Computes the eigenvectors of the symmetrised cross-covariance matrix: ((A^T * B) + (A^T * B)^T) / 2

    Parameters:
        A (torch.Tensor): The first input tensor.
        B (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The transpose of the eigenvectors of the symmetrised cross-covariance matrix (ie: as rows).
    """
    # Compute the symmetrised cross-covariance matrix
    AT_B = torch.matmul(A.T, B)
    symmetrised_AT_B = (AT_B + AT_B.T) / 2
    
    # Compute the eigenvectors of the symmetrised cross-covariance matrix
    _, eigenvectors = torch.linalg.eigh(symmetrised_AT_B)
    
    return eigenvectors.T  # as rows

def project_data_onto_direction(data: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Projects the data onto the given direction vector.

    Parameters:
        data (torch.Tensor): The input data to be projected.
        direction (torch.Tensor): The direction vector onto which the data is projected.

    Returns:
        torch.Tensor: The projected data.
    """
    # Normalize the direction vector to ensure it is a unit vector
    direction = direction / torch.norm(direction)

    return torch.matmul(data, direction.reshape(-1, 1)).squeeze()

def compute_discriminant_ratio(projected_scoresA: torch.Tensor, projected_scoresB: torch.Tensor) -> torch.Tensor:
    """
    Computes the discriminant ratio between two sets of projected scores.

    Parameters:
        projected_scoresA (torch.Tensor): The first set of projected scores.
        projected_scoresB (torch.Tensor): The second set of projected scores.

    Returns:
        torch.Tensor: The discriminant ratio.
    """
    mean1 = torch.mean(projected_scoresA)
    mean2 = torch.mean(projected_scoresB)
    overall_mean = torch.mean(torch.cat([projected_scoresA, projected_scoresB]))
    n1 = projected_scoresA.size(0)
    n2 = projected_scoresB.size(0)
    between_class_variance = n1 * (mean1 - overall_mean) ** 2 + n2 * (mean2 - overall_mean) ** 2
    within_class_variance = torch.sum((projected_scoresA - mean1) ** 2) + torch.sum((projected_scoresB - mean2) ** 2)
    return between_class_variance / within_class_variance if within_class_variance != 0 else 0

def compute_variance_reduction(projected_scoresA: torch.Tensor, projected_scoresB: torch.Tensor) -> float:
    """
    Computes the variance reduction between two sets of projected scores.

    Parameters:
        projected_scoresA (torch.Tensor): The first set of projected scores.
        projected_scoresB (torch.Tensor): The second set of projected scores.

    Returns:
        float: The variance reduction value.
    """
    combined_scores = torch.cat([projected_scoresA, projected_scoresB])
    variance_reduction = max(0, 1 - (projected_scoresA.var() + projected_scoresB.var()) / (2 * combined_scores.var()))
    return variance_reduction
    
class DirectionAnalyzer:

    def __init__(
        self,
        hidden_state_data_manager,
        start_layer_index,
        skip_end_layers,
        discriminant_ratio_tolerance
    ):
        self.direction_matrices = self._analyze_directions(
            hidden_state_data_manager,
            start_layer_index,
            skip_end_layers,
            discriminant_ratio_tolerance
        )

    def _analyze_directions(
        self,
        hidden_state_data_manager,
        start_layer_index,
        skip_end_layers,
        discriminant_ratio_tolerance
    ):

        num_layers = hidden_state_data_manager.get_num_layers()

        # If passed a fraction, find the actual layer indices.
        if 0 < start_layer_index < 1:
            start_layer_index = round(start_layer_index * num_layers)
        if 0 < skip_end_layers < 1:
            skip_end_layers = round(skip_end_layers * num_layers)

        print(f"Testing Eigenvector Directions for layers {start_layer_index + 1} to {num_layers - skip_end_layers}:")

        num_dataset_types = hidden_state_data_manager.get_num_dataset_types()

        # [0] = de-bias direction, [1] = negative direction, [2] = positive direction.
        direction_matrices = [[[] for _ in range(num_layers)] for _ in range(num_dataset_types)]

        for layer_index in range(start_layer_index, num_layers - skip_end_layers):
            print(f"- Layer {layer_index + 1}: ", end = "", flush = True)

            data = hidden_state_data_manager.get_differenced_datasets(layer_index)

            if torch.cuda.is_available():
                data = [d.to('cuda').to(torch.float32) for d in data]  # Convert to CUDA and then to float32
            else:
                data = [d.to(torch.float32) for d in data]  # Convert to float32 on CPU
                print("CUDA is not available. Using CPU instead.")

            directions = compute_symmetrised_cross_covariance_eigenvectors(data[0], data[1])

            total_directions = directions.shape[0]

            results = []
            
            filtered_directions = 0

            # Project each direction onto datasets then store discriminant ratio and scaled/flipped direction.
            for i in range(directions.shape[0]):
                direction = directions[i,:]
                projected_scores = [project_data_onto_direction(d, direction) for d in data]
                discriminant_ratio = compute_discriminant_ratio(projected_scores[0], projected_scores[1])
                if discriminant_ratio >= discriminant_ratio_tolerance:
                    mean_desired = projected_scores[1].mean()
                    scaled_direction = mean_desired * direction # Scale and flip sign if needed.
                    results.append((discriminant_ratio, scaled_direction))
                    filtered_directions += 1

            if filtered_directions > 0:
                print(f"[{filtered_directions}/{total_directions} filtered]", end = "")
            else:
                print("[no directions filtered]", end = "")
                
            # Sort the directions into descending order using the scoring criterion.
            results.sort(key = lambda x: x[0], reverse = True)

            best_discriminant_ratio = 0.0
            best_variance_reduction = 0.0
            best_means = [0.0, 0.0]
            best_stds = [0.0, 0.0]
            best_direction_sum = torch.zeros_like(directions[0,:])

            selected_directions = 0

            # Greedily try to create an even better "compound direction".
            for result in results:
                direction_sum = best_direction_sum + result[1]
                direction = direction_sum / torch.norm(direction_sum)
                projected_scores = [project_data_onto_direction(d, direction) for d in data]
                discriminant_ratio = compute_discriminant_ratio(projected_scores[0], projected_scores[1])
                if discriminant_ratio > best_discriminant_ratio + discriminant_ratio_tolerance:
                    best_discriminant_ratio = discriminant_ratio
                    best_variance_reduction = compute_variance_reduction(projected_scores[0], projected_scores[1])
                    best_means = [projected_scores[0].mean(), projected_scores[1].mean()]
                    best_stds = [projected_scores[0].std(), projected_scores[1].std()]
                    best_direction_sum = direction_sum
                    selected_directions += 1

            # If we have a selected direction, then regularise it and use the scaled direction.
            if selected_directions > 0:
                midpoint = (best_means[0] + best_means[1]) / 2
                adjusted_means = [
                    best_means[0] - midpoint,
                    best_means[1] - midpoint
                ]
                raw_sum = abs(best_means[1]) + abs(best_means[0])
                raw_ratio = abs(best_means[1]) / raw_sum if raw_sum != 0 else 0.0
                print(f" [{selected_directions}/{total_directions} selected]", end = "")
                print(f" Δ = {best_discriminant_ratio * 100:.0f}%,", end = "")
                print(f" Δσ² = {best_variance_reduction * 100:.1f}%,", end = "")
                print(f" σ= ({best_stds[0]:.3f}, {best_stds[1]:.3f}),", end = "")
                print(f" μ = ({best_means[0]:.3f}, {best_means[1]:.3f} [{raw_ratio * 100:.1f}%]) --> ", end = "")
                print(f" μ' = ({midpoint:.3f}, {adjusted_means[0]:.3f}, {adjusted_means[1]:.3f})", end = "")
                print("")
                best_unit_direction = best_direction_sum / torch.norm(best_direction_sum)
                direction_matrices[0][layer_index].append(midpoint * best_unit_direction)           # de-bias vector.
                direction_matrices[1][layer_index].append(adjusted_means[0] * best_unit_direction)  # should be -ve of [2].
                direction_matrices[2][layer_index].append(adjusted_means[1] * best_unit_direction)  # should be -ve of [1].
            else:
                print(" [no directions selected]")

        direction_matrices = self._convert_to_torch_tensors(direction_matrices)

        return direction_matrices

    @staticmethod
    def _convert_to_torch_tensors(direction_matrices):
        direction_torch_tensors = []

        for i in range(len(direction_matrices)):
            layer_tensors = []
            for j in range(len(direction_matrices[i])):
                if direction_matrices[i][j]:
                    tensor = torch.stack(direction_matrices[i][j]).to(torch.float32).cpu()
                    layer_tensors.append(tensor)
                else:
                    layer_tensors.append(None)
            direction_torch_tensors.append(layer_tensors)

        return direction_torch_tensors
