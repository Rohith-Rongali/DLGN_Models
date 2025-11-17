import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union

class DLGN_VT(nn.Module):
    """
    Deep Learning Gated Network - Value Tensor (DLGN_VT) model.

    Optimized implementation with improved performance through:
    - Unified gate score computation
    - Reduced code duplication
    - Better memory efficiency
    - Type hints and documentation

    Args:
        input_dim: Dimensionality of input features
        output_dim: Dimensionality of output (typically 1 for binary classification)
        num_hidden_nodes: List specifying the width of each hidden layer
        beta: Soft gating parameter controlling sigmoid steepness
        mode: Operation mode (default: 'pwc')
        value_scale: Scaling factor for value tensor initialization
        BN: Whether to use batch normalization on weights
        prod: Product type - 'op' (outer product) or 'ip' (inner product)
        feat: Feature type - 'sf' (shallow features) or 'cf' (composite features)
    """
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None,
                 num_hidden_nodes: List[int] = [], beta: float = 30, mode: str = 'pwc',
                 value_scale: float = 500., BN: bool = False, prod: str = 'op', feat: str = 'sf'):
        super(DLGN_VT, self).__init__()
        self.num_hidden_layers = len(num_hidden_nodes)
        self.beta = beta  # Soft gating parameter
        self.mode = mode
        self.BN = BN
        self.prod = prod
        self.feat = feat
        self.num_nodes = [input_dim] + num_hidden_nodes + [output_dim]
        self.gating_layers = nn.ModuleList()

        if self.prod == 'op':
            self.value_layers = nn.Parameter(torch.randn([1] + num_hidden_nodes) / value_scale)
        elif self.prod == 'ip':
            self.value_layers = nn.Parameter(torch.randn(num_hidden_nodes[0]) / value_scale)

        self.num_layer = len(num_hidden_nodes)
        self.num_hidden_nodes = num_hidden_nodes

        for i in range(self.num_hidden_layers):
            if self.feat == 'sf':
                temp = nn.Linear(self.num_nodes[0], self.num_nodes[i+1], bias=False)
            elif self.feat == 'cf':
                temp = nn.Linear(self.num_nodes[i], self.num_nodes[i+1], bias=False)
            self.gating_layers.append(temp)

    @torch.jit.ignore
    def set_parameters_with_mask(self, to_copy, parameter_masks):
        """
        Copy parameters from another model using a mask.

        Args:
            to_copy: Source DLGN_FC object with same architecture
            parameter_masks: Compatible with dict(to_copy.named_parameters())
        """
        for (name, copy_param) in to_copy.named_parameters():
            copy_param = copy_param.clone().detach()
            orig_param = self.state_dict()[name]
            if name in parameter_masks:
                param_mask = parameter_masks[name] > 0
                orig_param[param_mask] = copy_param[param_mask]
            else:
                orig_param = copy_param.data.detach()

    def return_gating_functions(self) -> List[torch.Tensor]:
        """
        Return effective weights for all gating layers.

        Returns:
            List of weight tensors for each hidden layer
        """
        effective_weights = []
        for i in range(self.num_hidden_layers):
            curr_weight = self.gating_layers[i].weight.detach().clone()
            if self.BN:
                curr_weight = curr_weight / torch.norm(curr_weight, dim=1, keepdim=True)
            effective_weights.append(curr_weight)
        return effective_weights

    def _compute_gate_score_layer(self, x: torch.Tensor, h: torch.Tensor,
                                   layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified gate score computation for a single layer.

        Optimized to avoid code duplication between get_gate_scores and npk_forward.

        Args:
            x: Original input tensor
            h: Current hidden state
            layer_idx: Index of current layer

        Returns:
            Tuple of (gate_score, updated_h)
        """
        if self.BN:
            # Compute normalized output
            norm = torch.norm(self.gating_layers[layer_idx].weight, dim=1, keepdim=True).T
            if self.feat == 'cf':
                h = self.gating_layers[layer_idx](h) / norm
                gate_score = torch.sigmoid(self.beta * h)
            else:  # 'sf' or 'nf'
                gate_score = torch.sigmoid(self.beta * self.gating_layers[layer_idx](x) / norm)
        else:
            if self.feat == 'cf':
                h = self.gating_layers[layer_idx](h)
                gate_score = torch.sigmoid(self.beta * h)
            else:  # 'sf' or 'nf'
                gate_score = torch.sigmoid(self.beta * self.gating_layers[layer_idx](x))

        return gate_score, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size,)
        """
        cp = self.npk_forward(x)
        if self.prod == 'op':
            return torch.sum(cp * self.value_layers, dim=tuple(range(1, self.num_layer + 1)))
        elif self.prod == 'ip':
            return torch.sum(cp * self.value_layers, dim=1)

    def get_npk(self, X: Union[np.ndarray, torch.Tensor],
                Y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute the Neural Path Kernel (NPK) matrix between two sets of inputs.

        Args:
            X: Input tensor of shape (n, d)
            Y: Input tensor of shape (m, d)

        Returns:
            NPK matrix of shape (n, m) as numpy array
        """
        device = self.gating_layers[0].weight.device

        # Convert to torch tensors if needed and move to device
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(device)
        else:
            X = X.to(device)

        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float().to(device)
        else:
            Y = Y.to(device)

        if self.prod == 'op':
            gate_scores_x = self.get_gate_scores(X)
            gate_scores_y = self.get_gate_scores(Y)

            kval = torch.ones(X.shape[0], Y.shape[0], device=device)
            for i in range(len(gate_scores_x)):
                kval = kval * torch.matmul(gate_scores_x[i], gate_scores_y[i].T)

            return kval.detach().cpu().numpy()
        else:
            raise NotImplementedError("NPK only implemented for outer product mode")

    def get_gate_scores(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute gate scores for all layers.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            List of gate score tensors for each layer
        """
        gate_scores = []
        h = x

        for i in range(len(self.gating_layers)):
            gate_score, h = self._compute_gate_score_layer(x, h, i)
            gate_scores.append(gate_score)

            if self.feat == 'nf':
                h = gate_score

        return gate_scores

    def npk_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Neural Path Kernel forward pass (cross-product of gate scores).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Cross-product tensor of all gate scores
        """
        h = x
        cp = None

        for i in range(self.num_hidden_layers):
            gate_score, h = self._compute_gate_score_layer(x, h, i)

            if self.feat == 'nf':
                h = gate_score

            if self.prod == 'op':
                # Create fiber shape for outer product
                fiber = [len(x)] + [1] * self.num_hidden_layers
                fiber[i + 1] = self.num_hidden_nodes[i]
                gate_score = gate_score.reshape(tuple(fiber))

            if cp is None:
                cp = gate_score
            else:
                cp = cp * gate_score

        return cp

    def log_features(self, bias: bool = False) -> torch.Tensor:
        """
        Log and return the learned features.

        Args:
            bias: Whether to include bias terms (not currently used)

        Returns:
            Concatenated feature tensor
        """
        weights = self.return_gating_functions()

        if self.feat == 'cf':
            # Composite features: accumulate compositions
            Feature_list = [weights[0].T / torch.linalg.norm(weights[0].T, ord=2, dim=1, keepdim=True)]
            for w in weights[1:]:
                composed = w.T @ Feature_list[-1]
                Feature_list.append(composed / torch.linalg.norm(composed, ord=2, dim=1, keepdim=True))
        else:
            # Shallow features: just normalize
            Feature_list = [w.T / torch.linalg.norm(w.T, ord=2, dim=1, keepdim=True)
                          for w in weights]

        features = torch.cat(Feature_list, dim=0).cpu()
        return features
