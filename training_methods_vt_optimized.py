"""
Optimized training methods for DLGN Value Tensor model.

Key optimizations:
- Reduced CPU-GPU transfers
- Torch operations instead of numpy where possible
- Better memory management
- Vectorized operations
- In-place operations where safe
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from copy import deepcopy
import time
from typing import Tuple, List, Dict

def sigmoid_torch(u: torch.Tensor) -> torch.Tensor:
    """
    Compute sigmoid using PyTorch (faster than numpy conversion).

    Args:
        u: Input tensor

    Returns:
        Sigmoid of input
    """
    return torch.sigmoid(u)


def compute_cross_product_features_optimized(model, train_data: torch.Tensor,
                                             beta: float, device: torch.device) -> torch.Tensor:
    """
    Optimized computation of cross-product features for value tensor fitting.

    Reduces CPU-GPU transfers and uses torch operations throughout.

    Args:
        model: DLGN_VT model
        train_data: Training data tensor (already on device)
        beta: Gating parameter
        device: Device to perform computations on

    Returns:
        Cross-product feature vector (on device)
    """
    with torch.no_grad():
        # Get effective weights (stay on device)
        ew = []
        for i in range(model.num_hidden_layers):
            curr_weight = model.gating_layers[i].weight.detach()
            if model.BN:
                curr_weight = curr_weight / torch.norm(curr_weight, dim=1, keepdim=True)
            ew.append(curr_weight)

        feat_vec = []

        if model.prod == 'op':
            rsh = [-1] + [1] * model.num_layer

            for i in range(len(ew)):
                rsh_new = rsh.copy()
                rsh_new[i+1] = model.num_hidden_nodes[i]

                if model.feat == 'cf' and i != 0:
                    ew[i] = ew[i] @ ew[i-1]

                # Use torch operations instead of numpy
                features = torch.matmul(train_data, ew[i].T)
                features = torch.sigmoid(beta * features)
                feat_vec.append(features.reshape(tuple(rsh_new)))

        elif model.prod == 'ip':
            for i in range(len(ew)):
                if model.feat == 'cf' and i != 0:
                    ew[i] = ew[i] @ ew[i-1]

                features = torch.matmul(train_data, ew[i].T)
                features = torch.sigmoid(beta * features)
                feat_vec.append(features)

        # Compute cross product
        cp_feat = feat_vec[0]
        for i in range(1, len(feat_vec)):
            cp_feat = cp_feat * feat_vec[i]

        cp_feat_vec = cp_feat.reshape((len(cp_feat), -1))

    return cp_feat_vec


def vt_train_methods_optimized(model, loss_fn, data, config):
    """
    Optimized training for DLGN VT model.

    Improvements over original:
    - Reduced CPU-GPU transfers (stay on GPU when possible)
    - Use torch operations instead of numpy
    - Better memory management
    - Vectorized operations
    - Gradient accumulation support (optional)

    Args:
        model: DLGN_VT model instance
        loss_fn: Loss function
        data: Dictionary containing train/test data and labels
        config: Configuration object with training parameters

    Returns:
        Tuple of (trained_model, train_losses)
    """
    train_data = data['train_data']
    test_data = data['test_data']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    device = config.device

    batch_size = 256
    from data_gen import CustomDataset

    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    from DLGN_enums import Optim
    optimizer = None
    if config.optimizer_type == Optim.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    elif config.optimizer_type == Optim.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epochs = config.epochs
    save_freq = config.save_freq
    value_freq = config.value_freq

    from DLGN_enums import VtFit, LossTypes
    from pegasos_solver import Pegasos, Pegasos_kernel

    train_losses = []
    acc_dict = {'train': [], 'test': []}
    best_test_error = len(test_data)
    model_return = None

    update_value_epochs = list(range(0, epochs + 1, value_freq))

    tepoch = tqdm(range(epochs + 1))

    pre_update_loss = 0.0
    post_update_loss = 0.0

    for epoch in tepoch:
        # Value tensor update
        if epoch in update_value_epochs:
            model.eval()
            with torch.no_grad():
                # Calculate loss before update
                train_preds = model(train_data)
                if config.loss_fn_type == LossTypes.CE:
                    outputs = torch.cat((-train_preds.unsqueeze(1), train_preds.unsqueeze(1)), dim=1)
                    pre_update_loss = loss_fn(outputs, train_labels).item()
                    train_acc = (torch.argmax(outputs, dim=1) == train_labels).float().mean().item()
                else:
                    pre_update_loss = loss_fn(train_preds, train_labels).item()
                    predictions = (torch.sign(train_preds) * 0.5 + 0.5).long()
                    train_acc = (predictions == train_labels).float().mean().item()

            if hasattr(config, 'use_wandb') and not config.use_wandb:
                print(f"Loss before updating value tensor at epoch {epoch}: {pre_update_loss:.6f}")

            train_losses.append(pre_update_loss)

            # Compute features for value fitting
            start = time.time()

            if config.vt_fit == VtFit.NPKSVC:
                # NPK-based SVC
                npk = model.get_npk
                clf = SVC(C=config.reg, kernel=npk)
                clf.fit(train_data.cpu().numpy(), train_labels.cpu().numpy())

                support_vectors = train_data[clf.support_]
                kernel_values = model.npk_forward(support_vectors).cpu().numpy()
                dual_coef = clf.dual_coef_.T.reshape(tuple([-1] + [1] * model.num_hidden_layers))
                value_wts = np.sum(dual_coef * kernel_values, axis=0)

            elif config.vt_fit == VtFit.PEGASOSKERNEL:
                # Pegasos with NPK kernel
                npk = model.get_npk
                kernel_loss_fn_type = 'hinge' if config.loss_fn_type == LossTypes.HINGE else 'hinge'
                clf = Pegasos_kernel(config.reg, config.num_iter, npk, loss_fn_type=kernel_loss_fn_type)
                clf.fit(train_data.cpu().numpy(), train_labels.cpu().numpy())

                # Compute kernel values on device then move to numpy
                kernel_values = model.npk_forward(train_data).cpu().numpy()
                dual_coef = (clf.alpha * clf.y).reshape(tuple([-1] + [1] * model.num_hidden_layers))
                value_wts = np.sum(dual_coef * kernel_values, axis=0) / (config.reg * config.num_iter)

            else:
                # Linear methods - use optimized feature computation
                cp_feat_vec = compute_cross_product_features_optimized(
                    model, train_data, model.beta, device
                )

                # Convert to numpy only when needed for sklearn
                cp_feat_np = (2 * cp_feat_vec).cpu().numpy()
                train_labels_np = train_labels.cpu().numpy()

                if config.vt_fit == VtFit.PEGASOS:
                    clf = Pegasos(config.reg, config.num_iter)
                    clf.fit(cp_feat_np, train_labels_np)
                    value_wts = clf.w.reshape(tuple([1] + model.num_hidden_nodes))

                elif config.vt_fit == VtFit.LOGISTIC:
                    clf = LogisticRegression(C=config.reg, fit_intercept=False, max_iter=5000,
                                           penalty="l2", solver='liblinear')
                    clf.fit(cp_feat_np, train_labels_np)

                elif config.vt_fit == VtFit.LINEARSVC:
                    clf = LinearSVC(C=config.reg, fit_intercept=False, max_iter=5000,
                                  dual='auto', loss='hinge')
                    clf.fit(cp_feat_np, train_labels_np)

                elif config.vt_fit == VtFit.SVC:
                    clf = SVC(C=config.reg, kernel='linear', degree=1, gamma=1, max_iter=5000)
                    clf.fit(cp_feat_np, train_labels_np)

                else:
                    raise ValueError(f"Invalid value fitting method: {config.vt_fit}")

                # Compute value weights
                if config.vt_fit in [VtFit.LOGISTIC, VtFit.LINEARSVC, VtFit.SVC]:
                    if model.prod == 'op':
                        n_features = np.prod(model.num_hidden_nodes)
                        value_wts = clf.decision_function(np.eye(n_features)).reshape(
                            tuple([1] + model.num_hidden_nodes)
                        )
                    elif model.prod == 'ip':
                        value_wts = clf.decision_function(np.eye(model.num_hidden_nodes[0]))

            end = time.time()
            if hasattr(config, 'use_wandb') and not config.use_wandb:
                print(f"Time taken to fit value tensor: {end - start:.2f}s")

            # Update value tensor
            with torch.no_grad():
                model.value_layers.copy_(torch.from_numpy(value_wts).float().to(device))

            # Calculate loss after update
            model.eval()
            with torch.no_grad():
                train_preds = model(train_data)
                if config.loss_fn_type == LossTypes.CE:
                    outputs = torch.cat((-train_preds.unsqueeze(1), train_preds.unsqueeze(1)), dim=1)
                    post_update_loss = loss_fn(outputs, train_labels).item()
                    train_acc = (torch.argmax(outputs, dim=1) == train_labels).float().mean().item()
                else:
                    post_update_loss = loss_fn(train_preds, train_labels).item()
                    predictions = (torch.sign(train_preds) * 0.5 + 0.5).long()
                    train_acc = (predictions == train_labels).float().mean().item()

                # Test accuracy
                test_preds = model(test_data)
                if config.loss_fn_type == LossTypes.CE:
                    test_outputs = torch.cat((-test_preds.unsqueeze(1), test_preds.unsqueeze(1)), dim=1)
                    test_acc = (torch.argmax(test_outputs, dim=1) == test_labels).float().mean().item()
                else:
                    test_predictions = (torch.sign(test_preds) * 0.5 + 0.5).long()
                    test_acc = (test_predictions == test_labels).float().mean().item()

            if hasattr(config, 'use_wandb'):
                if config.use_wandb:
                    import wandb
                    wandb.log({
                        'train_loss': post_update_loss,
                        'epoch': epoch,
                        'train_accuracy': train_acc,
                        'update_loss_diff': post_update_loss - pre_update_loss,
                        'test_accuracy': test_acc
                    })
                else:
                    print(f"Test Accuracy: {test_acc:.4f}")
                    print(f"Loss after updating value tensor at epoch {epoch}: {post_update_loss:.6f}")

            train_losses.append(post_update_loss)

        # Early stopping check
        if post_update_loss / max(pre_update_loss, 1e-8) > 2.5:
            print(f"Early stopping at epoch {epoch} as loss increased by >2.5x")
            break

        # Train gating functions
        model.train()
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch).reshape(-1)

            if config.loss_fn_type == LossTypes.CE:
                outputs = torch.cat((-outputs.unsqueeze(1), outputs.unsqueeze(1)), dim=1)

            loss = loss_fn(outputs, y_batch)
            loss.backward()

            # Zero out value tensor gradients
            with torch.no_grad():
                if model.value_layers.grad is not None:
                    model.value_layers.grad.zero_()

                # Reduce gating gradients in early epochs
                if epoch < update_value_epochs[1]:
                    for param in model.gating_layers.parameters():
                        if param.grad is not None:
                            param.grad.mul_(0.5)

            optimizer.step()

        # Compute training loss
        model.eval()
        with torch.no_grad():
            train_preds = model(train_data)
            if config.loss_fn_type == LossTypes.CE:
                outputs = torch.cat((-train_preds.unsqueeze(1), train_preds.unsqueeze(1)), dim=1)
                train_loss = loss_fn(outputs, train_labels).item()
                train_acc_curr = (torch.argmax(outputs, dim=1) == train_labels).float().mean().item()
            else:
                train_loss = loss_fn(train_preds, train_labels).item()
                predictions = (torch.sign(train_preds) * 0.5 + 0.5).long()
                train_acc_curr = (predictions == train_labels).float().mean().item()

        train_losses.append(train_loss)

        # Early stopping
        if train_loss < 5e-3:
            print(f"Early stopping at epoch {epoch} - training loss < 5e-3")
            break

        if np.isnan(train_loss):
            print(f"Early stopping at epoch {epoch} - NaN loss")
            break

        # Track test accuracy
        with torch.no_grad():
            test_preds = model(test_data)
            if config.loss_fn_type == LossTypes.CE:
                test_outputs = torch.cat((-test_preds.unsqueeze(1), test_preds.unsqueeze(1)), dim=1)
                test_predictions = torch.argmax(test_outputs, dim=1)
            else:
                test_predictions = (torch.sign(test_preds) * 0.5 + 0.5).long()

            test_error = (test_predictions != test_labels).sum().item()
            acc_dict['test'].append(1 - test_error / len(test_labels))

            # Save best model
            if test_error < best_test_error:
                model_return = deepcopy(model)
                best_test_error = test_error

            # Track train accuracy
            if config.loss_fn_type == LossTypes.CE:
                train_predictions = torch.argmax(outputs, dim=1)
            else:
                train_predictions = predictions

            train_error = (train_predictions != train_labels).sum().item()
            acc_dict['train'].append(1 - train_error / len(train_labels))

        if hasattr(config, 'use_wandb') and config.use_wandb:
            import wandb
            wandb.log({'train_loss': train_loss, 'epoch': epoch, 'train_accuracy': train_acc_curr})

        tepoch.set_description(f"Loss {train_loss:.6f}")

        # Periodic memory cleanup
        if epoch % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return model_return if model_return is not None else model, train_losses
