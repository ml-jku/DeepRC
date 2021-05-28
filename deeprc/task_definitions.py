# -*- coding: utf-8 -*-
"""
Specifications of targets to train DeepRC model on

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import numpy as np
import pandas as pd
from typing import List, Union
import torch
from sklearn import metrics


class Target(torch.nn.Module):
    def __init__(self, target_id: str, n_output_features: int, task_weight: float = 1.):
        """Base class for targets. 
        
        Targets represent a task to train on and are combined via `deeprc.task_definitions.TaskDefinition`. 
        The DeepRC model will be trained on the tasks listed in `deeprc.task_definitions.TaskDefinition`.
        You may create your own target classes by inheriting from this class.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        target_id: str
             ID of target as string.
        n_output_features: int
             Number of output features required by the network for this task.
        task_weight: float
            Weight of this task for the total training loss. The training loss is computed as weighted sum of the 
            individual task losses times their respective tasks-weights.
        """
        super(Target, self).__init__()
        self.__target_id__ = target_id
        self.__n_output_features__ = n_output_features
        self.__task_weight__ = torch.nn.Parameter(torch.tensor(task_weight, dtype=torch.float), requires_grad=False)
    
    def get_targets(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Get target values of all samples from `dataframe` as np.array. `dataframe` is the content of the metadata
        file.
        
        Parameters
        ----------
        dataframe: pd.DataFrame
             Content of the metadata file as read by `deeprc.dataset_readers.RepertoireDataset`.
        
        Returns
        ---------
        targets: np.ndarray
            Target values of all samples from `dataframe` as np.ndarray of datatype `np.float` and shape 
            `(n_samples, self.n_output_features)`.
        """
        raise NotImplementedError("Please add your own get_targets() method to your Target class")
    
    def activation_function(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """Activation function to apply to network outputs to create prediction
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
             `(n_samples, self.n_output_features)`.
        
        Returns
        ---------
        activated_output: torch.Tensor
            Activated output of DeepRC network for this task as `torch.Tensor`.
        """
        raise NotImplementedError("Please add your own activation_function() method to your Target class")
    
    def loss_function(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss function used for training on this task
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, self.n_output_features)`
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, self.n_output_features)`
        
        Returns
        ---------
        loss: torch.Tensor
            Loss for this task as torch.Tensor of shape `(n_samples, 1)`.
        """
        raise NotImplementedError("Please add your own loss_function() method to your Target class")
    
    def get_id(self) -> str:
        """Get target ID as string"""
        return self.__target_id__
    
    def get_task_weight(self) -> torch.Tensor:
        """Get task weight as string"""
        return self.__task_weight__
    
    def get_scores(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Get scores for this task as dictionary
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, self.n_output_features)`
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, self.n_output_features)`
        
        Returns
        ---------
        scores: dict
            Dictionary of format `{score_id: score_value}`, e.g. `{"loss": 0.01}`.
        """
        return dict(loss=self.loss_function(raw_outputs=raw_outputs, targets=targets).detach().mean().cpu().item())


class BinaryTarget(Target):
    def __init__(self, column_name: str, true_class_value, target_id: str = '', task_weight: float = 1.,
                 pos_weight: float = 1.):
        """Creates a binary classification target.
        
        Network output for this task will be 1 output feature, activated using a sigmoid output function.
        Network loss is computed using `torch.nn.BCEWithLogitsLoss`.
        Scores are computed using 0.5 as prediction threshold.
        
        Targets are combined via `TaskDefinition`. The DeepRC model will be trained on the targets listed in
        `TaskDefinition`.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        column_name: str
             Name of column in metadata file that contains the values for this task.
        true_class_value: str
             Entries with value `true_class_value` will be positive class, others will be negative class. Entries
             with no value are treated as NaN entries, do not belong to any class, and are ignored during training if
             `deeprc.task_definitions.train(ignore_missing_target_values=True)`.
        target_id: str
             ID of target as string. If None, uses `column_name` as ID.
        task_weight: float
            Weight of this task for the total training loss. The training loss is computed as weighted sum of the 
            individual task losses times their respective tasks-weights.
        pos_weight: float
             Up- or down-weight the contribution of positive class samples. Used as `pos_weight` argument of
              `torch.nn.BCEWithLogitsLoss()`.
        """
        super(BinaryTarget, self).__init__(target_id=target_id if len(target_id) else column_name,
                                           n_output_features=1, task_weight=task_weight)
        self.column_name = column_name
        self.true_class_value = np.array(true_class_value)
        self.__pos_weight__ = torch.nn.Parameter(torch.tensor(pos_weight, dtype=torch.float), requires_grad=False)
        self.binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.__pos_weight__, reduction='none')
    
    def get_targets(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Get target values of all samples from `dataframe` as np.array. `dataframe` is the content of the metadata
        file.
        
        Parameters
        ----------
        dataframe: pd.DataFrame
             Content of the metadata file as read by `deeprc.dataset_readers.RepertoireDataset`.
        
        Returns
        ---------
        targets: np.ndarray
            Target values of all samples from `dataframe` as np.ndarray of datatype `np.float` and shape 
            `(n_samples, 1)`.
        """
        return np.asarray(self.true_class_value[None] == dataframe[self.column_name].values[:, None], dtype=np.float)
    
    def activation_function(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """Sigmoid activation function to apply to network outputs to create prediction
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
             `(n_samples, 1)`.
        
        Returns
        ---------
        activated_output: torch.Tensor
            Activated output of DeepRC network for this task as `torch.Tensor` of shape
            `(n_samples, 1)`.
        """
        return torch.sigmoid(raw_outputs)
    
    def loss_function(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """`torch.nn.BCEWithLogitsLoss()` loss used for training on this task
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, 1)`
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, 1)`
        
        Returns
        ---------
        loss: torch.Tensor
            Loss for this task as torch.Tensor of shape `(n_samples, 1)`.
        """
        return self.binary_cross_entropy_loss(raw_outputs, targets)
    
    def get_scores(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Get scores for this task as dictionary containing AUC, BACC, F1, and loss
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, 1)`.
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, 1)`.
        
        Returns
        ---------
        scores: dict
            Dictionary of format `{score_id: score_value}`, e.g. `{"auc": 0.6, "bacc": 0.5, "f1": 0.2, "loss": 0.01}`.
        """
        predictions = self.activation_function(raw_outputs=raw_outputs).detach()
        predictions_thresholded = (predictions > 0.5).float().cpu().numpy()
        predictions = predictions.float().cpu().numpy()
        labels = targets.detach().cpu().numpy()
        labels = labels[..., 0]
        predictions_thresholded = predictions_thresholded[..., 0]
        roc_auc = metrics.roc_auc_score(y_true=labels, y_score=predictions, average=None)
        bacc = metrics.balanced_accuracy_score(y_true=labels, y_pred=predictions_thresholded)
        f1 = metrics.f1_score(y_true=labels, y_pred=predictions_thresholded, average='binary',
                              pos_label=1)
        loss = self.loss_function(raw_outputs=raw_outputs, targets=targets).detach().mean().cpu().item()
        return dict(roc_auc=roc_auc, bacc=bacc, f1=f1, loss=loss)


class MulticlassTarget(Target):
    def __init__(self, column_name: str, possible_target_values: List[str], task_weight: float = 1.,
                 target_id: str = '', class_weights: Union[list, np.ndarray] = None):
        """Creates a multi-class classification target.
        
        Network output for this task will be `len(possible_target_values)` output features, activated using a softmax
        output function.
        Network loss is computed using `torch.nn.CrossEntropyLoss`.
        
        Targets are combined via `TaskDefinition`. The DeepRC model will be trained on the targets listed in
        `TaskDefinition`.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        column_name
             Name of column in metadata file that contains the values for this task.
        possible_target_values
             Values to expect in column `column_name`. Each value corresponds to a class.
        target_id
             ID of target as string. If None, uses `column_name` as ID.
        task_weight
            Weight of this task for the total training loss. The training loss is computed as weighted sum of the
            individual task losses times their respective tasks-weights.
        class_weights
             Up- or down-weight the contribution of the individual classes. Used as `weight` argument of
              `torch.nn.CrossEntropyLoss()`.
        """
        super(MulticlassTarget, self).__init__(target_id=target_id if len(target_id) else column_name,
                                               n_output_features=len(possible_target_values), task_weight=task_weight)
        self.column_name = column_name
        self.possible_target_values = np.array(possible_target_values)
        if class_weights is None:
            class_weights = np.ones(shape=(len(self.possible_target_values),), dtype=np.float)
        self.__class_weights__ = torch.nn.Parameter(torch.tensor(class_weights, dtype=torch.float), requires_grad=False)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.__class_weights__, reduction='none')
    
    def get_targets(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Get target values of all samples from `dataframe` as np.array. `dataframe` is the content of the metadata
        file.
        
        Parameters
        ----------
        dataframe: pd.DataFrame
             Content of the metadata file as read by `deeprc.dataset_readers.RepertoireDataset`.
        
        Returns
        ---------
        targets: np.ndarray
            Target values of all samples from `dataframe` as np.ndarray of datatype `np.float` and shape
            `(n_samples, len(self.possible_target_values))`.
        """
        return np.asarray(self.possible_target_values[None, :] == dataframe[self.column_name].values[:, None],
                          dtype=np.float)
    
    def activation_function(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """Softmax activation function to apply to network outputs to create prediction
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
             `(n_samples, len(self.possible_target_values))`.
        
        Returns
        ---------
        activated_output: torch.Tensor
            Activated output of DeepRC network for this task as `torch.Tensor` of shape
             `(n_samples, len(self.possible_target_values))`.
        """
        return torch.softmax(raw_outputs, dim=-1)
    
    def loss_function(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """`torch.nn.CrossEntropyLoss()` loss used for training on this task
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, 1)`
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, 1)`
        
        Returns
        ---------
        loss: torch.Tensor
            Loss for this task as torch.Tensor of shape `(n_samples, 1)`.
        """
        return self.cross_entropy_loss(raw_outputs, targets.argmax(dim=-1))[..., None]


class RegressionTarget(Target):
    def __init__(self, column_name: str, target_id: str = '', task_weight: float = 1., normalization_mean: float = 0.,
                 normalization_std: float = 1.):
        """Creates a regression target.
        
        Network output for this task will be 1 output feature, activated using a linear output function.
        Network loss is computed using `torch.nn.MSELoss`.
        
        Targets are combined via `TaskDefinition`. The DeepRC model will be trained on the targets listed in
        `TaskDefinition`.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        column_name: str
             Name of column in metadata file that contains the values for this task.
        target_id: str
             ID of target as string. If None, uses `column_name` as ID.
        task_weight: float
            Weight of this task for the total training loss. The training loss is computed as weighted sum of the
            individual task losses times their respective tasks-weights.
        normalization_mean: float
             Normalize target value 'target_value' as (('target_value' - `normalization_mean`) / `normalization_std`).
        normalization_std: float
             Normalize target value 'target_value' as (('target_value' - `normalization_mean`) / `normalization_std`).
        """
        super(RegressionTarget, self).__init__(target_id=target_id if len(target_id) else column_name,
                                               n_output_features=1, task_weight=task_weight)
        self.column_name = column_name
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.mse_loss = torch.nn.MSELoss(reduction='none')
    
    def get_targets(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Get target values of all samples from `dataframe` as np.array. `dataframe` is the content of the metadata
        file.
        
        Parameters
        ----------
        dataframe: pd.DataFrame
             Content of the metadata file as read by `deeprc.dataset_readers.RepertoireDataset`.
        
        Returns
        ---------
        targets: np.ndarray
            Target values of all samples from `dataframe` as np.ndarray of datatype `np.float` and shape
            `(n_samples, 1)`.
        """
        return (np.array(dataframe[self.column_name].values[:, None], dtype=np.float)
                - self.normalization_mean) / self.normalization_std
    
    def activation_function(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """Linear activation function to apply to network outputs to create prediction (does not affect loss function!)
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
             `(n_samples, 1)`.
        
        Returns
        ---------
        activated_output: torch.Tensor
            Activated output of DeepRC network for this task as `torch.Tensor` of shape
            `(n_samples, 1)`.
        """
        return raw_outputs
    
    def loss_function(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """`torch.nn.MSELoss()` loss used for training on this task (using raw outputs before activation function!)
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, 1)`
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, 1)`
        
        Returns
        ---------
        loss: torch.Tensor
            Loss for this task as torch.Tensor of shape `(n_samples, 1)`.
        """
        return self.mse_loss(raw_outputs, targets)

    def get_scores(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Get scores for this task as dictionary containing R^2, mean absolute error, mean squared error, and loss
        
        R^2 ("r2"), mean absolute error ("mae"), and mean squared error ("mse") are computed using
        sklearn.metrics r2_score, mean_absolute_error, and mean_squared_error, respectively.
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, 1)`.
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, 1)`.
        
        Returns
        ---------
        scores: dict
            Dictionary of format `{score_id: score_value}`, e.g. `{"r2": 0.6, "mae": 0.5, "mse": 0.2, "loss": 0.01}`.
        """
        # Get denormalized predictions and targets as torch.float values
        predictions = self.activation_function(raw_outputs=raw_outputs).float().cpu().numpy()
        predictions_denorm = predictions * self.normalization_std + self.normalization_mean
        targets = targets.float().cpu().numpy()
        targets_denorm = targets * self.normalization_std + self.normalization_mean
        
        # Compute scores
        r2 = metrics.r2_score(targets_denorm, predictions_denorm)
        mae = metrics.mean_absolute_error(targets_denorm, predictions_denorm)
        mse = metrics.mean_squared_error(targets_denorm, predictions_denorm)
        loss = self.loss_function(raw_outputs=raw_outputs, targets=targets).detach().mean().cpu().item()
        return dict(r2=r2, mae=mae, mse=mse, loss=loss)


class TaskDefinition(torch.nn.Module):
    def __init__(self, targets: List[Target]):
        """Combines one or more `deeprc.task_definitions.Target` instances into one task to train the DeepRC model on
        
        Provides access to the losses, scores, loss functions, activation functions, task weights, and IDs of the
        individual `deeprc.task_definitions.Target` instances.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        targets
             List of `deeprc.task_definitions.Target` instances to train the DeepRC model on.
        """
        super(TaskDefinition, self).__init__()
        self.__registered_task_modules__ = torch.nn.ModuleList(targets)
        self.__targets__ = targets
        self.__target_ids__ = tuple([t.__target_id__ for t in self.__targets__])
        self.__activation_functions__ = [t.activation_function for t in self.__targets__]
        self.__loss_functions__ = [t.loss_function for t in self.__targets__]
        self.__n_output_features__ = np.array([t.__n_output_features__ for t in self.__targets__])
        self.__total_output_features__ = int(np.sum(self.__n_output_features__))
        self.__task_weights__ = np.array([t.__task_weight__ for t in self.__targets__])
        cumsum_n_output_features = [0] + list(np.cumsum(self.__n_output_features__))
        self.__targets_slices__ = [slice(start, stop) for start, stop in zip(cumsum_n_output_features[:-1],
                                                                             cumsum_n_output_features[1:])]
    
    def get_targets(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Get target values of all samples from `dataframe` as np.array for all `deeprc.task_definitions.Target`
         instances. `dataframe` is the content of the metadata file.
        
        Parameters
        ----------
        dataframe: pd.DataFrame
             Content of the metadata file as read by `deeprc.dataset_readers.RepertoireDataset`.
        
        Returns
        ---------
        targets: np.ndarray
            Target values of all samples from `dataframe` as np.ndarray of datatype `np.float` and shape
            `(n_samples, n_target_features)`.
        """
        return np.concatenate([np.asarray(t.get_targets(dataframe), dtype=np.float32)
                               for t in self.__targets__], axis=-1)
    
    def activation_function(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """Returns network output after activation functions for all `deeprc.task_definitions.Target` instances.
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network as torch.Tensor of shape `(n_samples, n_target_features)`.
        
        Returns
        ---------
        activated_output: torch.Tensor
            Activated output of DeepRC network for all `deeprc.task_definitions.Target` instances as `torch.Tensor`.
        """
        return torch.cat([a(raw_outputs[..., s])
                          for s, a in zip(self.__targets_slices__, self.__activation_functions__)], dim=-1)
    
    def get_losses(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get individual losses for all `deeprc.task_definitions.Target` instances.
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network as torch.Tensor of shape
            `(n_samples, n_output_features)`
        targets: torch.Tensor
             Targets for all `deeprc.task_definitions.Target` instances, as returned by .get_targets(), as torch.Tensor
             of shape `(n_samples, n_target_features)`
        
        Returns
        ---------
        loss: torch.Tensor
            Loss for all `deeprc.task_definitions.Target` instances as torch.Tensor of shape
            `(n_samples, n_target_instances)`.
        """
        losses = torch.stack([l(raw_outputs[..., s], targets[..., s]) * w
                              for s, l, w in zip(self.__targets_slices__, self.__loss_functions__,
                                                 self.__task_weights__)])
        return losses
    
    def get_loss(self, raw_outputs: torch.Tensor, targets: torch.Tensor, ignore_missing_target_values: bool = True)\
            -> torch.Tensor:
        """Get combined loss of all `deeprc.task_definitions.Target` instances.
        
        The combined loss is computed as weighted sum of the individual task losses times their respective
        tasks-weights.
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network as torch.Tensor of shape
            `(n_samples, n_output_features)`
        targets: torch.Tensor
             Targets for all `deeprc.task_definitions.Target` instances, as returned by .get_targets(), as torch.Tensor
             of shape `(n_samples, n_target_features)`
        ignore_missing_target_values: bool
             If True, missing target values will be ignored for training. This can be useful if auxiliary tasks are not
             available for all samples but might increase the computation time per update.
        
        Returns
        ---------
        loss: torch.Tensor
            Combined loss for all `deeprc.task_definitions.Target` instances as scalar torch.Tensor of shape `()`.
            The combined loss is computed as weighted sum of the individual task losses times their respective
            tasks-weights.
        """
        losses_per_target_per_sample = self.get_losses(raw_outputs, targets)  # shape: (n_tasks, n_samples, 1)
        if ignore_missing_target_values:
            # Reduce sample dimension and omit NaNs from missing target entries
            loss_per_target = [losses_per_sample[~torch.isnan(losses_per_sample)].mean()
                               for losses_per_sample in losses_per_target_per_sample]
            # Sum up losses of different tasks, ignore NaNs (only relevant if all samples for a task are missing)
            loss_per_target = torch.stack(loss_per_target)
            loss = loss_per_target[~torch.isnan(loss_per_target)].sum()
        else:
            # Mean over samples, sum over tasks
            loss = losses_per_target_per_sample.mean(dim=1).sum()
        return loss
    
    def get_task_ids(self) -> tuple:
        """Get IDs for all `deeprc.task_definitions.Target` instances."""
        return self.__target_ids__
    
    def get_task_weights(self) -> np.ndarray:
        """Get task weights for all `deeprc.task_definitions.Target` instances. The combined loss is computed as
         weighted sum of the individual task losses times their respective tasks-weights."""
        return self.__task_weights__
    
    def get_n_output_features(self) -> int:
        """Get number of output features required for all `deeprc.task_definitions.Target` instances. This will be
        the number of output features used for the DeepRC network."""
        return self.__total_output_features__
    
    def get_scores(self, raw_outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Get scores for this task as dictionary
        
        Parameters
        ----------
        raw_outputs: torch.Tensor
             Raw output of the DeepRC network for this task as torch.Tensor of shape
            `(n_samples, self.n_output_features)`
        targets: torch.Tensor
             Targets for this task, as returned by .get_targets() as torch.Tensor of shape
             `(n_samples, self.n_output_features)`
        
        Returns
        ---------
        scores: dict
            Nested dictionary of format `{task_id: {score_id: score_value}}`, e.g.
            `{"binary_task_1": {"auc": 0.6, "bacc": 0.5, "f1": 0.2, "loss": 0.01}}`. The scores returned are computed
            using the .get_scores() methods of the individual target instances (e.g.
            `deeprc.task_definitions.BinaryTarget()`).
            See `deeprc/examples/` for examples.
        """
        scores = dict([(t.get_id(), t.get_scores(raw_outputs=raw_outputs[..., s], targets=targets[..., s]))
                       for s, t in zip(self.__targets_slices__, self.__targets__)])
        return scores
