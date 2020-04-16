# -*- coding: utf-8 -*-
"""
Network architectures
"""
import numpy as np
import torch
import torch.nn as nn
import torch.jit as jit
from typing import List


def compute_position_features(max_seq_len, seq_lens, dtype=np.float16):
    """Compute position features"""
    sequences = np.zeros((max_seq_len+1, max_seq_len, 3), dtype=dtype)
    half_seq_lens = np.asarray(np.ceil(seq_lens / 2.), dtype=np.int)
    for i in range(len(sequences)):
        sequence, seq_len, half_seq_len = sequences[i], seq_lens[i], half_seq_lens[i]
        sequence[:seq_len, -1] = np.abs(0.5 - np.linspace(1.0, 0, num=seq_len)) * 2.
        sequence[:half_seq_len, -3] = sequence[:half_seq_len, -1]
        sequence[half_seq_len:seq_len, -2] = sequence[half_seq_len:seq_len, -1]
        sequence[:seq_len, -1] = 1. - sequence[:seq_len, -1]
    return sequences


class SequenceEmbeddingCNN(nn.Module):
    def __init__(self, n_input_features: int, kernel_size: int = 9, n_kernels: int = 32, n_additional_convs: int = 0):
        """Sequence embedding using 1D-CNN (`h_1` in paper)
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        kernel_size : int
            Size of 1D-CNN kernels
        n_kernels : int
            Number of 1D-CNN kernels in each layer
        n_additional_convs : int
            Number of additional 1D-CNN layers after first layer
        """
        super(SequenceEmbeddingCNN, self).__init__()
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.n_additional_convs = n_additional_convs
        
        # AA/position CNN layer
        self.conv_aas = nn.Conv1d(in_channels=n_input_features, out_channels=self.n_kernels,
                                  kernel_size=self.kernel_size, bias=True)
        self.conv_aas.weight.data.normal_(0.0, np.sqrt(1 / np.prod(self.conv_aas.weight.shape)))
        n_output_channels = self.n_kernels
        
        # Additional CNN layers
        additional_convs = []
        for i in range(self.n_additional_convs):
            add_conv = nn.Conv1d(in_channels=n_output_channels, out_channels=self.n_kernels,
                                 kernel_size=3, bias=True)
            add_conv.weight.data.normal_(0.0, np.sqrt(1 / np.prod(add_conv.weight.shape)))
            additional_convs.append(add_conv)
            additional_convs.append(nn.SELU(inplace=True))
            n_output_channels = self.n_kernels
        
        self.additional_convs = torch.nn.Sequential(*additional_convs)
        self.n_output_channels = n_output_channels
    
    def forward(self, inputs):
        """Apply sequence embedding CNN to inputs.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_sequences, n_in_features, n_sequence_positions), where n_in_features is
            `n_aa_features + n_position_features = 20 + 3 = 23`.
        
        Returns
        ---------
        max_conv_acts: torch.Tensor
            Sequences embedded to tensor of shape (n_sequences, n_kernels)
        """
        # Calculate activations for AAs and positions
        conv_acts = torch.selu(self.conv_aas(inputs))
        # Apply additional conv. layers
        conv_acts = self.additional_convs(conv_acts)
        # Take maximum over sequence positions (-> 1 output per kernel per sequence)
        max_conv_acts, _ = conv_acts.max(dim=-1)
        return max_conv_acts


class AttentionNetwork(nn.Module):
    def __init__(self, n_input_features: int, n_layers: int = 2, n_units: int = 32):
        """Attention network (`h_2` in paper) as fully connected network
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_layers : int
            Number of attention layers to compute keys
        n_units : int
            Number of units in each attention layer
        """
        super(AttentionNetwork, self).__init__()
        self.n_attention_layers = n_layers
        self.n_units = n_units
        
        fc_attention = []
        for _ in range(self.n_attention_layers):
            att_linear = nn.Linear(n_input_features, self.n_units)
            att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
            fc_attention.append(att_linear)
            fc_attention.append(nn.SELU())
            n_input_features = self.n_units
        
        att_linear = nn.Linear(n_input_features, 1)
        att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
        fc_attention.append(att_linear)
        self.attention_nn = torch.nn.Sequential(*fc_attention)
    
    def forward(self, inputs):
        """Apply single-head attention network.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_sequences, n_in_features)
        
        Returns
        ---------
        attention_weights: torch.Tensor
            Attention weights for sequences as tensor of shape (n_sequences, 1)
        """
        attention_weights = self.attention_nn(inputs)
        return attention_weights


class OutputNetwork(nn.Module):
    def __init__(self, n_input_features, n_output_features: int = 1, n_layers: int = 0, n_units: int = 32):
        """Output network (`o` in paper) as fully connected network
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_output_features : int
            Number of output features (1 for binary classification)
        n_layers : int
            Number of layers in output network (in addition to final output layer)
        n_units : int
            Number of units in each attention layer
        """
        super(OutputNetwork, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        
        output_network = []
        for _ in range(self.n_layers):
            o_linear = nn.Linear(n_input_features, self.n_units)
            o_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(o_linear.weight.shape)))
            output_network.append(o_linear)
            output_network.append(nn.SELU())
            n_input_features = self.n_units
        
        o_linear = nn.Linear(n_input_features, n_output_features)
        o_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(o_linear.weight.shape)))
        output_network.append(o_linear)
        self.output_nn = torch.nn.Sequential(*output_network)
    
    def forward(self, inputs):
        """Apply output network.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_bags, n_in_features).
        
        Returns
        ---------
        prediction: torch.Tensor
            Prediction as tensor of shape (n_bags, n_output_features).
        """
        predictions = self.output_nn(inputs)
        return predictions


class DeepRC(nn.Module):
    def __init__(self, n_input_features, n_output_features, max_seq_len,
                 kernel_size: int = 9, n_kernels: int = 32, n_additional_convs: int = 0,
                 n_attention_network_layers: int = 2, n_attention_network_units: int = 32,
                 n_output_network_layers: int = 0, n_output_network_units: int = 32,
                 consider_seq_counts: bool = False, add_positional_information: bool = True,
                 sequence_reduction_fraction: float = 0.1, reduction_mb_size: int = 5e4,
                 device: torch.device = torch.device('cuda:0')):
        """DeepRC network as described in paper
        
        Apply reduce_and_stack_minibatch() to reduce number of sequences by network_config['sequence_reduction_fraction']
        based on their attention weights and stack/concatenate the bags to a minibatch.
        Then apply forward() to the minibatch to compute the predictions.
        
        Reduction of sequences per bag is performed using minibatches of network_config['reduction_mb_size'] sequences
        to compute the attention weights.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        consider_seq_counts : bool
            Scale inputs by sequence counts?
            (Only used for CMV-like datasets with high sequence count values.)
        add_positional_information : bool
            Concatenate position features to inputs?
        sequence_reduction_fraction : float
            Fraction of number of sequences to which to reduce the number of
            sequences per bag based on attention weights. Has to be in range [0,1].
        reduction_mb_size : int
            Reduction of sequences per bag is performed using minibatches of
            `reduction_mb_size` sequences to compute the attention weights.
        kernel_size : int
            Size of 1D-CNN kernels.
        n_kernels : int
            Number of 1D-CNN kernels in each layer
        n_additional_convs : int
            Number of additional 1D-CNN layers after first layer
        n_attention_network_layers : int
            Number of attention layers to compute keys
        n_attention_network_units : int
            Number of units in each attention layer
        n_output_network_layers : int
            Number of layers in output network (in addition to final output layer)
        n_output_network_units : int
            Number of units in each attention layer
        device : torch.devide
            Device to perform computations on
        """
        super(DeepRC, self).__init__()
        self.n_input_features = n_input_features
        self.n_outputs = n_output_features
        self.max_seq_len = max_seq_len
        self.device = device
        self.seq_counts = consider_seq_counts
        self.add_positional_information = add_positional_information
        self.sequence_reduction_fraction = sequence_reduction_fraction
        self.reduction_mb_size = int(reduction_mb_size)
        
        # 16 bit sequence embedding network (h_1)
        self.sequence_embedding_16bit = SequenceEmbeddingCNN(
                n_input_features, kernel_size=kernel_size, n_kernels=n_kernels,
                n_additional_convs=n_additional_convs).to(device=device, dtype=torch.float16)
        
        # Attention network (h_2)
        self.attention_nn = AttentionNetwork(self.sequence_embedding_16bit.n_output_channels,
                                             n_layers=n_attention_network_layers,
                                             n_units=n_attention_network_units)
        
        # Output network (o)
        self.output_nn = OutputNetwork(self.sequence_embedding_16bit.n_output_channels,
                                       n_layers=n_output_network_layers, n_units=n_output_network_units)
        
        # Pre-compute position features for all possible sequence lengths
        position_features = compute_position_features(max_seq_len=max_seq_len, seq_lens=np.arange(max_seq_len+1))
        self.position_features = torch.from_numpy(position_features).to(device=device, dtype=torch.float16).detach()
    
    def reduce_and_stack_minibatch(self, labels_list, aa_indices_list, sequence_lengths_list, counts_per_sequence_list):
        """ Apply attention-based reduction of number of sequences per bag and stacked/concatenated bags to minibatch.
        
        Reduces sequences per bag `d_k` to top `d_k*network_config['sequence_reduction_fraction']` important sequences,
        sorted descending by importance.
        Reduction is performed using minibatches of network_config['reduction_mb_size'] sequences.
        Bags are then stacked/concatenated to one minibatch.
        
        Parameters
        ----------
        labels_list: list of torch.Tensor
            Labels of bags as list of tensors of shapes (n_classes,)
        aa_indices_list: list of torch.Tensor
            AA indices of bags as list of int8 tensors of shape (n_sequences, n_sequence_positions) = (d_k, d_l)
        sequence_lengths_list: list of torch.Tensor
            Sequences lengths of bags as tensors of dtype torch.long and shape (n_sequences,) = (d_k,)
        counts_per_sequence_list: list of torch.Tensor
            Sequences counts per bag as tensors of shape (n_sequences,) = (d_k,).
            The sequences counts are the log(max(counts, 1)).
        
        Returns
        ----------
        mb_labels: list of torch.Tensor
            Labels of bags as tensor of shape (n_bags, n_classes)
        mb_reduced_inputs: torch.Tensor
            Top `n_sequences*network_config['sequence_reduction_fraction']` important sequences per bag,
            as tensor of shape (n_bags*n_reduced_sequences, n_input_features, n_sequence_positions),
            where `n_reduced_sequences=n_sequences*network_config['sequence_reduction_fraction']`
        mb_reduced_sequence_lengths: torch.Tensor
            Sequences lengths of `reduced_inputs` per bag as tensor of dtype torch.long and shape
            (n_bags*n_reduced_sequences,),
            where `n_reduced_sequences=n_sequences*network_config['sequence_reduction_fraction']`
        mb_n_sequences: torch.Tensor
            Number of sequences per bag as tensor of dtype torch.long and shape (n_bags,)
        """
        with torch.no_grad():
            # Move tensors to device
            aa_indices_list = [t.to(self.device) for t in aa_indices_list]
            max_mb_seq_len = max(t.max() for t in sequence_lengths_list)
            sequence_lengths_list = [t.to(self.device) for t in sequence_lengths_list]
            
            # Compute features (turn into 1-hot AAs and add position features)
            inputs_list = [self.__compute_features_ncl__(aa_indices, seq_lens, max_mb_seq_len, counts_per_sequence)
                           for aa_indices, seq_lens, counts_per_sequence
                           in zip(aa_indices_list, sequence_lengths_list, counts_per_sequence_list)]
            
            # Reduce number of sequences (apply __reduce_sequences_for_bag__ separately to all bags in mb)
            reduced_inputs, reduced_sequence_lengths = \
                list(zip(*[self.__reduce_sequences_for_bag__(inp, seq_lens)
                           for inp, seq_lens
                           in zip(inputs_list, sequence_lengths_list)]))
            
            # Stack bags in minibatch to tensor
            mb_labels = torch.stack(labels_list, dim=0).to(device=self.device)
            mb_reduced_sequence_lengths = torch.cat(reduced_sequence_lengths, dim=0)
            mb_reduced_inputs = torch.cat(reduced_inputs, dim=0)
            mb_n_sequences = torch.tensor([len(rsl) for rsl in reduced_sequence_lengths], dtype=torch.long,
                                          device=self.device)
        
        return mb_labels, mb_reduced_inputs, mb_reduced_sequence_lengths, mb_n_sequences
    
    def forward(self, inputs_flat, n_sequences_per_bag):
        """ Apply DeepRC (see Fig.2 in paper)
        
        Parameters
        ----------
        inputs_flat: torch.Tensor
            Concatenated bags as input of shape (n_bags*n_sequences_per_bag, n_input_features, n_sequence_positions) =
            (N*d_k, 20+3, d_l)
        n_sequences_per_bag: torch.Tensor
            Number of sequences per bag as tensor of dtype torch.long and shape (n_bags,)
        
        Returns
        ----------
        predictions: torch.Tensor
            Prediction for bags of shape (n_bags, n_outputs)
        """
        # Get sequence embedding h_1 for all bags in mb (shape: (d_k, d_v))
        mb_emb_seqs = self.sequence_embedding_16bit(inputs_flat).to(dtype=torch.float32)
        
        # Calculate attention weights h_2 before softmax function for all bags in mb (shape: (d_k, 1))
        mb_attention_weights = self.attention_nn(mb_emb_seqs)
        
        # Compute representation per bag (N times shape (d_v,))
        mb_emb_seqs_after_attention = []
        start_i = 0
        for n_seqs in n_sequences_per_bag:
            # Get sequence embedding h_1 for single bag (shape: (n_sequences_per_bag, d_v))
            attention_weights = mb_attention_weights[start_i:start_i+n_seqs]
            # Get attention weights for single bag (shape: (n_sequences_per_bag, 1))
            emb_seqs = mb_emb_seqs[start_i:start_i+n_seqs]
            # Calculate attention activations (softmax over n_sequences_per_bag) (shape: (n_sequences_per_bag, 1))
            attention_weights = torch.softmax(attention_weights, dim=0)
            # Apply attention weights to sequence features (shape: (n_sequences_per_bag, d_v))
            emb_seqs_after_attention = emb_seqs * attention_weights
            # Compute weighted sum over sequence features after attention (format: (d_v,))
            mb_emb_seqs_after_attention.append(emb_seqs_after_attention.sum(dim=0))
            start_i += n_seqs
        
        # Stack representations of bags (shape (N, d_v))
        emb_seqs_after_attention = torch.stack(mb_emb_seqs_after_attention, dim=0)
        
        # Calculate predictions (shape (N, n_outputs))
        predictions = self.output_nn(emb_seqs_after_attention)
        predictions = predictions.squeeze(dim=-1)  # squeeze since binary single-task
        
        return predictions
    
    def __compute_features_ncl__(self, aa_indices, seq_lens, max_mb_seq_len, counts_per_sequence):
        """Compute one-hot AA features + position features with shape (n_sequences, n_features, sequence_length)
        from AA indices
        """
        features_one_hot = self.__compute_features_nlc__(aa_indices, seq_lens, max_mb_seq_len, counts_per_sequence)
        features_one_hot = torch.transpose(features_one_hot, 1, 2)
        return features_one_hot
    
    def __compute_features_nlc__(self, aa_indices, seq_lens, max_mb_seq_len, counts_per_sequence):
        """Compute one-hot AA features + position features with shape (n_sequences, sequence_length, n_features)
        from AA indices
        """
        n_features = 20 + 3 * self.add_positional_information  # 20 features if used without position features
        # Send indices to device
        aa_indices = aa_indices.to(dtype=torch.long, device=self.device)
        seq_lens = seq_lens.to(dtype=torch.long, device=self.device)
        # Only send sequence counts to device, if using sequence counts
        if self.seq_counts:
            counts_per_sequence = counts_per_sequence.to(dtype=torch.float16, device=self.device)
        # Allocate tensor for one-hot AA features + position features
        features_one_hot_shape = (aa_indices.shape[0], max_mb_seq_len, n_features)
        features_one_hot_padded = torch.zeros(size=features_one_hot_shape, dtype=torch.float16, device=self.device)
        # Set one-hot AA features
        features_one_hot = features_one_hot_padded[:, :aa_indices.shape[1]]
        features_one_hot = features_one_hot.reshape((-1, n_features))
        features_one_hot[torch.arange(features_one_hot.shape[0]), aa_indices.reshape((-1))] = 1.
        # Set padded sequence-parts to 0 (aa_indices == -1 marks the padded positions)
        features_one_hot[aa_indices.reshape((-1)) == -1, -1] = 0.
        features_one_hot = features_one_hot.reshape((aa_indices.shape[0], aa_indices.shape[1], n_features))
        features_one_hot_padded[:, :aa_indices.shape[1], :] = features_one_hot
        # Scale by sequence counts
        if self.seq_counts:
            features_one_hot_padded = features_one_hot_padded * counts_per_sequence[:, None, None]
        # Add position information
        if self.add_positional_information:
            features_one_hot_padded[:, :aa_indices.shape[1], -3:] = \
                self.position_features[seq_lens, :aa_indices.shape[1]]
        # Perform normalization to std=~1
        if not self.seq_counts:
            # Use pre-computed std values if not using sequence counts
            if self.add_positional_information:
                features_one_hot_padded = features_one_hot_padded / 0.2817713347133853
            else:
                features_one_hot_padded = features_one_hot_padded / 0.21794494717703364
        else:
            features_one_hot_padded = features_one_hot_padded / features_one_hot_padded.std()
        return features_one_hot_padded
    
    def __reduce_sequences_for_bag__(self, inputs, sequence_lengths):
        """ Reduces sequences to top `n_sequences*network_config['sequence_reduction_fraction']` important sequences,
        sorted descending by importance. Reduction is performed using minibatches of network_config['reduction_mb_size']
        sequences.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Input of shape (n_sequences, n_input_features, n_sequence_positions) = (d_k, 20+3, d_l)
        sequence_lengths: torch.Tensor
            Sequences lengths as tensor of dtype torch.long and shape (n_sequences,) = (d_k,)
        
        Returns
        ----------
        reduced_inputs: torch.Tensor
            Top `n_sequences*network_config['sequence_reduction_fraction']` important sequences,
            sorted descending by importance as tensor of shape
            (n_reduced_sequences, n_input_features, n_sequence_positions),
            where `n_reduced_sequences=n_sequences*network_config['sequence_reduction_fraction']`
        reduced_sequence_lengths: torch.Tensor
            Sequences lengths of `reduced_inputs` as tensor of dtype torch.long and shape (n_reduced_sequences,),
            where `n_reduced_sequences=n_sequences*network_config['sequence_reduction_fraction']`
        """
        if self.sequence_reduction_fraction != 1.0:
            # Get number of sequences to reduce to
            n_reduced_sequences = int(sequence_lengths.shape[0] * self.sequence_reduction_fraction)
            # Get number of minibatches for reduction
            n_mbs = int(np.ceil(inputs.shape[0] / self.reduction_mb_size))
            mb_is = torch.arange(start=0, end=n_mbs, dtype=torch.int)
            
            # Calculate attention weights for sequences (loop over minibatch of sequences)
            attention_acts = torch.jit.annotate(List[torch.Tensor], [])
            for mb_i in mb_is.unbind(dim=0):
                # Get inputs for current minibatch
                inputs_mb = inputs[mb_i*self.reduction_mb_size:(mb_i+1)*self.reduction_mb_size].to(device=self.device,
                                                                                                   dtype=torch.float16)
                
                # Get sequence embedding (h_1)
                emb_seqs = self.sequence_embedding_16bit(inputs_mb).to(dtype=torch.float32)
                
                # Calculate attention weights before softmax (h_2)
                attention_acts.append(self.attention_nn(emb_seqs).squeeze(dim=-1))
            
            # Concatenate attention weights for all sequences
            attention_acts = torch.cat(attention_acts, dim=0)
            
            # Get indices of k sequences with highest attention weights
            _, used_sequences = torch.topk(attention_acts, n_reduced_sequences, dim=0, largest=True, sorted=True)
            
            # Get top k sequences and sequence lengths
            reduced_inputs = inputs[used_sequences.to(device=self.device)].detach().to(device=self.device,
                                                                                       dtype=torch.float16)
            reduced_sequence_lengths = \
                sequence_lengths[used_sequences.to(device=self.device)].detach().to(device=self.device,
                                                                                    dtype=torch.float16)
        else:
            with torch.no_grad():
                reduced_inputs = inputs.detach().to(device=self.device, dtype=torch.float16)
                reduced_sequence_lengths = sequence_lengths.detach().to(device=self.device, dtype=torch.float16)
        
        return reduced_inputs, reduced_sequence_lengths
