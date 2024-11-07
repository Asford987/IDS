import tensorflow as tf


class MixtureOfExpertsTopK(tf.keras.Model):
    def __init__(self, experts_paths, gate_path, k=3, trainable_gate=True, trainable_experts=False):
        # Load expert models
        self.experts = list(range(len(experts_paths)))
        for expert_index, expert_path in experts_paths.items():
            self.experts[expert_index] = tf.keras.model.load_model(expert_path)
            self.experts[expert_index].trainable = trainable_experts
        
        # Load gate model
        self.gate = tf.keras.model.load_model(gate_path)
        self.gate.trainable = trainable_gate
        
        # Compute number of experts
        self.num_experts = len(self.experts)
    
    def call(self, inputs, k=2):
        # Get gate outputs (logits)
        gate_logits = self.gate(inputs)  # Shape: [batch_size, num_experts]
        
        # Get top_k experts for each input
        top_k_values, top_k_indices = tf.math.top_k(gate_logits, k=k, sorted=True)  # Shapes: [batch_size, k]
        
        # Create a mask for the top_k experts
        mask = tf.one_hot(top_k_indices, depth=self.num_experts, dtype=tf.bool)  # Shape: [batch_size, k, num_experts]
        mask = tf.reduce_any(mask, axis=1)  # Shape: [batch_size, num_experts]
        
        # Compute outputs for all experts (inefficient if experts are heavy)
        all_expert_outputs = [expert(inputs) for expert in self.experts]  # List of tensors [batch_size, output_dim]
        all_expert_outputs = tf.stack(all_expert_outputs, axis=1)  # Shape: [batch_size, num_experts, output_dim]
        
        # Mask out non-top_k expert outputs
        masked_expert_outputs = tf.where(
            tf.expand_dims(mask, axis=2),
            all_expert_outputs,
            tf.zeros_like(all_expert_outputs)
        )  # Shape: [batch_size, num_experts, output_dim]
        
        # Apply softmax to top_k gate outputs
        gate_weights = tf.nn.softmax(gate_logits, axis=1)  # Shape: [batch_size, num_experts]
        
        # Zero out weights for non-top_k experts
        masked_gate_weights = tf.where(
            mask,
            gate_weights,
            tf.zeros_like(gate_weights)
        )  # Shape: [batch_size, num_experts]
        
        # Multiply expert outputs by weights
        weighted_expert_outputs = masked_expert_outputs * tf.expand_dims(masked_gate_weights, axis=2)
        
        # Sum over experts
        aggregated_output = tf.reduce_sum(weighted_expert_outputs, axis=1)  # Shape: [batch_size, output_dim]
        
        return aggregated_output # returns logits
    
    
class MixtureOfExpertsSoft(tf.keras.Model):
    def __init__(self, experts_paths, gate_path, trainable_gate=True, trainable_experts=False):
        # Load expert models
        self.experts = list(range(len(experts_paths)))
        for expert_index, expert_path in experts_paths.items():
            self.experts[expert_index] = tf.keras.model.load_model(expert_path)
            self.experts[expert_index].trainable = trainable_experts
        
        # Load gate model
        self.gate = tf.keras.model.load_model(gate_path)
        self.gate.trainable = trainable_gate
        
        # Compute number of experts
        self.num_experts = len(self.experts)
    
    def call(self, inputs):
        # Compute experts outputs
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs_tensor = tf.stack(expert_outputs, axis=0)  # Shape: [num_experts, batch_size, num_classes]
        
        # Compute gate outputs
        gate_weights = self.gate(inputs)
        
        # Expand gate weights for broadcasting
        gate_weights_expanded = tf.expand_dims(tf.transpose(gate_weights), axis=2)  # Shape: [num_experts, batch_size, 1]
    
        # Multiply expert probabilities by gate weights
        weighted_expert_outputs = expert_outputs_tensor * gate_weights_expanded  # Shape: [num_experts, batch_size, num_classes]
    
        # Sum over experts
        aggregated_output = tf.reduce_sum(weighted_expert_outputs, axis=0)  # Shape: [batch_size, num_classes]
    
        return aggregated_output # return logits