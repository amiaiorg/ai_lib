
import torch
import torch.nn as nn
import numpy as np

class ArchetypeBlender:
    def __init__(self, num_archetypes, feature_dim):
        self.num_archetypes = num_archetypes
        self.feature_dim = feature_dim
        self.archetypes = {}
        self.blending_network = nn.Sequential(
            nn.Linear(num_archetypes * feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
    def add_archetype(self, name, features):
        """Add a new archetype with its characteristic features"""
        if len(features) != self.feature_dim:
            raise ValueError(f"Features must have dimension {self.feature_dim}")
        self.archetypes[name] = torch.tensor(features, dtype=torch.float32)
        
    def blend_archetypes(self, weights):
        """Blend multiple archetypes based on weights"""
        if len(weights) != len(self.archetypes):
            raise ValueError("Weights must match number of archetypes")
            
        combined_features = []
        for name, w in zip(self.archetypes.keys(), weights):
            combined_features.append(self.archetypes[name] * w)
            
        combined = torch.cat(combined_features)
        return self.blending_network(combined)
    
    def get_harmonic_blend(self, context):
        """Generate context-aware archetype blend"""
        weights = torch.softmax(torch.randn(len(self.archetypes)), dim=0)
        return self.blend_archetypes(weights)

# Example usage
feature_dim = 64
blender = ArchetypeBlender(num_archetypes=3, feature_dim=feature_dim)

# Add some example archetypes
blender.add_archetype("wisdom", torch.randn(feature_dim))
blender.add_archetype("nurture", torch.randn(feature_dim))
blender.add_archetype("guidance", torch.randn(feature_dim))
