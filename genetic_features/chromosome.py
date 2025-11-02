"""
Enhanced Chromosome Representation for Feature Selection
Supports multiple encoding schemes for genetic algorithms
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import random

class ChromosomeBase(ABC):
    """Base class for chromosome representations"""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.genes = None
        self.fitness = None
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize chromosome with random values"""
        pass
    
    @abstractmethod
    def get_selected_features(self) -> List[int]:
        """Return indices of selected features"""
        pass
    
    @abstractmethod
    def crossover(self, other: 'ChromosomeBase') -> Tuple['ChromosomeBase', 'ChromosomeBase']:
        """Perform crossover with another chromosome"""
        pass
    
    @abstractmethod
    def mutate(self, mutation_rate: float) -> None:
        """Mutate the chromosome"""
        pass
    
    def __len__(self) -> int:
        return self.n_features

class BinaryChromosome(ChromosomeBase):
    """Binary representation: [1, 0, 1, 0, 1] - classic approach"""
    
    def __init__(self, n_features: int, max_features: int = None):
        super().__init__(n_features)
        self.max_features = max_features or n_features // 2
        
    def initialize(self) -> None:
        """Initialize with random binary values, respecting max_features constraint"""
        self.genes = np.zeros(self.n_features, dtype=int)
        
        # Randomly select features up to max_features
        n_selected = random.randint(1, min(self.max_features, self.n_features))
        selected_indices = np.random.choice(self.n_features, n_selected, replace=False)
        self.genes[selected_indices] = 1
    
    def get_selected_features(self) -> List[int]:
        """Return indices where gene value is 1"""
        return np.where(self.genes == 1)[0].tolist()
    
    def crossover(self, other: 'BinaryChromosome') -> Tuple['BinaryChromosome', 'BinaryChromosome']:
        """Two-point crossover"""
        child1 = BinaryChromosome(self.n_features, self.max_features)
        child2 = BinaryChromosome(self.n_features, self.max_features)
        
        # Two-point crossover
        point1 = random.randint(0, self.n_features - 1)
        point2 = random.randint(point1, self.n_features - 1)
        
        child1.genes = np.copy(self.genes)
        child2.genes = np.copy(other.genes)
        
        child1.genes[point1:point2] = other.genes[point1:point2]
        child2.genes[point1:point2] = self.genes[point1:point2]
        
        # Ensure constraints are met
        child1._enforce_constraints()
        child2._enforce_constraints()
        
        return child1, child2
    
    def mutate(self, mutation_rate: float) -> None:
        """Flip bits with given probability"""
        for i in range(self.n_features):
            if random.random() < mutation_rate:
                self.genes[i] = 1 - self.genes[i]
        
        self._enforce_constraints()
    
    def _enforce_constraints(self) -> None:
        """Ensure chromosome respects max_features constraint"""
        selected_count = np.sum(self.genes)
        
        if selected_count == 0:
            # Select at least one feature randomly
            random_idx = random.randint(0, self.n_features - 1)
            self.genes[random_idx] = 1
        elif selected_count > self.max_features:
            # Remove excess features randomly
            selected_indices = np.where(self.genes == 1)[0]
            excess = selected_count - self.max_features
            to_remove = np.random.choice(selected_indices, excess, replace=False)
            self.genes[to_remove] = 0

class RealValuedChromosome(ChromosomeBase):
    """Real-valued representation: [0.8, 0.2, 0.9, 0.1] - threshold-based selection"""
    
    def __init__(self, n_features: int, threshold: float = 0.5, max_features: int = None):
        super().__init__(n_features)
        self.threshold = threshold
        self.max_features = max_features or n_features // 2
        
    def initialize(self) -> None:
        """Initialize with random real values between 0 and 1"""
        self.genes = np.random.random(self.n_features)
    
    def get_selected_features(self) -> List[int]:
        """Return indices where gene value > threshold, respecting max_features"""
        # Get all features above threshold
        candidates = np.where(self.genes > self.threshold)[0]
        
        if len(candidates) == 0:
            # If no features above threshold, select the highest valued one
            return [np.argmax(self.genes)]
        elif len(candidates) > self.max_features:
            # If too many features, select top max_features by value
            sorted_indices = np.argsort(self.genes)[::-1]
            return sorted_indices[:self.max_features].tolist()
        else:
            return candidates.tolist()
    
    def crossover(self, other: 'RealValuedChromosome') -> Tuple['RealValuedChromosome', 'RealValuedChromosome']:
        """Blend crossover (BLX-α)"""
        alpha = 0.5
        child1 = RealValuedChromosome(self.n_features, self.threshold, self.max_features)
        child2 = RealValuedChromosome(self.n_features, self.threshold, self.max_features)
        
        for i in range(self.n_features):
            min_val = min(self.genes[i], other.genes[i])
            max_val = max(self.genes[i], other.genes[i])
            range_val = max_val - min_val
            
            # Extend range by alpha
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            # Ensure values stay in [0, 1]
            lower = max(0, lower)
            upper = min(1, upper)
            
            child1.genes = np.random.uniform(lower, upper, self.n_features)
            child2.genes = np.random.uniform(lower, upper, self.n_features)
        
        return child1, child2
    
    def mutate(self, mutation_rate: float) -> None:
        """Gaussian mutation"""
        for i in range(self.n_features):
            if random.random() < mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.1)
                self.genes[i] = np.clip(self.genes[i] + noise, 0, 1)

class PermutationChromosome(ChromosomeBase):
    """Permutation representation: [2, 0, 4, 1, 3] - feature ranking order"""
    
    def __init__(self, n_features: int, selection_ratio: float = 0.3):
        super().__init__(n_features)
        self.selection_ratio = selection_ratio
        self.n_selected = max(1, int(n_features * selection_ratio))
        
    def initialize(self) -> None:
        """Initialize with random permutation of feature indices"""
        self.genes = np.random.permutation(self.n_features)
    
    def get_selected_features(self) -> List[int]:
        """Return first n_selected features from the permutation"""
        return self.genes[:self.n_selected].tolist()
    
    def crossover(self, other: 'PermutationChromosome') -> Tuple['PermutationChromosome', 'PermutationChromosome']:
        """Order crossover (OX)"""
        child1 = PermutationChromosome(self.n_features, self.selection_ratio)
        child2 = PermutationChromosome(self.n_features, self.selection_ratio)
        
        # Select crossover points
        point1 = random.randint(0, self.n_features - 1)
        point2 = random.randint(point1, self.n_features - 1)
        
        # Initialize children
        child1.genes = np.full(self.n_features, -1)
        child2.genes = np.full(self.n_features, -1)
        
        # Copy segments
        child1.genes[point1:point2] = self.genes[point1:point2]
        child2.genes[point1:point2] = other.genes[point1:point2]
        
        # Fill remaining positions
        self._fill_remaining(child1.genes, other.genes, point1, point2)
        self._fill_remaining(child2.genes, self.genes, point1, point2)
        
        return child1, child2
    
    def _fill_remaining(self, child_genes: np.ndarray, parent_genes: np.ndarray, 
                       point1: int, point2: int) -> None:
        """Fill remaining positions in order crossover"""
        used = set(child_genes[point1:point2])
        fill_pos = (point2) % self.n_features
        
        for gene in parent_genes:
            if gene not in used:
                while child_genes[fill_pos] != -1:
                    fill_pos = (fill_pos + 1) % self.n_features
                child_genes[fill_pos] = gene
    
    def mutate(self, mutation_rate: float) -> None:
        """Swap mutation"""
        if random.random() < mutation_rate:
            # Swap two random positions
            pos1, pos2 = random.sample(range(self.n_features), 2)
            self.genes[pos1], self.genes[pos2] = self.genes[pos2], self.genes[pos1]

class AdaptiveChromosome(ChromosomeBase):
    """Adaptive representation that switches between encoding schemes"""
    
    def __init__(self, n_features: int, feature_importance: np.ndarray = None):
        super().__init__(n_features)
        self.feature_importance = feature_importance
        self.encoding_type = self._select_encoding()
        self.chromosome = self._create_chromosome()
    
    def _select_encoding(self) -> str:
        """Select best encoding based on problem characteristics"""
        if self.n_features <= 20:
            return "binary"
        elif self.feature_importance is not None:
            return "permutation"  # Good when we have importance info
        else:
            return "real_valued"
    
    def _create_chromosome(self) -> ChromosomeBase:
        """Create appropriate chromosome type"""
        if self.encoding_type == "binary":
            return BinaryChromosome(self.n_features)
        elif self.encoding_type == "real_valued":
            return RealValuedChromosome(self.n_features)
        else:  # permutation
            return PermutationChromosome(self.n_features)
    
    def initialize(self) -> None:
        """Initialize underlying chromosome"""
        if self.encoding_type == "permutation" and self.feature_importance is not None:
            # Initialize based on feature importance
            sorted_indices = np.argsort(self.feature_importance)[::-1]
            noise = np.random.normal(0, 0.1, self.n_features)
            self.chromosome.genes = sorted_indices + noise
            self.chromosome.genes = np.argsort(np.argsort(self.chromosome.genes))
        else:
            self.chromosome.initialize()
    
    def get_selected_features(self) -> List[int]:
        """Delegate to underlying chromosome"""
        return self.chromosome.get_selected_features()
    
    def crossover(self, other: 'AdaptiveChromosome') -> Tuple['AdaptiveChromosome', 'AdaptiveChromosome']:
        """Crossover with same encoding type"""
        if self.encoding_type != other.encoding_type:
            # Convert to same type or use binary as default
            return self._convert_and_crossover(other)
        
        child1_chr, child2_chr = self.chromosome.crossover(other.chromosome)
        
        child1 = AdaptiveChromosome(self.n_features, self.feature_importance)
        child2 = AdaptiveChromosome(self.n_features, self.feature_importance)
        child1.chromosome = child1_chr
        child2.chromosome = child2_chr
        
        return child1, child2
    
    def _convert_and_crossover(self, other: 'AdaptiveChromosome') -> Tuple['AdaptiveChromosome', 'AdaptiveChromosome']:
        """Convert to binary and perform crossover"""
        # Convert both to binary representation
        binary1 = BinaryChromosome(self.n_features)
        binary2 = BinaryChromosome(self.n_features)
        
        # Set genes based on selected features
        selected1 = self.get_selected_features()
        selected2 = other.get_selected_features()
        
        binary1.genes = np.zeros(self.n_features, dtype=int)
        binary2.genes = np.zeros(self.n_features, dtype=int)
        
        binary1.genes[selected1] = 1
        binary2.genes[selected2] = 1
        
        # Perform crossover
        child1_chr, child2_chr = binary1.crossover(binary2)
        
        # Create adaptive children
        child1 = AdaptiveChromosome(self.n_features, self.feature_importance)
        child2 = AdaptiveChromosome(self.n_features, self.feature_importance)
        child1.chromosome = child1_chr
        child2.chromosome = child2_chr
        
        return child1, child2
    
    def mutate(self, mutation_rate: float) -> None:
        """Delegate to underlying chromosome"""
        self.chromosome.mutate(mutation_rate)

# Factory function for easy chromosome creation
def create_chromosome(encoding_type: str, n_features: int, **kwargs) -> ChromosomeBase:
    """Factory function to create chromosomes"""
    if encoding_type == "binary":
        return BinaryChromosome(n_features, kwargs.get('max_features'))
    elif encoding_type == "real_valued":
        return RealValuedChromosome(n_features, kwargs.get('threshold', 0.5), 
                                  kwargs.get('max_features'))
    elif encoding_type == "permutation":
        return PermutationChromosome(n_features, kwargs.get('selection_ratio', 0.3))
    elif encoding_type == "adaptive":
        return AdaptiveChromosome(n_features, kwargs.get('feature_importance'))
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
