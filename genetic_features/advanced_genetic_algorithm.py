import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from .chromosome import create_chromosome, AdaptiveChromosome
import warnings
warnings.filterwarnings('ignore')

class AdvancedGeneticFeatureSelector:
 
    
    def __init__(self, X, y, 
                 population_size=50, 
                 generations=20, 
                 cx_prob=0.7, 
                 mut_prob=0.2,
                 tournament_size=3,
                 elite_size=5,
                 max_features_ratio=0.5,
                 cv_folds=3,
                 scoring_method='balanced',
                 early_stopping_patience=5,
                 sample_size=None,
                 chromosome_type='adaptive'):

        
        self.original_X = X
        self.original_y = y
        self.n_features = X.shape[1]
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.max_features_ratio = max_features_ratio
        self.cv_folds = cv_folds
        self.scoring_method = scoring_method
        self.early_stopping_patience = early_stopping_patience
        self.chromosome_type = chromosome_type
        
        # Process large datasets
        self._prepare_data(sample_size)
        
        # Calculate feature importance to guide the algorithm
        self._calculate_feature_importance()
        
        # Setup genetic algorithm
        self._setup_genetic_algorithm()
        
        # Tracking variables
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.feature_selection_history = []
        self.generations_without_improvement = 0
        self.best_global_fitness = 0
        
    def _prepare_data(self, sample_size):
        """Prepare data for fast processing"""
        
        # Sample from large datasets if necessary
        if sample_size and len(self.original_y) > sample_size:
            from sklearn.model_selection import train_test_split
            self.X, _, self.y, _ = train_test_split(
                self.original_X, self.original_y, 
                train_size=sample_size, 
                stratify=self.original_y, 
                random_state=42
            )
            print(f"Sampled {sample_size} samples from {len(self.original_y)} total samples")
        else:
            self.X = self.original_X.copy()
            self.y = self.original_y.copy()
        
        # Normalize data
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        # Setup cross-validation
        self.cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
    def _calculate_feature_importance(self):
        """Calculate feature importance to guide initial selection"""
        
        # Use Mutual Information to calculate importance
        self.feature_importance = mutual_info_classif(self.X, self.y, random_state=42)
        
        # Normalize importance between 0 and 1
        if self.feature_importance.max() > 0:
            self.feature_importance = self.feature_importance / self.feature_importance.max()
        
        # Rank features by importance
        self.feature_ranking = np.argsort(self.feature_importance)[::-1]
        
    def _setup_genetic_algorithm(self):
        """Setup genetic algorithm components"""
        
        # Create object types
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Setup toolbox
        self.toolbox = base.Toolbox()
        
        # Register operations with advanced chromosome support
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._chromosome_crossover)
        self.toolbox.register("mutate", self._chromosome_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
    def _create_individual(self):
        """Create an individual using advanced chromosome representations"""
        max_features = int(self.max_features_ratio * self.n_features)
        
        # Create chromosome with feature importance for adaptive types
        if self.chromosome_type == 'adaptive':
            chromosome = create_chromosome(
                self.chromosome_type, 
                self.n_features, 
                feature_importance=self.feature_importance
            )
        else:
            chromosome = create_chromosome(
                self.chromosome_type, 
                self.n_features, 
                max_features=max_features
            )
        
        chromosome.initialize()
        
        # Convert to DEAP individual format
        individual = creator.Individual()
        individual.chromosome = chromosome
        selected_features = chromosome.get_selected_features()
        
        # Convert to binary representation for compatibility
        binary_repr = [0] * self.n_features
        for idx in selected_features:
            if 0 <= idx < self.n_features:
                binary_repr[idx] = 1
        
        individual[:] = binary_repr
        return individual
    
    def _chromosome_crossover(self, ind1, ind2):
        """Perform crossover using chromosome-specific methods"""
        if hasattr(ind1, 'chromosome') and hasattr(ind2, 'chromosome'):
            # Use chromosome-specific crossover
            child1_chr, child2_chr = ind1.chromosome.crossover(ind2.chromosome)
            
            # Update individuals
            self._update_individual_from_chromosome(ind1, child1_chr)
            self._update_individual_from_chromosome(ind2, child2_chr)
            
            # Apply feature limit enforcement
            self._enforce_feature_limit_on_individual(ind1)
            self._enforce_feature_limit_on_individual(ind2)
        else:
            # Fallback to smart crossover
            self._smart_crossover(ind1, ind2)
        
        return ind1, ind2
    
    def _chromosome_mutation(self, individual):
        """Perform mutation using chromosome-specific methods"""
        if hasattr(individual, 'chromosome'):
            # Use chromosome-specific mutation
            individual.chromosome.mutate(self.mut_prob)
            self._update_individual_from_chromosome(individual, individual.chromosome)
            self._enforce_feature_limit_on_individual(individual)
        else:
            # Fallback to smart mutation
            self._smart_mutation(individual)
        
        return individual,
    
    def _update_individual_from_chromosome(self, individual, chromosome):
        """Update DEAP individual from chromosome representation"""
        individual.chromosome = chromosome
        selected_features = chromosome.get_selected_features()
        
        # Convert to binary representation
        binary_repr = [0] * self.n_features
        for idx in selected_features:
            if 0 <= idx < self.n_features:
                binary_repr[idx] = 1
        
        individual[:] = binary_repr
    
    def _enforce_feature_limit_on_individual(self, individual):
        """Enforce feature limits on individual using chromosome methods"""
        if hasattr(individual, 'chromosome'):
            # Let the chromosome handle its own constraints
            pass  # Chromosomes handle their own constraints
        else:
            # Fallback to original method
            self._enforce_feature_limit(individual)

    def _intelligent_feature_init(self):
        """Intelligent feature initialization based on importance"""
        
        # Higher probability for selecting important features
        prob = np.random.random()
        feature_idx = np.random.randint(0, self.n_features)
        
        # More important features have higher selection probability
        importance_prob = 0.3 + 0.7 * self.feature_importance[feature_idx]
        
        return 1 if prob < importance_prob else 0
    
    def _smart_crossover(self, ind1, ind2):
        """Smart crossover that preserves important features"""
        
        # Apply normal crossover
        tools.cxTwoPoint(ind1, ind2)
        
        # Ensure feature limit is not exceeded
        self._enforce_feature_limit(ind1)
        self._enforce_feature_limit(ind2)
        
        return ind1, ind2
    
    def _smart_mutation(self, individual):
        """Smart mutation that favors important features"""
        
        # Normal mutation
        tools.mutFlipBit(individual, indpb=0.1)
        
        # Additional mutation for important features
        for i in self.feature_ranking[:int(0.2 * self.n_features)]:
            if np.random.random() < 0.1:
                individual[i] = 1 - individual[i]
        
        # Ensure limits
        self._enforce_feature_limit(individual)
        
        return individual,
    
    def _enforce_feature_limit(self, individual):
        """Enforce maximum limit for selected features"""
        
        max_features = int(self.max_features_ratio * self.n_features)
        selected_count = sum(individual)
        
        if selected_count > max_features:
            # Remove least important features
            selected_indices = [i for i, val in enumerate(individual) if val == 1]
            importance_scores = [(i, self.feature_importance[i]) for i in selected_indices]
            importance_scores.sort(key=lambda x: x[1])
            
            # Remove excess features
            to_remove = selected_count - max_features
            for i in range(to_remove):
                individual[importance_scores[i][0]] = 0
        
        elif selected_count == 0:
            # Select at least the most important feature
            best_feature = self.feature_ranking[0]
            individual[best_feature] = 1
    
    def _evaluate_individual(self, individual):
        """Evaluate individual based on multiple criteria"""
        
        selected_features = np.array(individual, dtype=bool)
        n_selected = sum(individual)
        
        # Ensure selected features exist
        if n_selected == 0:
            return 0.0,
        
        try:
            X_selected = self.X[:, selected_features]
            
            # Choose model based on data size
            if len(self.y) > 1000:
                model = LogisticRegression(max_iter=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            
            # Cross-validation
            if self.scoring_method == 'balanced':
                # Balanced evaluation combining accuracy and feature reduction
                scores = cross_val_score(model, X_selected, self.y, cv=self.cv, scoring='accuracy')
                accuracy = scores.mean()
                
                # Feature reduction bonus
                feature_reduction_bonus = 1 - (n_selected / self.n_features)
                
                # Final score
                fitness = 0.7 * accuracy + 0.3 * feature_reduction_bonus
                
            elif self.scoring_method == 'accuracy':
                scores = cross_val_score(model, X_selected, self.y, cv=self.cv, scoring='accuracy')
                fitness = scores.mean()
                
            elif self.scoring_method == 'f1':
                scores = cross_val_score(model, X_selected, self.y, cv=self.cv, scoring='f1_weighted')
                fitness = scores.mean()
                
            else:  # roc_auc
                scores = cross_val_score(model, X_selected, self.y, cv=self.cv, scoring='roc_auc_ovr')
                fitness = scores.mean()
            
            # Track best score
            if fitness > self.best_global_fitness:
                self.best_global_fitness = fitness
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
            
            return fitness,
            
        except Exception as e:
            return 0.0,
    
    def run(self):
        """Run the enhanced genetic algorithm"""
        
        print(f"Starting genetic algorithm for feature selection...")
        print(f"Original features: {self.n_features}")
        print(f"Data size: {len(self.y)} samples")
        print(f"Max features: {int(self.max_features_ratio * self.n_features)}")
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Evolution log
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        print(f"Generation 0: Best = {record['max']:.4f}, Average = {record['avg']:.4f}")
        
        # Save statistics
        self.best_fitness_history.append(record['max'])
        self.avg_fitness_history.append(record['avg'])
        
        # Evolution across generations
        for generation in range(1, self.generations + 1):
            
            # Early stopping
            if self.generations_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at generation {generation} (no improvements for {self.early_stopping_patience} generations)")
                break
            
            # Select elite
            elite = tools.selBest(population, self.elite_size)
            
            # Select individuals for reproduction
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if np.random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Merge elite with new generations
            population[:] = elite + offspring
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=generation, nevals=len(invalid_ind), **record)
            
            if generation % 5 == 0 or generation == self.generations:
                print(f"Generation {generation}: Best = {record['max']:.4f}, Average = {record['avg']:.4f}")
            
            # Save statistics
            self.best_fitness_history.append(record['max'])
            self.avg_fitness_history.append(record['avg'])
        
        # Get best solution
        best_individual = tools.selBest(population, k=1)[0]
        selected_features = np.array(best_individual, dtype=bool)
        
        # Analyze results
        results = self._analyze_results(selected_features, best_individual.fitness.values[0])
        
        return selected_features, best_individual.fitness.values[0], logbook, results
    
    def _analyze_results(self, selected_features, fitness_score):
        """Detailed results analysis"""
        
        n_selected = sum(selected_features)
        selected_indices = np.where(selected_features)[0]
        
        # Calculate importance of selected features
        selected_importance = self.feature_importance[selected_indices]
        
        results = {
            'n_selected_features': int(n_selected),
            'selection_ratio': float(n_selected / self.n_features),
            'fitness_score': float(fitness_score),
            'selected_indices': selected_indices.tolist(),
            'selected_importance_scores': selected_importance.tolist(),
            'avg_importance': float(selected_importance.mean()) if len(selected_importance) > 0 else 0.0,
            'improvement_over_random': float(fitness_score - 0.5),  # Comparison with random selection
            'generations_run': int(len(self.best_fitness_history)),
            'convergence_info': {
                'best_fitness_history': [float(x) for x in self.best_fitness_history],
                'avg_fitness_history': [float(x) for x in self.avg_fitness_history]
            }
        }
        
        return results
    
    def get_feature_ranking(self):
        """Get feature ranking by importance"""
        return {
            'feature_indices': self.feature_ranking.tolist(),
            'importance_scores': self.feature_importance[self.feature_ranking].tolist()
        }
    
    def plot_convergence(self):
        """Plot convergence curve"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.best_fitness_history, label='Best Fitness', linewidth=2)
            plt.plot(self.avg_fitness_history, label='Average Fitness', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Fitness Score')
            plt.title('Genetic Algorithm Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
