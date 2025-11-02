import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from .chromosome import create_chromosome, BinaryChromosome

class GeneticFeatureSelector:
    def __init__(self, X, y, population_size=30, generations=5, cx_prob=0.7, mut_prob=0.3, 
                 chromosome_type='binary', max_features=None):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.chromosome_type = chromosome_type
        self.max_features = max_features or self.n_features // 2
        self.best_fitness = 0
        self.generations_no_improve = 0
        
        # Take a sample if dataset is too large
        if len(y) > 5000:
            from sklearn.model_selection import train_test_split
            self.X, _, self.y, _ = train_test_split(X, y, train_size=5000, stratify=y, random_state=42)
        
        # Scale the features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        # Get class frequencies
        from collections import Counter
        self.class_counts = Counter(self.y)
        
        # Create fitness and individual classes (with cleanup to avoid warnings)
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Initialize toolbox with chromosome support
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _create_individual(self):
        """Create an individual using the specified chromosome type"""
        chromosome = create_chromosome(
            self.chromosome_type, 
            self.n_features, 
            max_features=self.max_features
        )
        chromosome.initialize()
        
        # Convert chromosome to DEAP individual format
        individual = creator.Individual(chromosome.get_selected_features())
        # Store chromosome object for genetic operations
        individual.chromosome = chromosome
        # Convert to binary representation for compatibility
        binary_repr = [0] * self.n_features
        for idx in chromosome.get_selected_features():
            binary_repr[idx] = 1
        individual[:] = binary_repr
        
        return individual
    
    def _crossover(self, ind1, ind2):
        """Perform crossover using chromosome-specific methods"""
        if hasattr(ind1, 'chromosome') and hasattr(ind2, 'chromosome'):
            # Use chromosome-specific crossover
            child1_chr, child2_chr = ind1.chromosome.crossover(ind2.chromosome)
            
            # Update individuals
            self._update_individual_from_chromosome(ind1, child1_chr)
            self._update_individual_from_chromosome(ind2, child2_chr)
        else:
            # Fallback to standard crossover
            tools.cxTwoPoint(ind1, ind2)
        
        return ind1, ind2
    
    def _mutate(self, individual):
        """Perform mutation using chromosome-specific methods"""
        if hasattr(individual, 'chromosome'):
            # Use chromosome-specific mutation
            individual.chromosome.mutate(0.1)
            self._update_individual_from_chromosome(individual, individual.chromosome)
        else:
            # Fallback to standard mutation
            tools.mutFlipBit(individual, indpb=0.1)
        
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

    def _evaluate_individual(self, individual):
        if sum(individual) == 0:
            return 0.0,
        
        # Get selected features
        selected_features = np.array(individual, dtype=bool)
        X_selected = self.X[:, selected_features]
        
        try:
            # Fast evaluation using decision tree instead of random forest
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=5,
                class_weight='balanced'
            )
            
            # Use simple train-test split for speed
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, self.y, test_size=0.2, 
                stratify=self.y, random_state=42
            )
            
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            
            # Combine accuracy with feature reduction reward
            feature_reduction_ratio = 1 - (sum(individual) / len(individual))
            fitness = 0.7 * score + 0.3 * feature_reduction_ratio
            
            # Check for improvement
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.generations_no_improve = 0
            else:
                self.generations_no_improve += 1
            
            return fitness,
            
        except Exception as e:
            # Return low fitness score if evaluation fails
            return 0.0,
    
    def run(self):
        pop = self.toolbox.population(n=self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Custom evolution with early stopping
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        
        for gen in range(1, self.generations + 1):
            # Early stopping if no improvement for 2 generations
            if self.generations_no_improve >= 2:
                break
                
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
        
            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if np.random.random() < self.cx_prob:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values, offspring[i].fitness.values
        
            for i in range(len(offspring)):
                if np.random.random() < self.mut_prob:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
            # Replace the old population by the offspring
            pop[:] = offspring
        
            # Append the current generation statistics to the logbook
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
        
        # Get best solution
        best_individual = tools.selBest(pop, k=1)[0]
        selected_features = np.array(best_individual, dtype=bool)
        
        return selected_features, best_individual.fitness.values[0], logbook