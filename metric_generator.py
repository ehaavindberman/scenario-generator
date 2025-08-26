import numpy as np
import pandas as pd
from collections import defaultdict, deque
import re

class MetricGenerator:
    def __init__(self, num_days, seed=42):
        self.num_days = num_days
        self.seed = seed
        np.random.seed(seed)
        
    def generate_base_metric(self, name, breakpoints, params, randomness, integer=False):
        """Generate a base metric with piecewise linear segments"""
        base_values = np.zeros(self.num_days)
        noise = np.zeros(self.num_days)
        
        for j, (start, end) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
            m, b = params[j]
            x = np.arange(end - start)
            base_values[start:end] = m * x + b
            noise[start:end] = np.random.randn(end - start) * randomness * b
            
        if integer:
            actual_values = np.round(base_values + noise)
        else:
            actual_values = base_values + noise
            
        return actual_values.astype(float)
    
    def generate_breakdown_segments(self, base_values, segment_names, weights, randomness):
        """Generate breakdown segments for a base metric"""
        weights = np.array(weights)
        weights /= weights.sum()
        proportions = np.random.dirichlet(
            weights * (1.0 - randomness) + 1.0, 
            size=self.num_days
        )
        
        segments = {}
        for j, seg_name in enumerate(segment_names):
            seg_values = base_values * proportions[:, j]
            segments[seg_name] = seg_values
            
        return segments

class CalculatedMetricProcessor:
    def __init__(self, base_metrics):
        self.base_metrics = base_metrics.copy()
        self.all_metrics = base_metrics.copy()
        
    def extract_dependencies(self, expr, known_names):
        """Extract metric dependencies from a formula expression"""
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)
        return [t for t in tokens if t in known_names]
    
    def build_dependency_graph(self, calc_metrics):
        """Build dependency graph for calculated metrics"""
        dep_graph = defaultdict(list)
        in_degree = defaultdict(int)
        calc_by_name = {}
        
        # Initialize all calculated metrics with in_degree 0
        for metric in calc_metrics:
            in_degree[metric["name"]] = 0
            calc_by_name[metric["name"]] = metric
        
        # Build dependency graph
        base_metric_names = set(self.base_metrics.keys())
        calc_metric_names = {m["name"] for m in calc_metrics}
        
        for metric in calc_metrics:
            name = metric["name"]
            
            for formula in metric["formulas"]:
                if formula.strip():
                    deps = self.extract_dependencies(formula, base_metric_names | calc_metric_names)
                    
                    for dep in deps:
                        if dep != name:  # Avoid self-reference
                            # Only create dependency edges for OTHER calculated metrics
                            # Base metrics are already computed, so they don't create dependencies
                            if dep in calc_metric_names:
                                dep_graph[dep].append(name)
                                in_degree[name] += 1
                            # If dep is in base_metric_names, we don't increment in_degree
                            # because base metrics are already available
        
        return dep_graph, in_degree, calc_by_name
    
    def topological_sort(self, calc_metrics):
        """Perform topological sort to determine calculation order"""
        dep_graph, in_degree, calc_by_name = self.build_dependency_graph(calc_metrics)
        
        # Debug information
        print("Debug: Dependency analysis")
        for metric in calc_metrics:
            name = metric["name"]
            formulas = [f for f in metric["formulas"] if f.strip()]
            print(f"  {name}: formulas = {formulas}")
            
            for formula in formulas:
                if formula.strip():
                    base_names = set(self.base_metrics.keys())
                    calc_names = {m["name"] for m in calc_metrics}
                    deps = self.extract_dependencies(formula, base_names | calc_names)
                    print(f"    '{formula}' -> deps: {deps}")
            
            print(f"    in_degree: {in_degree[name]}")
        
        queue = deque([name for name in calc_by_name.keys() if in_degree[name] == 0])
        calc_order = []
        
        print(f"Debug: Starting queue: {list(queue)}")
        
        while queue:
            current = queue.popleft()
            calc_order.append(current)
            for neighbor in dep_graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        print(f"Debug: Final order: {calc_order}")
        print(f"Debug: Expected count: {len(calc_metrics)}, Got count: {len(calc_order)}")
        
        # Check for circular dependencies
        if len(calc_order) != len(calc_metrics):
            remaining = [name for name in calc_by_name.keys() if name not in calc_order]
            remaining_deps = {name: in_degree[name] for name in remaining}
            raise ValueError(f"Circular dependency detected. Could not process: {remaining}. Remaining in_degrees: {remaining_deps}")
        
        return calc_order, calc_by_name
    
    def evaluate_calculated_metric(self, metric, num_days):
        """Evaluate a single calculated metric"""
        result = np.zeros(num_days)
        days = np.arange(num_days)
        
        for j, (start, end) in enumerate(zip(metric["breakpoints"][:-1], metric["breakpoints"][1:])):
            formula = metric["formulas"][j]
            if formula.strip():
                # Create a safe evaluation environment
                safe_dict = {"np": np, "days": days}
                safe_dict.update(self.all_metrics)
                
                local_result = eval(formula, {"__builtins__": {}}, safe_dict)
                if np.isscalar(local_result):
                    result[start:end] = local_result
                else:
                    result[start:end] = local_result[start:end]
            else:
                result[start:end] = 0
        
        # Add noise if specified
        if metric["randomness"] > 0:
            std = np.std(result) if np.std(result) > 0 else 1.0
            noise = np.random.randn(len(result)) * metric["randomness"] * std
            result = result + noise
        
        # Convert to integer if specified
        final_result = np.round(result).astype(int) if metric["integer"] else result
        
        return final_result
    
    def process_calculated_metrics(self, calc_metrics, num_days):
        """Process all calculated metrics in dependency order"""
        if not calc_metrics:
            return {}
        
        calc_order, calc_by_name = self.topological_sort(calc_metrics)
        calculated = {}
        
        for name in calc_order:
            metric = calc_by_name[name]
            try:
                result = self.evaluate_calculated_metric(metric, num_days)
                self.all_metrics[name] = result
                calculated[name] = result
            except Exception as e:
                raise ValueError(f"Error computing {name}: {e}")
        
        return calculated