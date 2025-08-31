import numpy as np
import pandas as pd
from collections import defaultdict, deque
import re
from typing import Dict, List, Any

class MetricGenerator:
    def __init__(self, num_times, seed=42):
        self.num_times = num_times
        self.seed = seed
        np.random.seed(seed)
        
    def generate_base_metric(self, name, breakpoints, params, randomness, integer=False):
        """Generate a base metric with piecewise linear segments"""
        base_values = np.zeros(self.num_times)
        noise = np.zeros(self.num_times)
        
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
                            if dep in calc_metric_names:
                                dep_graph[dep].append(name)
                                in_degree[name] += 1
        
        return dep_graph, in_degree, calc_by_name
    
    def topological_sort(self, calc_metrics):
        """Perform topological sort to determine calculation order"""
        dep_graph, in_degree, calc_by_name = self.build_dependency_graph(calc_metrics)
        
        queue = deque([name for name in calc_by_name.keys() if in_degree[name] == 0])
        calc_order = []
        
        while queue:
            current = queue.popleft()
            calc_order.append(current)
            for neighbor in dep_graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(calc_order) != len(calc_metrics):
            remaining = [name for name in calc_by_name.keys() if name not in calc_order]
            remaining_deps = {name: in_degree[name] for name in remaining}
            raise ValueError(f"Circular dependency detected. Could not process: {remaining}. Remaining in_degrees: {remaining_deps}")
        
        return calc_order, calc_by_name
    
    def evaluate_calculated_metric(self, metric, num_times):
        """Evaluate a single calculated metric"""
        result = np.zeros(num_times)
        times = np.arange(num_times)
        
        for j, (start, end) in enumerate(zip(metric["breakpoints"][:-1], metric["breakpoints"][1:])):
            formula = metric["formulas"][j]
            if formula.strip():
                # Create a safe evaluation environment
                safe_dict = {"np": np, "times": times, "days": times}  # Keep 'days' for backward compatibility
                safe_dict.update(self.all_metrics)
                
                try:
                    local_result = eval(formula, {"__builtins__": {}}, safe_dict)
                    if np.isscalar(local_result):
                        result[start:end] = local_result
                    else:
                        result[start:end] = local_result[start:end]
                except Exception as e:
                    raise ValueError(f"Error evaluating formula '{formula}': {e}")
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
    
    def process_calculated_metrics(self, calc_metrics, num_times):
        """Process all calculated metrics in dependency order"""
        if not calc_metrics:
            return {}
        
        calc_order, calc_by_name = self.topological_sort(calc_metrics)
        calculated = {}
        
        for name in calc_order:
            metric = calc_by_name[name]
            try:
                result = self.evaluate_calculated_metric(metric, num_times)
                self.all_metrics[name] = result
                calculated[name] = result
            except Exception as e:
                raise ValueError(f"Error computing {name}: {e}")
        
        return calculated

class BreakdownProcessor:
    """Handles breakdown processing for both base and calculated metrics"""
    
    def __init__(self, breakdown_manager, seed=42):
        self.breakdown_manager = breakdown_manager
        self.seed = seed
    
    def apply_breakdowns_to_metrics(self, metrics_dict: Dict[str, np.ndarray], 
                                  breakdown_configs: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Apply breakdowns to all specified metrics"""
        result_metrics = metrics_dict.copy()
        
        for metric_name, values in metrics_dict.items():
            if metric_name in breakdown_configs:
                config = breakdown_configs[metric_name]
                if config.get("enabled", False) and config.get("selected"):
                    # Generate breakdown segments using the new structure
                    segments = self.breakdown_manager.generate_breakdown_segments(
                        values,
                        config["selected"],  # This is now a dict of breakdown configs
                        config.get("randomness", 0.1),
                        self.seed
                    )
                    
                    # Add segments to result with proper naming
                    for segment_name, segment_values in segments.items():
                        full_segment_name = f"{metric_name}_{segment_name}"
                        result_metrics[full_segment_name] = segment_values
        
        return result_metrics