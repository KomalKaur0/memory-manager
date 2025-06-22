#!/usr/bin/env python3
"""
Enhanced Benchmark Suite for AI Memory System

This script provides comprehensive performance testing with configurable parameters.
"""

import time
import json
import random
import statistics
import psutil
import argparse
import sys
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Import core components
from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode, ConnectionType

# Import benchmark configuration
sys.path.append(str(Path(__file__).parent))
from benchmark_config import get_config, list_configs, validate_config


@dataclass
class BenchmarkResult:
    """Results from a benchmark test"""
    test_name: str
    duration: float
    operations_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]


class EnhancedMemoryBenchmark:
    """Enhanced benchmark suite for memory system with configurable parameters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # Test data
        self.test_memories: List[MemoryNode] = []
        self.memory_graph = MemoryGraph(decay_rate=0.01)
        
        # Topics and concepts for generating diverse test data
        self.topics = [
            "Python programming", "Machine learning", "Data science", "Web development",
            "Database systems", "Cloud computing", "DevOps", "Cybersecurity",
            "Artificial intelligence", "Natural language processing", "Computer vision"
        ]
        
        self.concepts = [
            "async programming", "dependency injection", "microservices", "containerization",
            "neural networks", "deep learning", "data preprocessing", "feature engineering",
            "model evaluation", "hyperparameter tuning", "deployment strategies"
        ]
        
    def setup(self):
        """Generate test data based on configuration"""
        num_memories = self.config["num_memories"]
        content_length = self.config["memory_content_length"]
        connection_prob = self.config["connection_probability"]
        
        if self.config.get("verbose_logging", True):
            print(f"Generating {num_memories:,} test memories...")
        
        # Generate test memories
        for i in range(num_memories):
            topic = random.choice(self.topics)
            concept = random.choice(self.concepts)
            
            # Generate content with specified length
            content = f"This is a detailed explanation of {concept} and how it applies to {topic}. " * 10
            if len(content) > content_length:
                content = content[:content_length] + "..."
            
            memory = MemoryNode(
                concept=f"{concept} in {topic}",
                summary=f"Comprehensive guide to {concept} for {topic} applications",
                full_content=content,
                keywords=[concept, topic, "tutorial", "guide"],
                tags=[topic.lower().replace(" ", "-"), concept.lower().replace(" ", "-")],
                importance_score=random.uniform(0.3, 0.9)
            )
            
            self.test_memories.append(memory)
            self.memory_graph.add_node(memory)
            
            # Add connections based on probability
            if i > 0 and random.random() < connection_prob:
                target_idx = random.randint(0, i-1)
                connection_type = random.choice(list(ConnectionType))
                self.memory_graph.create_connection(
                    memory.id, 
                    self.test_memories[target_idx].id,
                    connection_type,
                    initial_weight=random.uniform(0.1, 0.8)
                )
        
        if self.config.get("verbose_logging", True):
            print(f"Generated {len(self.test_memories):,} memories with connections")
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "cpu_usage_percent": process.cpu_percent()
        }
    
    def benchmark_memory_creation(self) -> BenchmarkResult:
        """Benchmark memory node creation"""
        num_operations = self.config["num_operations"]
        
        if self.config.get("verbose_logging", True):
            print(f"Benchmarking memory creation ({num_operations:,} operations)...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        for i in range(num_operations):
            try:
                memory = MemoryNode(
                    concept=f"Benchmark memory {i}",
                    summary=f"Test summary {i}",
                    full_content=f"Test content {i} for benchmarking memory creation performance. " * 10,
                    keywords=[f"benchmark{i}", "test"],
                    tags=["benchmark", "test"],
                    importance_score=random.uniform(0.1, 1.0)
                )
                operations += 1
                
            except Exception as e:
                errors += 1
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics["memory_usage_mb"] + end_metrics["memory_usage_mb"]) / 2
        avg_cpu = (start_metrics["cpu_usage_percent"] + end_metrics["cpu_usage_percent"]) / 2
        
        return BenchmarkResult(
            test_name="Memory Creation",
            duration=duration,
            operations_per_second=operations / duration if duration > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success_rate=(operations - errors) / operations if operations > 0 else 0,
            error_count=errors,
            metadata={"total_operations": operations}
        )
    
    def benchmark_graph_operations(self) -> BenchmarkResult:
        """Benchmark memory graph operations"""
        num_operations = self.config["num_operations"]
        
        if self.config.get("verbose_logging", True):
            print(f"Benchmarking graph operations ({num_operations:,} operations)...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        for i in range(num_operations):
            try:
                if i < len(self.test_memories):
                    memory = self.test_memories[i]
                    
                    # Test node retrieval
                    retrieved = self.memory_graph.get_node(memory.id)
                    operations += 1
                    
                    # Test connection creation
                    if i > 0:
                        target = self.test_memories[i-1]
                        success = self.memory_graph.create_connection(
                            memory.id, target.id, ConnectionType.SIMILARITY, 0.5
                        )
                        operations += 1
                    
                    # Test connection strengthening
                    if memory.connections:
                        target_id = list(memory.connections.keys())[0]
                        success = self.memory_graph.strengthen_connection(memory.id, target_id, 0.1)
                        operations += 1
                
            except Exception as e:
                errors += 1
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics["memory_usage_mb"] + end_metrics["memory_usage_mb"]) / 2
        avg_cpu = (start_metrics["cpu_usage_percent"] + end_metrics["cpu_usage_percent"]) / 2
        
        return BenchmarkResult(
            test_name="Graph Operations",
            duration=duration,
            operations_per_second=operations / duration if duration > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success_rate=(operations - errors) / operations if operations > 0 else 0,
            error_count=errors,
            metadata={"total_operations": operations}
        )
    
    def benchmark_search_operations(self) -> BenchmarkResult:
        """Benchmark search operations"""
        num_operations = self.config["num_operations"]
        search_terms = self.config["search_terms"]
        
        if self.config.get("verbose_logging", True):
            print(f"Benchmarking search operations ({num_operations:,} operations)...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        for i in range(num_operations):
            try:
                term = random.choice(search_terms)
                
                # Test different search methods
                results = self.memory_graph.find_nodes_by_concept(term)
                operations += 1
                
                results = self.memory_graph.find_nodes_by_keyword(term)
                operations += 1
                
                results = self.memory_graph.find_nodes_by_tag(term)
                operations += 1
                
            except Exception as e:
                errors += 1
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics["memory_usage_mb"] + end_metrics["memory_usage_mb"]) / 2
        avg_cpu = (start_metrics["cpu_usage_percent"] + end_metrics["cpu_usage_percent"]) / 2
        
        return BenchmarkResult(
            test_name="Search Operations",
            duration=duration,
            operations_per_second=operations / duration if duration > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success_rate=(operations - errors) / operations if operations > 0 else 0,
            error_count=errors,
            metadata={"total_operations": operations}
        )
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests"""
        print("🚀 Enhanced AI Memory System Benchmark Suite")
        print("=" * 60)
        print(f"Configuration: {self.config.get('name', 'custom')}")
        print(f"Memories: {self.config['num_memories']:,}")
        print(f"Operations: {self.config['num_operations']:,}")
        print("=" * 60)
        
        # Setup
        self.setup()
        
        # Run benchmarks
        benchmarks = [
            self.benchmark_memory_creation(),
            self.benchmark_graph_operations(),
            self.benchmark_search_operations()
        ]
        
        self.results.extend(benchmarks)
        
        # Generate summary
        summary = self._generate_summary()
        
        if self.config.get("verbose_logging", True):
            print("✅ Benchmark suite completed")
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Calculate overall statistics
        total_duration = sum(r.duration for r in self.results)
        avg_ops_per_sec = statistics.mean([r.operations_per_second for r in self.results if r.operations_per_second > 0])
        avg_memory = statistics.mean([r.memory_usage_mb for r in self.results])
        avg_cpu = statistics.mean([r.cpu_usage_percent for r in self.results])
        avg_success_rate = statistics.mean([r.success_rate for r in self.results])
        total_errors = sum(r.error_count for r in self.results)
        
        # Find best and worst performers
        best_performer = max(self.results, key=lambda r: r.operations_per_second)
        worst_performer = min(self.results, key=lambda r: r.operations_per_second)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "overall_metrics": {
                "total_duration_seconds": total_duration,
                "average_operations_per_second": avg_ops_per_sec,
                "average_memory_usage_mb": avg_memory,
                "average_cpu_usage_percent": avg_cpu,
                "average_success_rate": avg_success_rate,
                "total_errors": total_errors
            },
            "best_performer": {
                "test_name": best_performer.test_name,
                "operations_per_second": best_performer.operations_per_second,
                "success_rate": best_performer.success_rate
            },
            "worst_performer": {
                "test_name": worst_performer.test_name,
                "operations_per_second": worst_performer.operations_per_second,
                "success_rate": worst_performer.success_rate
            },
            "detailed_results": [asdict(r) for r in self.results],
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        min_ops_per_sec = self.config.get("min_ops_per_sec", 100)
        min_success_rate = self.config.get("min_success_rate", 0.95)
        max_memory_usage = self.config.get("max_memory_usage_mb", 500)
        
        for result in self.results:
            if result.operations_per_second < min_ops_per_sec:
                recommendations.append(f"Consider optimizing {result.test_name} - throughput below {min_ops_per_sec} ops/sec")
            
            if result.success_rate < min_success_rate:
                recommendations.append(f"Investigate errors in {result.test_name} - success rate below {min_success_rate:.1%}")
            
            if result.memory_usage_mb > max_memory_usage:
                recommendations.append(f"Monitor memory usage in {result.test_name} - exceeds {max_memory_usage} MB")
        
        if not recommendations:
            recommendations.append("All benchmarks passed performance thresholds - system performing well")
        
        return recommendations
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = self.config.get("name", "custom")
            filename = f"enhanced_benchmark_{config_name}_{timestamp}.json"
        
        summary = self._generate_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.config.get("verbose_logging", True):
            print(f"💾 Results saved to: {filename}")
        
        return filename


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description="Enhanced AI Memory System Benchmark Suite")
    parser.add_argument("--config", "-c", default="default", 
                       help="Benchmark configuration to use")
    parser.add_argument("--list-configs", "-l", action="store_true",
                       help="List available benchmark configurations")
    parser.add_argument("--output", "-o", 
                       help="Output filename for results")
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_configs()
        return
    
    # Load configuration
    config = get_config(args.config)
    config["name"] = args.config
    
    # Validate configuration
    if not validate_config(config):
        print("Invalid configuration. Exiting.")
        return
    
    # Run benchmarks
    benchmark = EnhancedMemoryBenchmark(config)
    results = benchmark.run_all_benchmarks()
    
    # Print summary
    print("\n📊 Benchmark Results Summary")
    print("=" * 60)
    print(f"Total Duration: {results['overall_metrics']['total_duration_seconds']:.2f} seconds")
    print(f"Average Ops/sec: {results['overall_metrics']['average_operations_per_second']:.2f}")
    print(f"Average Memory: {results['overall_metrics']['average_memory_usage_mb']:.2f} MB")
    print(f"Average CPU: {results['overall_metrics']['average_cpu_usage_percent']:.2f}%")
    print(f"Success Rate: {results['overall_metrics']['average_success_rate']:.2%}")
    print(f"Total Errors: {results['overall_metrics']['total_errors']}")
    
    print(f"\n🏆 Best Performer: {results['best_performer']['test_name']}")
    print(f"   Ops/sec: {results['best_performer']['operations_per_second']:.2f}")
    
    print(f"\n⚠️  Worst Performer: {results['worst_performer']['test_name']}")
    print(f"   Ops/sec: {results['worst_performer']['operations_per_second']:.2f}")
    
    print("\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"   • {rec}")
    
    # Save results
    filename = benchmark.save_results(args.output)
    print(f"\n🎉 Benchmark completed! Results saved to: {filename}")


if __name__ == "__main__":
    main()
