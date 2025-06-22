#!/usr/bin/env python3
"""
Simple Benchmark Suite for AI Memory System

This script provides basic performance testing for core components:
- Memory Graph operations
- Memory Node operations
- Basic search and retrieval
- System resource usage
"""

import time
import json
import random
import statistics
import psutil
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import core components
from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode, ConnectionType


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


class SimpleMemoryBenchmark:
    """Simple benchmark suite for memory system core components"""
    
    def __init__(self, num_memories: int = 1000, num_operations: int = 100):
        self.num_memories = num_memories
        self.num_operations = num_operations
        self.results: List[BenchmarkResult] = []
        
        # Test data
        self.test_memories: List[MemoryNode] = []
        self.memory_graph = MemoryGraph(decay_rate=0.01)
        
    def setup(self):
        """Generate test data"""
        print(f"Generating {self.num_memories} test memories...")
        
        topics = [
            "Python programming", "Machine learning", "Data science", "Web development",
            "Database systems", "Cloud computing", "DevOps", "Cybersecurity"
        ]
        
        concepts = [
            "async programming", "dependency injection", "microservices", "containerization",
            "neural networks", "deep learning", "data preprocessing", "feature engineering"
        ]
        
        # Generate test memories
        for i in range(self.num_memories):
            topic = random.choice(topics)
            concept = random.choice(concepts)
            
            memory = MemoryNode(
                concept=f"{concept} in {topic}",
                summary=f"Comprehensive guide to {concept} for {topic} applications",
                full_content=f"This is a detailed explanation of {concept} and how it applies to {topic}. "
                            f"It covers best practices, common pitfalls, and real-world examples. "
                            f"Memory ID: {i}, Topic: {topic}, Concept: {concept}",
                keywords=[concept, topic, "tutorial", "guide"],
                tags=[topic.lower().replace(" ", "-"), concept.lower().replace(" ", "-")],
                importance_score=random.uniform(0.3, 0.9)
            )
            
            self.test_memories.append(memory)
            self.memory_graph.add_node(memory)
            
            # Add some connections
            if i > 0 and random.random() < 0.3:
                target_idx = random.randint(0, i-1)
                connection_type = random.choice(list(ConnectionType))
                self.memory_graph.create_connection(
                    memory.id, 
                    self.test_memories[target_idx].id,
                    connection_type,
                    initial_weight=random.uniform(0.1, 0.8)
                )
        
        print(f"Generated {len(self.test_memories)} memories with connections")
    
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
        print("Benchmarking memory creation...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        for i in range(self.num_operations):
            try:
                memory = MemoryNode(
                    concept=f"Benchmark memory {i}",
                    summary=f"Test summary {i}",
                    full_content=f"Test content {i} for benchmarking memory creation performance",
                    keywords=[f"benchmark{i}", "test"],
                    tags=["benchmark", "test"],
                    importance_score=random.uniform(0.1, 1.0)
                )
                operations += 1
                
            except Exception as e:
                errors += 1
                print(f"Error creating memory: {e}")
        
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
        print("Benchmarking graph operations...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        # Test node retrieval
        for i in range(self.num_operations):
            try:
                if i < len(self.test_memories):
                    memory = self.test_memories[i]
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
                print(f"Error in graph operation: {e}")
        
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
            metadata={
                "total_operations": operations,
                "node_retrievals": operations // 3,
                "connections_created": operations // 3,
                "connections_strengthened": operations // 3
            }
        )
    
    def benchmark_search_operations(self) -> BenchmarkResult:
        """Benchmark search operations"""
        print("Benchmarking search operations...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        search_terms = ["python", "machine", "data", "web", "database", "cloud", "devops", "security"]
        
        for i in range(self.num_operations):
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
                print(f"Error in search operation: {e}")
        
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
            metadata={
                "total_operations": operations,
                "concept_searches": operations // 3,
                "keyword_searches": operations // 3,
                "tag_searches": operations // 3
            }
        )
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests"""
        print("🚀 Starting Simple Memory System Benchmark Suite")
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
        
        for result in self.results:
            if result.operations_per_second < 100:
                recommendations.append(f"Consider optimizing {result.test_name} - low throughput detected")
            
            if result.success_rate < 0.95:
                recommendations.append(f"Investigate errors in {result.test_name} - success rate below 95%")
            
            if result.memory_usage_mb > 500:
                recommendations.append(f"Monitor memory usage in {result.test_name} - high memory consumption")
        
        if not recommendations:
            recommendations.append("All benchmarks passed performance thresholds - system performing well")
        
        return recommendations
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_benchmark_results_{timestamp}.json"
        
        summary = self._generate_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
        return filename


def main():
    """Main benchmark execution"""
    print("🚀 Simple AI Memory System Benchmark Suite")
    print("=" * 60)
    
    # Configuration
    num_memories = 1000
    num_operations = 500
    
    # Run benchmarks
    benchmark = SimpleMemoryBenchmark(num_memories, num_operations)
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
    filename = benchmark.save_results()
    print(f"\n🎉 Benchmark completed! Results saved to: {filename}")


if __name__ == "__main__":
    main()
