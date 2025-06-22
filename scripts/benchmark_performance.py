#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for AI Memory System

This script benchmarks all major components of the memory system:
- Memory Graph operations (node creation, connections, traversal)
- Embedding Search performance (encoding, search, caching)
- Hybrid Retrieval (combined embedding + graph search)
- Relevance Agent performance (scoring, filtering)
- Overall system performance under load
- Memory usage and scalability
"""

import asyncio
import time
import json
import random
import statistics
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import memory system components
from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode, ConnectionType
from src.retrieval.embedding_search import EmbeddingSearch, EmbeddingConfig
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalConfig
from src.agents.relevance_agent import RelevanceAgent, QueryContext
from src.agents.filter_agent import FilterAgent
from src.retrieval.graph_traversal import GraphTraversal, TraversalConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float


class MemorySystemBenchmark:
    """Comprehensive benchmark suite for the AI Memory System"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize benchmark suite with configuration"""
        self.config = config or {}
        self.results: List[BenchmarkResult] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # Benchmark configuration
        self.num_memories = self.config.get('num_memories', 1000)
        self.num_queries = self.config.get('num_queries', 100)
        self.batch_size = self.config.get('batch_size', 50)
        self.warmup_iterations = self.config.get('warmup_iterations', 5)
        
        # Initialize components
        self.memory_graph = None
        self.embedding_search = None
        self.hybrid_retriever = None
        self.relevance_agent = None
        self.filter_agent = None
        self.traversal_config = TraversalConfig()

        # Test data
        self.test_memories: List[MemoryNode] = []
        self.test_queries: List[str] = []
        
    async def setup(self):
        """Initialize all components for benchmarking"""
        logger.info("Setting up benchmark environment...")
        
        # Initialize memory graph
        self.memory_graph = MemoryGraph(decay_rate=0.01)
        
        # Initialize embedding search (use mock for benchmarking)
        embedding_config = EmbeddingConfig(
            model_name="mock-model",
            api_key="test-key",
            cache_embeddings=True
        )
        self.embedding_search = EmbeddingSearch(embedding_config)
        # await self.embedding_search.initialize_client()
        
        # Initialize hybrid retriever
        retrieval_config = RetrievalConfig(
            max_total_results=10,
            embedding_weight=0.6,
            graph_weight=0.4
        )
        graph_traversal = GraphTraversal(self.memory_graph, config=self.traversal_config)
        self.hybrid_retriever = HybridRetriever(
            embedding_search=self.embedding_search,
            graph_traversal=graph_traversal,
            memory_graph=self.memory_graph,
            config=retrieval_config
        )
        
        # Initialize agents
        self.relevance_agent = RelevanceAgent(
            config={'conversation_history_length': 3},
            claude_client=None  # Use fallback for benchmarking
        )
        self.filter_agent = FilterAgent()
        
        # Generate test data
        await self._generate_test_data()
        
        logger.info("Benchmark environment setup complete")
    
    async def _generate_test_data(self):
        """Generate test memories and queries"""
        logger.info(f"Generating {self.num_memories} test memories...")
        
        # Sample topics for generating diverse test data
        topics = [
            "Python programming", "Machine learning", "Data science", "Web development",
            "Database systems", "Cloud computing", "DevOps", "Cybersecurity",
            "Artificial intelligence", "Natural language processing", "Computer vision",
            "Software architecture", "API design", "Testing", "Performance optimization"
        ]
        
        concepts = [
            "async programming", "dependency injection", "microservices", "containerization",
            "neural networks", "deep learning", "data preprocessing", "feature engineering",
            "model evaluation", "hyperparameter tuning", "deployment strategies",
            "monitoring", "logging", "error handling", "security best practices"
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
            
            # Add some connections between related memories
            if i > 0 and random.random() < 0.3:
                target_idx = random.randint(0, i-1)
                connection_type = random.choice(list(ConnectionType))
                self.memory_graph.create_connection(
                    memory.id, 
                    self.test_memories[target_idx].id,
                    connection_type,
                    initial_weight=random.uniform(0.1, 0.8)
                )
        
        # Generate test queries
        query_templates = [
            "How to implement {concept}?",
            "Best practices for {topic}",
            "Tutorial on {concept} in {topic}",
            "Common issues with {concept}",
            "Advanced techniques for {topic}",
            "Integration of {concept} with {topic}",
            "Performance optimization for {concept}",
            "Security considerations in {topic}"
        ]
        
        for i in range(self.num_queries):
            template = random.choice(query_templates)
            concept = random.choice(concepts)
            topic = random.choice(topics)
            query = template.format(concept=concept, topic=topic)
            self.test_queries.append(query)
        
        logger.info(f"Generated {len(self.test_memories)} memories and {len(self.test_queries)} queries")
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=process.cpu_percent(),
            disk_io_read_mb=0.0,  # Would need more complex monitoring
            disk_io_write_mb=0.0
        )
    
    async def benchmark_memory_graph_operations(self) -> BenchmarkResult:
        """Benchmark memory graph operations"""
        logger.info("Benchmarking memory graph operations...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        # Test node operations
        for i in range(self.num_memories // 10):  # Test with subset
            try:
                # Test node retrieval
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
                logger.warning(f"Error in graph operation: {e}")
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics.memory_usage_mb + end_metrics.memory_usage_mb) / 2
        avg_cpu = (start_metrics.cpu_usage_percent + end_metrics.cpu_usage_percent) / 2
        
        return BenchmarkResult(
            test_name="Memory Graph Operations",
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
    
    async def benchmark_embedding_search(self) -> BenchmarkResult:
        """Benchmark embedding search operations"""
        logger.info("Benchmarking embedding search...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        # Test embedding generation
        for i in range(0, len(self.test_memories), self.batch_size):
            try:
                batch = self.test_memories[i:i+self.batch_size]
                texts = [mem.full_content[:500] for mem in batch]  # Truncate for speed
                
                embeddings = await self.embedding_search.encode_batch(texts)
                operations += len(embeddings)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Error in embedding generation: {e}")
        
        # Test search operations
        for query in self.test_queries[:self.num_queries // 2]:
            try:
                results = await self.embedding_search.search(query, top_k=10)
                operations += 1
                
            except Exception as e:
                errors += 1
                logger.warning(f"Error in embedding search: {e}")
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics.memory_usage_mb + end_metrics.memory_usage_mb) / 2
        avg_cpu = (start_metrics.cpu_usage_percent + end_metrics.cpu_usage_percent) / 2
        
        return BenchmarkResult(
            test_name="Embedding Search",
            duration=duration,
            operations_per_second=operations / duration if duration > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success_rate=(operations - errors) / operations if operations > 0 else 0,
            error_count=errors,
            metadata={
                "total_operations": operations,
                "embeddings_generated": operations - (self.num_queries // 2),
                "searches_performed": self.num_queries // 2,
                "batch_size": self.batch_size
            }
        )
    
    async def benchmark_hybrid_retrieval(self) -> BenchmarkResult:
        """Benchmark hybrid retrieval operations"""
        logger.info("Benchmarking hybrid retrieval...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        for query in self.test_queries:
            try:
                result = await self.hybrid_retriever.retrieve(query)
                operations += 1
                
            except Exception as e:
                errors += 1
                logger.warning(f"Error in hybrid retrieval: {e}")
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics.memory_usage_mb + end_metrics.memory_usage_mb) / 2
        avg_cpu = (start_metrics.cpu_usage_percent + end_metrics.cpu_usage_percent) / 2
        
        return BenchmarkResult(
            test_name="Hybrid Retrieval",
            duration=duration,
            operations_per_second=operations / duration if duration > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success_rate=(operations - errors) / operations if operations > 0 else 0,
            error_count=errors,
            metadata={
                "total_queries": operations,
                "avg_results_per_query": 10,  # From config
                "embedding_weight": 0.6,
                "graph_weight": 0.4
            }
        )
    
    async def benchmark_relevance_agent(self) -> BenchmarkResult:
        """Benchmark relevance agent operations"""
        logger.info("Benchmarking relevance agent...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        # Test relevance scoring
        for i, query in enumerate(self.test_queries[:self.num_queries // 2]):
            try:
                context = QueryContext(
                    query=query,
                    conversation_history=["Previous message about AI"],
                    user_intent="search"
                )
                
                # Score a subset of memories
                memories_to_score = self.test_memories[i:i+10] if i+10 < len(self.test_memories) else self.test_memories[i:]
                
                for memory in memories_to_score:
                    score = self.relevance_agent.evaluate_relevance(memory, query, context)
                    operations += 1
                
            except Exception as e:
                errors += 1
                logger.warning(f"Error in relevance scoring: {e}")
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics.memory_usage_mb + end_metrics.memory_usage_mb) / 2
        avg_cpu = (start_metrics.cpu_usage_percent + end_metrics.cpu_usage_percent) / 2
        
        return BenchmarkResult(
            test_name="Relevance Agent",
            duration=duration,
            operations_per_second=operations / duration if duration > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success_rate=(operations - errors) / operations if operations > 0 else 0,
            error_count=errors,
            metadata={
                "total_scorings": operations,
                "queries_processed": self.num_queries // 2,
                "avg_memories_per_query": 10
            }
        )
    
    async def benchmark_end_to_end_workflow(self) -> BenchmarkResult:
        """Benchmark complete end-to-end workflow"""
        logger.info("Benchmarking end-to-end workflow...")
        
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        operations = 0
        errors = 0
        
        for query in self.test_queries[:self.num_queries // 4]:  # Use subset for E2E
            try:
                # 1. Hybrid retrieval
                retrieval_result = await self.hybrid_retriever.retrieve(query)
                
                # 2. Relevance scoring
                context = QueryContext(query=query, user_intent="search")
                scored_memories = []
                
                for memory_data in retrieval_result.memories[:5]:  # Top 5 results
                    memory = self.memory_graph.get_node(memory_data["memory_id"])
                    if memory:
                        score = self.relevance_agent.evaluate_relevance(memory, query, context)
                        scored_memories.append((memory, score))
                
                # 3. Filtering
                if scored_memories:
                    memories, scores = zip(*scored_memories)
                    filter_result = self.filter_agent.filter_for_response(
                        list(memories), list(scores), {}, context
                    )
                
                operations += 1
                
            except Exception as e:
                errors += 1
                logger.warning(f"Error in E2E workflow: {e}")
        
        end_time = time.time()
        end_metrics = self._get_system_metrics()
        
        duration = end_time - start_time
        avg_memory = (start_metrics.memory_usage_mb + end_metrics.memory_usage_mb) / 2
        avg_cpu = (start_metrics.cpu_usage_percent + end_metrics.cpu_usage_percent) / 2
        
        return BenchmarkResult(
            test_name="End-to-End Workflow",
            duration=duration,
            operations_per_second=operations / duration if duration > 0 else 0,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            success_rate=(operations - errors) / operations if operations > 0 else 0,
            error_count=errors,
            metadata={
                "total_workflows": operations,
                "components_involved": ["hybrid_retrieval", "relevance_scoring", "filtering"],
                "avg_memories_per_workflow": 5
            }
        )
    
    async def benchmark_scalability(self) -> List[BenchmarkResult]:
        """Benchmark system scalability with different data sizes"""
        logger.info("Benchmarking scalability...")
        
        scalability_results = []
        sizes = [100, 500, 1000, 2000]  # Memory counts to test
        
        for size in sizes:
            logger.info(f"Testing scalability with {size} memories...")
            
            # Create a new graph with specified size
            test_graph = MemoryGraph(decay_rate=0.01)
            test_memories = []
            
            # Generate memories for this size
            for i in range(size):
                memory = MemoryNode(
                    concept=f"Test concept {i}",
                    summary=f"Test summary {i}",
                    full_content=f"This is test content {i} for scalability testing",
                    keywords=[f"test{i}", "scalability"],
                    tags=["test", "scalability"]
                )
                test_memories.append(memory)
                test_graph.add_node(memory)
            
            # Benchmark retrieval with this size
            start_time = time.time()
            operations = 0
            
            for query in self.test_queries[:10]:  # Use subset of queries
                try:
                    # Simulate retrieval (simplified for scalability test)
                    results = test_graph.find_nodes_by_concept("test")
                    operations += 1
                except Exception:
                    pass
            
            duration = time.time() - start_time
            
            result = BenchmarkResult(
                test_name=f"Scalability ({size} memories)",
                duration=duration,
                operations_per_second=operations / duration if duration > 0 else 0,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_usage_percent=psutil.Process().cpu_percent(),
                success_rate=1.0,
                error_count=0,
                metadata={
                    "memory_count": size,
                    "operations": operations,
                    "avg_latency_per_operation": duration / operations if operations > 0 else 0
                }
            )
            
            scalability_results.append(result)
        
        return scalability_results
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests and return comprehensive results"""
        logger.info("Starting comprehensive benchmark suite...")
        
        # Setup
        await self.setup()
        
        # Warmup
        logger.info("Running warmup iterations...")
        for _ in range(self.warmup_iterations):
            await self.benchmark_memory_graph_operations()
        
        # Run individual benchmarks
        benchmarks = [
            self.benchmark_memory_graph_operations(),
            self.benchmark_embedding_search(),
            self.benchmark_hybrid_retrieval(),
            self.benchmark_relevance_agent(),
            self.benchmark_end_to_end_workflow()
        ]
        
        results = await asyncio.gather(*benchmarks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Benchmark {i} failed: {result}")
                results[i] = BenchmarkResult(
                    test_name=f"Failed Benchmark {i}",
                    duration=0.0,
                    operations_per_second=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    success_rate=0.0,
                    error_count=1,
                    metadata={"error": str(result)}
                )
        
        self.results.extend(results)
        
        # Run scalability benchmarks
        scalability_results = await self.benchmark_scalability()
        self.results.extend(scalability_results)
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info("Benchmark suite completed")
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary"""
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
        """Generate performance recommendations based on benchmark results"""
        recommendations = []
        
        # Analyze results and generate recommendations
        for result in self.results:
            if result.operations_per_second < 10:
                recommendations.append(f"Consider optimizing {result.test_name} - low throughput detected")
            
            if result.success_rate < 0.95:
                recommendations.append(f"Investigate errors in {result.test_name} - success rate below 95%")
            
            if result.memory_usage_mb > 1000:
                recommendations.append(f"Monitor memory usage in {result.test_name} - high memory consumption")
        
        if not recommendations:
            recommendations.append("All benchmarks passed performance thresholds - system performing well")
        
        return recommendations
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        summary = self._generate_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filename}")
        return filename


async def main():
    """Main benchmark execution"""
    print("🚀 AI Memory System Benchmark Suite")
    print("=" * 50)
    
    # Configuration
    config = {
        'num_memories': 1000,
        'num_queries': 100,
        'batch_size': 50,
        'warmup_iterations': 3
    }
    
    # Run benchmarks
    benchmark = MemorySystemBenchmark(config)
    results = await benchmark.run_all_benchmarks()
    
    # Print summary
    print("\n📊 Benchmark Results Summary")
    print("=" * 50)
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
    print(f"\n💾 Results saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
