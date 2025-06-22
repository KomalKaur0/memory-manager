# AI Memory System Benchmark Suite

This directory contains comprehensive benchmarking tools for the AI Memory System, designed to test performance, scalability, and resource usage across different components and configurations.

## Overview

The benchmark suite consists of two main scripts:

1. **`simple_benchmark.py`** - Basic performance testing for core components
2. **`enhanced_benchmark.py`** - Advanced benchmarking with configurable parameters
3. **`benchmark_config.py`** - Configuration management for different test scenarios

## Quick Start

### Prerequisites

1. Install required dependencies:
```bash
pip install psutil
```

2. Set the Python path to include the project root:
```bash
export PYTHONPATH=/path/to/mem-manager
```

### Running Basic Benchmarks

```bash
# Run simple benchmark with default settings
python scripts/simple_benchmark.py

# Run enhanced benchmark with default configuration
python scripts/enhanced_benchmark.py

# List available configurations
python scripts/enhanced_benchmark.py --list-configs
```

### Running Different Configurations

```bash
# Small dataset (quick testing)
python scripts/enhanced_benchmark.py --config small

# Large dataset (stress testing)
python scripts/enhanced_benchmark.py --config large

# Search performance focus
python scripts/enhanced_benchmark.py --config search_performance

# Memory usage stress test
python scripts/enhanced_benchmark.py --config memory_stress

# Graph traversal testing
python scripts/enhanced_benchmark.py --config traversal
```

## Available Configurations

| Configuration | Memories | Operations | Content Length | Connection Prob | Use Case |
|---------------|----------|------------|----------------|-----------------|----------|
| `small` | 100 | 50 | 100 chars | 0.3 | Quick testing |
| `default` | 1,000 | 500 | 200 chars | 0.3 | Standard testing |
| `large` | 10,000 | 2,000 | 500 chars | 0.5 | Stress testing |
| `memory_stress` | 5,000 | 1,000 | 1,000 chars | 0.7 | Memory usage testing |
| `search_performance` | 2,000 | 1,000 | 200 chars | 0.3 | Search optimization |
| `traversal` | 1,500 | 300 | 200 chars | 0.6 | Graph traversal testing |

## Benchmark Components

### 1. Memory Creation
- Tests the performance of creating `MemoryNode` objects
- Measures creation speed and memory allocation
- Configurable content length and complexity

### 2. Graph Operations
- Node retrieval performance
- Connection creation and management
- Connection strengthening operations
- Tests graph manipulation efficiency

### 3. Search Operations
- Concept-based search
- Keyword-based search
- Tag-based search
- Tests different search strategies

### 4. Graph Traversal (Enhanced)
- Connected node discovery
- Similar concept finding
- Tests graph navigation performance

## Output and Results

### Console Output
The benchmark provides real-time feedback including:
- Progress indicators for each test
- Summary statistics
- Performance recommendations
- Best and worst performing components

### JSON Results
Detailed results are saved to JSON files with:
- Timestamp and configuration details
- Overall performance metrics
- Individual test results
- Scalability analysis
- System resource usage
- Performance recommendations

### Example Output
```
🚀 Enhanced AI Memory System Benchmark Suite
============================================================
Configuration: search_performance
Memories: 2,000
Operations: 1,000
============================================================

📊 Benchmark Results Summary
============================================================
Total Duration: 0.01 seconds
Average Ops/sec: 1,136,697.34
Average Memory: 50.03 MB
Average CPU: 0.00%
Success Rate: 100.00%
Total Errors: 0

🏆 Best Performer: Search Operations
   Ops/sec: 2,524,661.32

⚠️  Worst Performer: Memory Creation
   Ops/sec: 242,922.74

💡 Recommendations:
   • All benchmarks passed performance thresholds - system performing well
```

## Performance Thresholds

The benchmark suite includes configurable performance thresholds:

- **Minimum Operations/Second**: 100 ops/sec (default)
- **Minimum Success Rate**: 95% (default)
- **Maximum Memory Usage**: 500 MB (default)

These thresholds can be adjusted in the configuration files.

## Custom Configurations

You can create custom benchmark configurations by modifying `benchmark_config.py`:

```python
CUSTOM_CONFIG = {
    "num_memories": 5000,
    "num_operations": 1000,
    "memory_content_length": 300,
    "connection_probability": 0.4,
    "min_ops_per_sec": 200,
    "min_success_rate": 0.98,
    "max_memory_usage_mb": 1000,
    "search_terms": ["custom", "terms", "here"],
    "verbose_logging": True
}
```

## Interpreting Results

### Performance Metrics

1. **Operations/Second**: Higher is better, indicates throughput
2. **Memory Usage**: Lower is better, indicates efficiency
3. **CPU Usage**: Lower is better, indicates resource efficiency
4. **Success Rate**: Should be 100% for reliable systems
5. **Error Count**: Should be 0 for stable systems

### Recommendations

The benchmark provides automated recommendations based on:
- Performance below thresholds
- High memory usage
- Low success rates
- Scalability issues

### Scalability Analysis

For larger datasets, the benchmark analyzes:
- Performance degradation with data size
- Memory usage scaling
- Latency increases
- Resource utilization patterns

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Set PYTHONPATH correctly
2. **Memory Errors**: Reduce dataset size or increase system memory
3. **Slow Performance**: Check system resources and configuration

### Performance Optimization

Based on benchmark results, consider:
- Optimizing search algorithms
- Implementing caching strategies
- Reducing memory allocations
- Improving data structures

## Integration with CI/CD

The benchmark suite can be integrated into continuous integration:

```bash
# Run benchmarks in CI
python scripts/enhanced_benchmark.py --config small --output ci_results.json

# Check for performance regressions
python -c "
import json
with open('ci_results.json') as f:
    results = json.load(f)
    assert results['overall_metrics']['average_operations_per_second'] > 1000
    assert results['overall_metrics']['average_success_rate'] > 0.95
"
```

## Contributing

When adding new benchmark tests:

1. Follow the existing pattern in `BenchmarkResult` dataclass
2. Add appropriate metadata to results
3. Update configuration options if needed
4. Add performance thresholds for new metrics
5. Update this README with new features

## Future Enhancements

Planned improvements include:
- Async/await performance testing
- Database integration benchmarks
- Network latency simulation
- Load testing scenarios
- Memory leak detection
- Performance regression detection 