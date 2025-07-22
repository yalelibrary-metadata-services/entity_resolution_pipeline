# Embedding Providers System

## Overview

The Entity Resolution Pipeline now supports multiple embedding providers through a unified, configurable abstraction layer. This system allows you to seamlessly switch between OpenAI, Mistral, HuggingFace, and other embedding models without changing your code - just update the configuration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Providers](#supported-providers)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Provider Comparison](#provider-comparison)
6. [Migration Guide](#migration-guide)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)
9. [Extending the System](#extending-the-system)
10. [API Reference](#api-reference)

---

## Quick Start

### Switch to Mistral Embeddings

1. **Set your API key**:
   ```bash
   export MISTRAL_API_KEY="your-api-key-here"
   ```

2. **Update config.yml**:
   ```yaml
   embedding:
     provider: "mistral"
   ```

3. **Run your pipeline** - it will automatically use Mistral's 1024-dimensional embeddings with optimized rate limiting.

### Use Local HuggingFace Models

1. **Update config.yml**:
   ```yaml
   embedding:
     provider: "huggingface"
     providers:
       huggingface:
         model: "sentence-transformers/all-MiniLM-L6-v2"
         device: "cuda"  # or "cpu"
   ```

2. **Run your pipeline** - no API keys needed, everything runs locally.

---

## Supported Providers

### OpenAI
- **Models**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Dimensions**: 1536 (small), 3072 (large), 1536 (ada-002)
- **Features**: Batch API support, comprehensive rate limiting, high accuracy
- **Cost**: Pay-per-token pricing
- **Best for**: Production deployments requiring high accuracy

### Mistral
- **Models**: `mistral-embed`
- **Dimensions**: 1024
- **Features**: High performance, generous rate limits (20M TPM, 200B TPM)
- **Cost**: $0.1/M tokens (5x higher than OpenAI)
- **Best for**: High-throughput deployments where rate limits matter more than cost

### HuggingFace (Local)
- **Models**: Any sentence-transformers model (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- **Dimensions**: Model-dependent (384, 768, 1024, etc.)
- **Features**: No API costs, GPU acceleration, offline capability
- **Cost**: Free (only compute costs)
- **Best for**: Development, cost-sensitive deployments, privacy-sensitive data

---

## Configuration

### New Provider Configuration Format

```yaml
# config.yml
embedding:
  # Active provider selection
  provider: "openai"  # Options: openai, mistral, huggingface
  
  # Provider-specific configurations
  providers:
    openai:
      model: "text-embedding-3-small"
      dimensions: 1536
      api_key_env: "OPENAI_API_KEY"
      api_base: "https://api.openai.com/v1"
      
      # Rate limiting
      batch_size: 512
      max_tokens_per_minute: 4800000
      max_requests_per_minute: 9500
      max_tokens_per_day: 480000000
      
      # Batch API settings
      supports_batch: true
      batch_endpoint: "/v1/embeddings/batch"
      max_batch_size: 50000
      
    mistral:
      model: "mistral-embed"
      dimensions: 1024
      api_key_env: "MISTRAL_API_KEY"
      api_base: "https://api.mistral.ai/v1"
      
      # Rate limiting - Mistral's generous limits
      batch_size: 500
      rate_limit_delay: 0.05
      max_tokens_per_minute: 19000000     # 95% of 20M TPM
      max_requests_per_minute: 2000
      max_tokens_per_month: 190000000000  # 95% of 200B TPM
      
      # Batch API settings
      supports_batch: false
      
    huggingface:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      dimensions: 384
      device: "cpu"  # or "cuda"
      
      # Local model settings
      batch_size: 256
      max_length: 512
      normalize_embeddings: true
      
      # No API rate limits for local models
      supports_batch: false
```

### Backward Compatibility

Existing configurations continue to work unchanged:

```yaml
# Legacy format (still supported)
embedding_model: "text-embedding-3-small"
embedding_dimensions: 1536
max_tokens_per_minute: 4800000
# ... other legacy settings
```

The system automatically detects and converts legacy configurations to the new provider format.

---

## Usage Examples

### Basic Provider Switching

Switch providers by changing a single line:

```yaml
# Use OpenAI
embedding:
  provider: "openai"

# Use Mistral  
embedding:
  provider: "mistral"

# Use HuggingFace
embedding:
  provider: "huggingface"
```

### Custom Model Configuration

#### High-Accuracy OpenAI Setup
```yaml
embedding:
  provider: "openai"
  providers:
    openai:
      model: "text-embedding-3-large"  # Higher accuracy
      dimensions: 3072
      batch_size: 256  # Smaller batches for large model
```

#### GPU-Accelerated HuggingFace
```yaml
embedding:
  provider: "huggingface"
  providers:
    huggingface:
      model: "sentence-transformers/all-mpnet-base-v2"  # Higher quality
      dimensions: 768
      device: "cuda"
      batch_size: 512  # Larger batches on GPU
```

#### Custom HuggingFace Model
```yaml
embedding:
  provider: "huggingface"
  providers:
    huggingface:
      model: "your-org/custom-embedding-model"
      dimensions: 1024
      device: "cuda"
      # Optional HuggingFace Hub token for private models
      # Set HF_TOKEN environment variable
```

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Mistral
export MISTRAL_API_KEY="..."

# HuggingFace (optional, for private models)
export HF_TOKEN="..."
export SENTENCE_TRANSFORMERS_HOME="/path/to/models"
```

---

## Provider Comparison

| Feature | OpenAI | Mistral | HuggingFace |
|---------|--------|---------|-------------|
| **Deployment** | API-based | API-based | Local |
| **Cost** | $0.02/M tokens | $0.1/M tokens | Free (compute only) |
| **Accuracy** | Excellent | Very Good | Good to Excellent* |
| **Speed** | Fast (API) | Very Fast | Variable (hardware-dependent) |
| **Rate Limits** | 5M TPM, 10K RPM | 20M TPM, 200B TPM | None (local) |
| **Privacy** | Data sent to API | Data sent to API | Fully private |
| **Offline** | No | No | Yes |
| **Batch API** | Yes | No | No |
| **Setup** | API key only | API key only | Model download |

*HuggingFace quality depends on model selection

### Cost Analysis

**For 1M embeddings (~100M tokens):**

- **OpenAI**: $2 (text-embedding-3-small at $0.02/M tokens)
- **Mistral**: $10 (mistral-embed at $0.1/M tokens)  
- **HuggingFace**: $0 (free, only compute costs)

**For 1B embeddings:**
- **OpenAI**: $2,000
- **Mistral**: $10,000
- **HuggingFace**: $0 + compute costs (potentially $100-500 for cloud GPU time)

---

## Migration Guide

### From Legacy OpenAI Configuration

Your existing setup continues to work without changes. To migrate to the new format:

1. **Run the migration utility**:
   ```bash
   python scripts/migrate_embedding_checkpoints.py --checkpoint-dir data/checkpoints --backup
   ```

2. **Optional: Update config.yml** to new format for better organization:
   ```yaml
   # OLD (still works)
   embedding_model: "text-embedding-3-small"
   embedding_dimensions: 1536
   
   # NEW (recommended)
   embedding:
     provider: "openai"
     providers:
       openai:
         model: "text-embedding-3-small"
         dimensions: 1536
   ```

### Switching Providers

When switching providers, consider:

1. **Dimension Changes**: Different models have different dimensions
   - OpenAI: 1536D or 3072D
   - Mistral: 1024D  
   - HuggingFace: 384D, 768D, 1024D (model-dependent)

2. **Weaviate Schema**: The system automatically creates compatible schemas

3. **Re-embedding**: Existing embeddings remain valid but new data uses the new provider

4. **Gradual Migration**: You can run side-by-side collections:
   ```bash
   # Create new collection with different provider
   python scripts/migrate_embeddings.py --to-provider mistral --collection-suffix _mistral
   ```

---

## Advanced Features

### Provider Factory Usage

```python
from src.embedding_providers.factory import EmbeddingProviderFactory

# Create provider from configuration
embedding_config = {
    'provider': 'mistral',
    'providers': {
        'mistral': {
            'model': 'mistral-embed',
            'dimensions': 1024,
            'api_key_env': 'MISTRAL_API_KEY'
        }
    }
}

provider = EmbeddingProviderFactory.create_from_full_config(embedding_config)

# Generate embeddings
texts = ["Hello world", "Test embedding"]
embeddings, token_counts = provider.get_embeddings(texts)
```

### Custom Rate Limiting

```yaml
embedding:
  provider: "mistral"
  providers:
    mistral:
      # Conservative rate limiting for shared API keys
      max_tokens_per_minute: 1000000    # 1M TPM instead of 19M
      max_requests_per_minute: 100      # 100 RPM instead of 2000
      rate_limit_delay: 0.5            # Longer delay between requests
```

### A/B Testing Providers

```python
def compare_providers(texts, providers=['openai', 'mistral']):
    """Compare embedding quality across providers."""
    results = {}
    
    for provider_name in providers:
        config = load_config()
        config['embedding']['provider'] = provider_name
        
        provider = EmbeddingProviderFactory.create_from_full_config(config['embedding'])
        embeddings, tokens = provider.get_embeddings(texts)
        
        results[provider_name] = {
            'embeddings': embeddings,
            'dimensions': provider.get_dimensions(),
            'tokens_used': sum(tokens),
            'cost_estimate': calculate_cost(provider_name, sum(tokens))
        }
    
    return results
```

### Dynamic Provider Selection

```python
def select_optimal_provider(text_count, budget_limit):
    """Dynamically select provider based on workload and budget."""
    
    if budget_limit == 0:
        return 'huggingface'  # Free local processing
    
    estimated_tokens = text_count * 100
    
    # Cost estimates (per token)
    openai_cost = estimated_tokens * 0.00000002  # $0.02 per 1M tokens
    mistral_cost = estimated_tokens * 0.0000001   # $0.1 per 1M tokens
    
    if openai_cost <= budget_limit:
        return 'openai'      # Cheapest API option
    elif mistral_cost <= budget_limit:
        return 'mistral'     # Higher cost but better rate limits
    else:
        return 'huggingface' # Free local option
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Found
```
Error: OpenAI API key not found in environment variable: OPENAI_API_KEY
```

**Solution**: Set the appropriate environment variable:
```bash
export OPENAI_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"
```

#### 2. Module Import Errors
```
ImportError: No module named 'sentence_transformers'
```

**Solution**: Install required dependencies:
```bash
pip install sentence-transformers  # For HuggingFace
pip install mistralai>=1.0.0       # For Mistral (use latest client)
```

#### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or switch to CPU:
```yaml
embedding:
  providers:
    huggingface:
      device: "cpu"          # Switch to CPU
      batch_size: 64         # Reduce batch size
```

#### 4. Rate Limit Exceeded
```
Error: Rate limit exceeded
```

**Solution**: Adjust rate limiting parameters:
```yaml
embedding:
  providers:
    mistral:
      rate_limit_delay: 1.0           # Increase delay
      max_requests_per_minute: 100    # Reduce request rate
```

#### 5. Dimension Mismatch
```
Error: Vector dimension mismatch: 1024 != 1536
```

**Solution**: This happens when switching providers. The system automatically handles this by creating new collections, but you may need to re-process existing data.

### Debug Mode

Enable verbose logging for troubleshooting:

```yaml
log_level: "DEBUG"
vector_diagnostics_verbose: true
```

### Health Check Script

```bash
python test_providers.py all  # Test all providers
python test_providers.py mistral  # Test specific provider
```

---

## Extending the System

### Adding a New Provider

1. **Create provider class**:

```python
# src/embedding_providers/cohere.py
from .base import EmbeddingProvider

class CohereEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config):
        import cohere
        self.client = cohere.Client(api_key=os.environ[config['api_key_env']])
        self.model = config['model']
        self.dimensions = config['dimensions']
    
    def get_embeddings(self, texts):
        response = self.client.embed(texts=texts, model=self.model)
        embeddings = [np.array(emb) for emb in response.embeddings]
        token_counts = [len(text.split()) * 1.3 for text in texts]
        return embeddings, token_counts
    
    def get_dimensions(self):
        return self.dimensions
    
    def get_model_name(self):
        return self.model
    
    def get_weaviate_vectorizer_config(self):
        return None  # Use custom vectors
    
    def supports_batch_api(self):
        return False
    
    def estimate_tokens(self, text):
        return len(text.split()) * 1.3
```

2. **Register the provider**:

```python
# src/embedding_providers/factory.py
_providers = {
    'openai': 'OpenAIEmbeddingProvider',
    'mistral': 'MistralEmbeddingProvider', 
    'huggingface': 'HuggingFaceEmbeddingProvider',
    'cohere': 'CohereEmbeddingProvider',  # Add here
}
```

3. **Add configuration**:

```yaml
embedding:
  providers:
    cohere:
      model: "embed-english-v3.0"
      dimensions: 1024
      api_key_env: "COHERE_API_KEY"
```

### Custom Vectorizer Integration

For providers that support Weaviate integration:

```python
def get_weaviate_vectorizer_config(self):
    try:
        from weaviate.classes.config import Configure
        return Configure.Vectorizer.text2vec_cohere(
            model=self.model,
            dimensions=self.dimensions
        )
    except AttributeError:
        return None  # Fall back to custom vectors
```

---

## API Reference

### EmbeddingProvider (Base Class)

Abstract base class that all providers must implement.

#### Methods

```python
def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]
    """Generate embeddings for texts."""

def get_dimensions(self) -> int
    """Return embedding dimensions."""

def get_model_name(self) -> str  
    """Return model identifier."""

def get_weaviate_vectorizer_config(self) -> Optional[Dict[str, Any]]
    """Return Weaviate vectorizer config or None for custom vectors."""

def supports_batch_api(self) -> bool
    """Whether provider supports batch processing."""

def estimate_tokens(self, text: str) -> int
    """Estimate token count for rate limiting."""
```

#### Utility Methods

```python
def validate_texts(self, texts: List[str]) -> List[str]
    """Validate and clean input texts."""

def chunk_texts(self, texts: List[str], chunk_size: int) -> List[List[str]]
    """Split texts into chunks for batch processing."""

def get_rate_limit_config(self) -> Dict[str, Any]
    """Get rate limiting configuration."""
```

### EmbeddingProviderFactory

Factory class for creating provider instances.

#### Class Methods

```python
@classmethod
def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> EmbeddingProvider
    """Create provider instance."""

@classmethod  
def create_from_full_config(cls, embedding_config: Dict[str, Any]) -> EmbeddingProvider
    """Create provider from full embedding configuration."""

@classmethod
def get_available_providers(cls) -> List[str]
    """Get list of available provider types."""

@classmethod
def register_provider(cls, provider_type: str, provider_class_name: str) -> None
    """Register new provider type."""
```

### OpenAIEmbeddingProvider

OpenAI-specific provider implementation.

#### Additional Features

- Comprehensive rate limiting (per-minute and daily)
- Batch API support
- tiktoken integration for accurate token counting
- API header monitoring
- Automatic fallback token estimation

### MistralEmbeddingProvider  

Mistral-specific provider implementation.

#### Additional Features

- Monthly token limiting (200B tokens/month)
- High-performance rate limiting (20M TPM)
- Automatic retry logic
- Optimized for Mistral's generous limits

### HuggingFaceEmbeddingProvider

Local model provider implementation.

#### Additional Features

- GPU/CPU device selection
- Model warmup functionality
- Tokenizer integration when available
- Memory-efficient batch processing
- Model information introspection

---

## Performance Tips

### Optimization Guidelines

1. **Choose the Right Provider**:
   - **Development**: HuggingFace (free, good quality)
   - **Production (cost-optimized)**: OpenAI (cheapest API at $0.02/M tokens)
   - **Production (accuracy-critical)**: OpenAI large models
   - **High-throughput (rate-limited)**: Mistral (20M TPM vs OpenAI's 5M TPM)

2. **Batch Size Tuning**:
   - **OpenAI**: 512 (balanced speed/memory)
   - **Mistral**: 500 (optimized for their limits)
   - **HuggingFace**: 256-512 (depends on GPU memory)

3. **GPU Utilization**:
   ```yaml
   embedding:
     providers:
       huggingface:
         device: "cuda"
         batch_size: 1024  # Larger batches on GPU
   ```

4. **Rate Limit Optimization**:
   - Mistral: Very generous limits, can be aggressive
   - OpenAI: Conservative defaults, tune based on your tier
   - HuggingFace: No limits, optimize for hardware

5. **Cost Optimization**:
   - Use HuggingFace for development/testing (free)
   - Use OpenAI for cost-effective production ($0.02/M tokens)
   - Use Mistral when you need higher rate limits despite higher cost ($0.1/M tokens)

---

## Best Practices

### Security

1. **API Key Management**:
   ```bash
   # Use environment variables, not config files
   export MISTRAL_API_KEY="$(cat ~/.mistral_key)"
   export OPENAI_API_KEY="$(cat ~/.openai_key)"
   ```

2. **Rate Limiting**:
   - Always use rate limiting in production
   - Monitor usage to avoid unexpected costs
   - Set monthly budgets and alerts

3. **Data Privacy**:
   - Use HuggingFace for sensitive data
   - Review provider privacy policies
   - Consider data residency requirements

### Production Deployment

1. **Monitoring**:
   ```python
   # Monitor provider performance
   provider_metrics = {
       'tokens_used': sum(token_counts),
       'embeddings_generated': len(embeddings), 
       'latency': end_time - start_time,
       'cost_estimate': calculate_cost(provider_name, sum(token_counts))
   }
   ```

2. **Fallback Strategy**:
   ```yaml
   # Primary provider
   embedding:
     provider: "mistral"
     
   # Fallback configuration
   fallback_embedding:
     provider: "huggingface"  # Always available
   ```

3. **Testing**:
   ```bash
   # Regular health checks
   python test_providers.py all
   
   # Performance benchmarking  
   python scripts/benchmark_providers.py
   ```

---

## Support and Contributing

### Getting Help

1. **Check logs**: Enable debug mode for detailed error information
2. **Run health checks**: Use the test scripts to verify setup
3. **Review configuration**: Ensure all required fields are set
4. **Check dependencies**: Verify all required packages are installed

### Contributing

1. **Adding Providers**: Follow the extension guide above
2. **Bug Reports**: Include provider type, configuration, and error logs
3. **Feature Requests**: Consider compatibility across all providers

### Roadmap

Planned features:
- [ ] Azure OpenAI provider
- [ ] Google Vertex AI provider  
- [ ] Anthropic Claude embeddings
- [ ] Automatic provider benchmarking
- [ ] Cost tracking and budgets
- [ ] Provider health monitoring
- [ ] Embedding caching layer

---

*Last updated: July 2025*