# Weaviate Search Diagnostic

`weaviate-search-diagnostic.py` is a CLI tool for exploring the Weaviate vector
index (`EntityString` collection) that holds embeddings for the entity-resolution
data. It lets you run similarity searches by `personId` or blindly sample the
index to see what clusters together.

## Usage

```bash
# Vector search for a specific personId
python weaviate-search-diagnostic.py <personId>
python weaviate-search-diagnostic.py <personId> --limit 20
python weaviate-search-diagnostic.py <personId> --limit 50 --distance 0.3

# Random title-vector sampling with near_vector searches
python weaviate-search-diagnostic.py --sample
python weaviate-search-diagnostic.py --sample --size 5 --search-limit 10
```

## How it works

The tool has two modes, dispatched in `main()`.

### 1. Search mode — `python weaviate-search-diagnostic.py <personId>`

- `search()` takes a `personId`, looks up its **composite** field vector, and
  runs a similarity search to find the nearest neighbors.
- `get_composite_vector()` maps `personId → composite hash` via `hash_lookup`,
  then queries Weaviate filtering on that `hash_value` + `field_type ==
  "composite"` with `include_vector=True` to pull back the actual embedding.
- `perform_near_vector_search()` runs `near_vector` with that embedding, keeps
  only `composite` matches, resolves each hit back to its text (`string_dict`)
  and `personId` (reverse scan of `hash_lookup`), and reports distances.

### 2. Sample mode — `python weaviate-search-diagnostic.py --sample`

- `get_random_title_vectors_and_search()` fetches a batch of `title` objects,
  randomly samples some, and uses each title's vector as a query to find
  neighbors — a way to blindly explore what's clustered together in the index,
  annotated with subjects.

Both modes finally print human-readable output, and `cleanup()` closes the
client.
