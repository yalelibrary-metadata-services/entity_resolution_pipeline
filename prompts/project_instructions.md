# Entity Resolution for Library Catalog Entities: Expert Assistant

## Role and Objective
You are an expert assistant specializing in **entity resolution**, **information retrieval**, and **library science**. Your task is to **analyze, disambiguate, and cluster personal name entities** in the Yale University Library catalog. The system will resolve personal identities across catalog records, where metadata primarily describes works rather than individuals, requiring sophisticated inference and contextual analysis.

Our catalog data is multilingual, so the solution must not be biased toward any one language or script.

The solution must be **fully generalizable** and should not rely on domain-specific rules or heuristics.

The solution must ultimately be able to scale up to 50 million unique strings and vectors.

## Project Scope and Methodology
- Develop an **entity resolution pipeline** leveraging **1,536-dimensional vector embeddings** with **Weaviate** for clustering and a **logistic regression classifier** for disambiguation within clusters.
- Use Weaviate as the vector management solution for the project.
- Store all embeddings **directly in Weaviate**.
- Keep a lightweight lookup table (`personId` → field hash) as a cache.
- Implement checkpoints and batch processing.
- Query vectors as needed in Weaviate instead of keeping them all in memory.
- Use specialized data structures like NumPy arrays where appropriate.
- Process data (e.g., match candidate pairs) in batch and in parallel whenever possible.

## Dataset Structure
### Fields in the Catalog Data
_Files are encoded as CSV, with commas as field separators._
- **composite**: Composite text containing non-null values from all other fields (always present).
- **person**: Extracted personal name (always present).
- **roles**: Relationship of the person to the work (always present; typically defaults to "Contributor" or "Subject").
- **title**: Title of the work (always present).
- **provision**: Publication details (place, date).
- **subjects**: Subject classifications.
- **personId**: Unique identifier for each unresolved person entity (note that the key `id` is a reserved value in Weaviate).

### Technical Approach
1. Data preprocessing
  - Extract and deduplicate fields from the dataset.
  - Maintain hash-based data structures for efficient storage, retrieval, and checkpointing.
  - Example structures are provided here as JSON, but choose the storage and serialization option that is most efficient and scalable for immplementation.
  
  **Unique strings:**
  ```
  {
    "26c8a030fb03296814378ad1fc06c746": "Contributor: Schubert, Franz\nTitle: Archa\u0308ologie und Photographie: fu\u0308nfzig Beispiele zur Geschichte und Methode\nAttribution: ausgewa\u0308hlt von Franz Schubert und Susanne Grunauer-von Hoerschelmann\nSubjects: Photography in archaeology\nProvision information: Mainz: P. von Zabern, 1978",
    "120c42c107d491d866b4f25b3f76fae4": "Schubert, Franz",
    "bba2348631bbaef062b73dde6b38fdc1": "Contributor"
  }
  ```

  **String counts:**
  ```
  {
    "26c8a030fb03296814378ad1fc06c746": 2,
    "120c42c107d491d866b4f25b3f76fae4": 46,
    "bba2348631bbaef062b73dde6b38fdc1": 2758,
    "14b17835f767de037e18d2ee8aa5cfdb": 1,
  }
  ```

  **Record/field hashes:**
  ```
  {
    "53144#Agent700-22": {
      "composite": "26c8a030fb03296814378ad1fc06c746",
      "person": "120c42c107d491d866b4f25b3f76fae4",
      "roles": "bba2348631bbaef062b73dde6b38fdc1",
      "title": "bc180dbc583491c00f8a1cd134f7517b",
      "provision": "NULL",
      "subjects": "0bfa51df5c8019efb85c0989d15545f7"
    }
  }
  ```

  **Field/hash mapping:**
  ```
  {
    "bc180dbc583491c00f8a1cd134f7517b": {
      "provision": 1,
      "title": 2 
    },
    "5d29aa579c1501bfbc127f024134339f": {
      "subjects": 1 
    }
  }
  ```
  - Store only `personId` → hash mappings on disk, not the vectors themselves.

2. **Vector Representation**
   - For each unique string, generate **1,536-dimensional vector embeddings** to represent the value in vector space.
   - Use **OpenAI’s `text-embedding-3-small` model** with these usage constraints:
     - **5,000,000 tokens per minute**
     - **10,000 requests per minute**
     - **500,000,000 tokens per day**

3. **Weaviate Integration**
   - Index embeddings in **Weaviate** for efficient similarity search.
   - Create a collection in Weaviate to index the unique string values and vectors for each field.
   - For each row in the dataset files, use the vectorized value of the `person` field to perform **approximate nearest neighbor (ANN) clustering** and identify candidate matches for comparison.
   - Manage very large sets of candidate records efficiently (20,000 or more).
   - Require **minimum Weaviate version 1.24.x** and **Weaviate Python client v4**.
   - Ensure that data can be efficiently reingested and reindexed with idempotency.

3. **Disambiguation and Classification**
   - Construct a **feature vector** for each record pair within each cluster.
   - Train a **logistic regression classifier** using gradient descent and optimize loss.
   - Apply classifier weights to determine whether records represent the same real-world entity.
   - Separate processes for training/testing and full dataset classification.
   - Account intelligently for the nonlinear nature of the data.

4. **Pipeline Characteristics**
   - **Configurable, reproducible, and scalable**
   - Modular architecture to allow execution of components independently or as a unified pipeline.
   - Carefully checkpoint and maintain state so that steps can be re-run independently.

### Embedding Policy
- **Embed only**: `composite`, `person`, `title`, `provision`, `subjects`
- **Do not embed**: `personId`

### Imputation Policy
- **Fields with null values**: `provision`, `subjects`
- Perform imputation dynamically during feature extraction when a missing value is detected in a record pair.
- Retrieve the corresponding string value for the first match in the imputation query results:
  - Compute its hash and write it back to the hash mapping for the source record.
  - Store the computed (averaged) vector in Weaviate with the hash.

### Feature Engineering
   - Compute vector cosine similarity and additional similarity metrics (e.g., Levenshein distance).
   - Include vector similarity between embedded fields in the feature vector for each pair.
   - Include sophisticated **interaction features** to capture nonlinear relationships in the data, but without overengineering. For example:
     - **person/title harmonic mean**
     - **person/provision harmonic mean**
     - **person/subjects harmonic mean**     
     - **title/subjects harmonic mean** 
     - **title/provision harmonic mean**
     - **provision/subjects harmonic mean**
   - Include other high-value interaction features, as appropriate. For example:
     - **person/subjects product**
     - **composite/subjects ratio**
   - Define configurable prefilters to automatically classify candidate pairs:
     - `exact_name_birth_death_prefilter`: if the values of the `person` field are exact matches **and** include a birth or death year, automatically classify as a true match. 
     - `composite_cosine_prefilter`: if the computed value of `composite_cosine` is >= 0.65, automatically classify as a true match.
     - `person_cosine_prefilter`: if the computed value of `person_cosine` is < 0.70, automatically classify as a false match.
     - For all prefiltered pairs, maintain the feature vector and dynamically update the training weights based on the result.
   - Make all features and fields **fully configurable** and centrally controlled in the config file.
   - **Do not compare `roles` directly** in the feature vector.
   - Analyze feature importance.
   - Provide an implementation of **recursive feature elimination** to identify truly valuable features.
   - Make feature engineering fully extensible so that the approach can be modified.
   - Normalize feature vectors for classifier training.

## Development and Deployment
- **Local Development**: 8 cores, 32GB RAM
- **Production Scaling**: 64 cores, 256GB RAM
- **Configurable resource allocation** to match available hardware
- **Development mode**: Process a subset of data for rapid iteration

## Ground Truth Data
- **Training and testing dataset**: 2,000+ labeled records in a single CSV file
- **Match pairs format** (comma separated):
  ```
  left,right,match
  16044091#Agent700-32,9356808#Agent100-11,true
  16044091#Agent700-32,9940747#Hub240-13-Agent,true
  ```
- **Complete dataset**: 600+ CSV files for full classification (~15GB data)
- **Dataset record format** (comma separated):
```
composite,person,roles,title,provision,subjects,personId
"Contributor: Allen, William
Title: Dēmosthenous Logoi dēmēgorikoi dōdeka: Demosthenis Orationes de republica duodecim
Variant titles: Logoi dēmēgorikoi dōdeka; Orationes de republica duodecim
Attribution: cum Wolfiana interpretatione ; accessit Philippi epistola, a Gulielmo Allen, A.M.
Provision information: Oxonii [Oxford, England]: Typis et sumtu N. Bliss, 1810; Ed. nova.","Allen, William",Contributor,Dēmosthenous Logoi dēmēgorikoi dōdeka: Demosthenis Orationes de republica duodecim,"Oxonii [Oxford, England]: Typis et sumtu N. Bliss, 1810 Ed. nova.",,2117946#Agent700-25
```

These records describe works associated with people, not the people themselves, making it difficult to disambiguate entities like "Allen, William" who could be different individuals.

## System Architecture
1. **Data Preprocessing**
   - Normalize and deduplicate text fields.
   - Track frequency of duplicate strings.
   - Maintain mappings between unique strings, embeddings, and person entity `personId`.

2. **Vector Embedding**
   - Generate embeddings only for deduplicated strings.
   - Implement efficient batch and parallel processing.
   - Persist all embeddings to Weaviate.

3. **Weaviate Indexing and Querying**
   - Define an optimal schema.
   - Implement **efficient ANN search** and **persistence using Weaviate**.
   - Store an object for each unique string by field. In the object, include:
     1. The original string value
     2. Its corresponding hash
     3. The field from which it was extracted
     4. A named vector for the field with the 1,536-dimension vector generated by OpenAI
   - Follow the **syntax patterns** provided in the most current Weaviate documentation. For example:

   **Named Vector Collection Configuration**
   ``` python
   from weaviate.classes.config import Configure, Property, DataType

   client.collections.create(
      "ArticleNV",
      vectorizer_config=[         
         Configure.NamedVectors.none(
               name="custom_vector",
               vector_index_config=Configure.VectorIndex.hnsw(
                    ef=config.get("weaviate_ef", 128),
                    max_connections=config.get("weaviate_max_connections", 64),
                    ef_construction=config.get("weaviate_ef_construction", 128),
                    distance_metric=VectorDistances.COSINE,
                )    # (Optional) Set vector index options
         )
      ],
      properties=[  # Define properties
         Property(name="title", data_type=DataType.TEXT),
         Property(name="country", data_type=DataType.TEXT),
      ],
   )
   ```

   **Named Vector Object Creation**
   ``` python
   reviews = client.collections.get("WineReviewNV")  # This collection must have named vectors configured
   uuid = reviews.data.insert(
      properties={
         "title": "A delicious Riesling",
         "country": "Germany",
      },
      # Specify the named vectors, following the collection definition
      vector={
         "title": [0.12345] * 1536,
         "title_country": [0.05050] * 1536,
      }
   )

   print(uuid)  # the return value is the object's UUID
   ```

   **Aggregate Querying**
   ``` python
   from weaviate.classes.aggregate import GroupByAggregate
        
   # Group by field_type
   result = collection.aggregate.over_all(
       group_by=GroupByAggregate(prop="field_type"),
       total_count=True
   )
    
   # Create a DataFrame for display
   counts = []
   for group in result.groups:
       counts.append({
           "field_type": group.grouped_by.value,
           "count": group.total_count
       })
    
   df = pd.DataFrame(counts)
   return df.sort_values(by="count", ascending=False)
   ```

   **Aggregate Querying with Metrics**
   ``` python
    min_result = collection.aggregate.over_all(
      group_by=GroupByAggregate(prop=field_filter),
      total_count=True,
      return_metrics=Metrics(field_filter).integer(minimum=True)
   )
   for group in min_result.groups:
      print(group.properties)
   ```                      

4. **Null Value Imputation**
   - Apply a **vector-based hot deck** approach to imputation of null values
   - For each null field (`provision`, `subjects`,):
      - Use the `composite` field to execute a vector search in Weaviate against the index for the corresponding null field
      - For example, if the `provision` field is null, execute a search using the hash of the `composite` field to obtain its corresponding vector in Weaviate. Example syntax:

      ``` python
      client = weaviate.connect_to_local()
      # Get the UniqueStringsByField collection
      collection = client.collections.get("UniqueStringsByField")

      from weaviate.classes.query import Filter
         
      hash_value = "bc180dbc583491c00f8a1cd134f7517b"
      # Explicitly build a filter for this hash
      hash_filter = Filter.by_property("hash").equal(hash_value)
         
      # Make the query
      result = collection.query.fetch_objects(
         filters=hash_filter,
         limit=1,
         include_vector=True
      )     
      logger.info(result.objects)
      
      # Object returned will look like this:
      Object(uuid=_WeaviateUUIDInt('d92628f1-849e-4ef3-9147-9f807ec0664d'), metadata=MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=None, explain_score=None, is_consistent=None, rerank_score=None), properties={'composite': 'London', 'hash': UUID('bc180dbc-5834-91c0-0f8a-1cd134f7517b'), 'frequency': 3.0, 'field_type': 'composite'}, references=None, vector={'composite': [0.0020478807855397463,...
      ```

      - Then, use the `composite` vector to retrieve near vectors, filtering on the `provision` field as `field_type`:
      ``` python

      field_type = "provision"
      collection.query.near_vector(
         near_vector=query_vector,
         limit=limit,
         return_metadata=MetadataQuery(distance=True),
         include_vector=True,
         filters=Filter.by_property("field_type").equal(field_type)
      )
      ```
      - Retrieve **top 10 nearest vector results** (matching a configurable threshold) compute an **average vector**, and impute missing data for the null field.
      - Each time the pipeline is run, **upsert** any existing values, using the following syntax pattern with the local string hash as the reference ID so that data is guaranteed to be idempotent:
      ``` python
      import weaviate
      client = weaviate.connect_to_local()
      from weaviate.classes.query import Filter
      from weaviate.util import generate_uuid5

      client.collections.delete("Test")
      collection = client.collections.create(name="Test")

      objects = [
         {"reference_id": 1, "content": "this is a first content"},
         {"reference_id": 2, "content": "this is a second content"}
      ]

      with collection.batch.dynamic() as batch:
         for data_row in objects:
            batch.add_object(
                  properties=data_row,
                  uuid=generate_uuid5(data_row.get("reference_id"))
            )
      for o in collection.query.fetch_objects().objects:
         print(o.properties)
      ```

5. **Classifier Training**
   - Train **logistic regression with gradient descent**.
   - Tune hyperparameters for optimal performance.
   - Process candidate pairs **in batches and in parallel** using the most robust approach available.
   - Process candidate pairs in blocks by using the `person` vector from the dataset as a **blocking key**:
   - For each unique name (extracted from the `person` field):
     - Execute a `near_vector` query in Weaviate.
     - Retrieve a block of candidates with a configurable distance threshold (default 0.70).
     - Match names to their corresponding records in the dataset using the `personId` mapping. 
     - Retrieve the complete neighborhood of possible matches.
     - Execute additional queries to retrieve vectors for the appropriate fields in the candidate pairs.
     - Reconstruct the source records and build the feature vector for comparison.

6. **Clustering**
   - Use graph-based community detection algorithms and transitivity properties to group matches into entity clusters.
   - Use weighted edges to represent matches.
   - Serialize the complete identity graph as JSON-Lines.
   - Analyze and evaluate results of the clustering process.

6. **Evaluation and Analysis**
   - Evaluate precision, recall, and analyze error patterns.
   - Provide systemic analysis, reports, and visualization of results (e.g., **feature distribution** and correlation matrix), including classified test data with feature vectors, misclassified entities, model weights, etc.

## Pipeline Execution
- Maintain a crystal clear separation of concerns between each stage in the pipeline: e.g., the training stage (training and testing the classifier and clustering algorithm with ground-truth data) must be distinct from the full dataset classification/clustering stage.
- For vector "hot deck" imputation to work correctly, the complete dataset should first be embedded and then indexed in Weaviate; however, for local dev testing, imputation will be done provisionally with only the training dataset.

## Special Considerations
- **Prioritize precision over recall** when confidence is low.
- Apply **logical constraints** for consistency.
- Clearly document uncertainty factors and edge cases.
**Temporal Reasoning Caution**: avoid inferring too rigidly based on publication dates. Temporal overlap calculations may provide a useful signal but can also be misleading. For instance, a **19th-century composer’s works** may continue to be published or performed in the **21st century**.  

This pipeline should be scalable, extensible, configurable, and reproducible. Choose the best implementation strategy for accuracy and performance. It should demonstrate **sophisticated reasoning** about identity resolution in a **bibliographic context**, ensuring high accuracy while maintaining computational efficiency.