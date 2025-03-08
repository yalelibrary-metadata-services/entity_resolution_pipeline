import weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("UniqueStringsByField")

# Get count of objects
count = collection.aggregate.over_all(total_count=True)
print(f"Total objects in collection: {count.total_count}")

# Get a sample object
results = collection.query.fetch_objects(limit=1, include_vector=True)
for obj in results.objects:
    print(f"Sample object: {obj.properties}")
    print(f"Vector dimensions: {len(next(iter(obj.vector.values())))}")