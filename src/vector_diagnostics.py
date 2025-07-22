"""
Vector Persistence Diagnostic Module

Implements comprehensive diagnostic tools for analyzing vector persistence issues
in the Weaviate integration pipeline. Provides detailed analysis of vector
serialization, transmission, and storage characteristics.
"""

import os
import sys
import uuid
import json
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.connect import ConnectionParams
from weaviate.util import generate_uuid5

logger = logging.getLogger(__name__)

class VectorDiagnosticTool:
    """Advanced diagnostic utilities for vector persistence troubleshooting."""
    
    def __init__(self, client, config=None):
        """
        Initialize diagnostic tool with Weaviate client instance.
        
        Args:
            client: Initialized Weaviate client
            config: Optional configuration dictionary
        """
        self.client = client
        self.config = config or {}
        self.embedding_dimensions = self.config.get("embedding_dimensions", 1536)
        self.verbose = self.config.get("vector_diagnostics_verbose", False)  # Default to minimal logging for background processing
        self.results = {}
    
    def debug_vector_transmission(self, item_vector, hash_value, field_type):
        """
        Comprehensive vector diagnostic and transmission validation.
        
        Args:
            item_vector: Vector data (numpy array or list)
            hash_value: Hash identifier for the vector
            field_type: Field type (person, title, composite)
            
        Returns:
            List of properly formatted vector values
        """
        # Vector type validation and normalization
        vector_type = type(item_vector).__name__
        vector_shape = getattr(item_vector, "shape", (len(item_vector),)) if hasattr(item_vector, "__len__") else "Unknown"
        vector_sample = str(item_vector[:5]) if hasattr(item_vector, "__getitem__") else "Not indexable"
        
        # Serialization test with multiple formats
        formats = {}
        
        # Raw list conversion
        if hasattr(item_vector, "flatten"):
            raw_list = list(item_vector.flatten())
        elif hasattr(item_vector, "__iter__"):
            raw_list = list(item_vector)
        else:
            raw_list = []
            logger.error(f"Vector cannot be converted to list: {type(item_vector)}")
        
        formats["raw_list"] = raw_list
        
        # Float list conversion (explicit type conversion)
        try:
            float_list = [float(x) for x in raw_list]
            formats["float_list"] = float_list
        except Exception as e:
            logger.error(f"Failed to convert vector to float list: {str(e)}")
            formats["float_list"] = []
        
        # JSON serialization test
        try:
            json_serialized = json.dumps(formats["float_list"])
            formats["json_serialized"] = json_serialized
        except Exception as e:
            logger.error(f"Failed to JSON serialize vector: {str(e)}")
            formats["json_serialized"] = ""
        
        # Log diagnostics based on verbosity setting
        if self.verbose:
            # Detailed diagnostics (original behavior)
            logger.info(f"VECTOR DEBUG [{hash_value}] ===========================")
            logger.info(f"  Type: {vector_type}, Shape: {vector_shape}")
            logger.info(f"  Sample: {vector_sample}")
            logger.info(f"  Field: {field_type}")
            
            for format_name, format_data in formats.items():
                if isinstance(format_data, (list, tuple)):
                    logger.info(f"  {format_name}: {type(format_data).__name__}, Length: {len(format_data)}")
                    if len(format_data) > 0:
                        logger.info(f"    First element type: {type(format_data[0]).__name__}")
                        logger.info(f"    First 5 elements: {format_data[:5]}")
                else:
                    logger.info(f"  {format_name}: {type(format_data).__name__}, Length: {len(format_data) if hasattr(format_data, '__len__') else 'N/A'}")
        else:
            # Essential progress tracking only - use debug level to avoid flooding logs
            status = "SUCCESS" if len(formats.get("float_list", [])) > 0 else "FAILED"
            logger.debug(f"Vector processed: {field_type} [{hash_value[:8]}...] - {status}")
            
            # Only log failures at info level for troubleshooting
            if status == "FAILED":
                logger.warning(f"Vector processing FAILED for {field_type} [{hash_value[:8]}...]")
        
        return formats["float_list"]
    
    def test_direct_vector_persistence(self):
        """
        Test vector persistence with direct API calls.
        
        Returns:
            bool: True if test succeeds, False otherwise
        """
        logger.info("Running direct vector persistence test")
        
        try:
            # Access collection
            collection = self.client.collections.get("EntityString")
            
            # Create test vector with explicit float values
            test_vector = [float(0.1 * i) for i in range(self.embedding_dimensions)]
            test_hash = f"test_vector_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Generated test vector with {len(test_vector)} dimensions")
            logger.info(f"Test hash: {test_hash}")
            
            # Attempt direct object creation
            try:
                obj_uuid = collection.data.insert(
                    properties={
                        "hash_value": test_hash,
                        "field_type": "test",
                        "original_string": "Test vector persistence",
                        "frequency": 1
                    },
                    vector=test_vector
                )
                
                logger.info(f"Test object created with UUID: {obj_uuid}")
                
                # Retrieve and verify
                result = collection.query.fetch_objects(
                    filters=Filter.by_property("hash_value").equal(test_hash),
                    include_vector=True
                ).objects
                
                if result:
                    logger.info(f"Retrieved {len(result)} objects for hash {test_hash}")
                    
                    obj = result[0]
                    if hasattr(obj, 'vector'):
                        if not obj.vector:
                            logger.error("Vector attribute exists but is empty")
                            self.log_object_structure(obj)
                            return False
                        
                        # Check vector type and content
                        vector_type = type(obj.vector).__name__
                        
                        if isinstance(obj.vector, dict):
                            logger.info(f"Vector stored as dictionary with keys: {list(obj.vector.keys())}")
                            vector_len = len(obj.vector['default'])
                            logger.info(f"Vector stored as list with length: {vector_len}")
                        elif isinstance(obj.vector, list):
                            vector_len = len(obj.vector)
                            logger.info(f"Vector stored as list with length: {vector_len}")
                            logger.info(f"First 5 elements: {obj.vector[:5]}")
                        else:
                            logger.info(f"Vector stored as {vector_type}")
                            vector_len = getattr(obj.vector, "__len__", lambda: "unknown")()
                        
                        logger.info(f"Verification result: Vector type={vector_type}, length={vector_len}")
                        return True
                    else:
                        logger.error("No vector attribute found on retrieved object")
                        self.log_object_structure(obj)
                        return False
                else:
                    logger.error(f"No objects found for hash {test_hash}")
                    return False
            except Exception as e:
                logger.error(f"Error inserting test object: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Test vector persistence failed: {str(e)}")
            return False
    
    def log_object_structure(self, obj):
        """Log detailed object structure for diagnostic purposes."""
        logger.info("Object structure debug:")
        logger.info(f"  Type: {type(obj).__name__}")
        logger.info(f"  Attributes: {dir(obj)}")
        logger.info(f"  Properties: {getattr(obj, 'properties', {})}")
        
        if hasattr(obj, '__dict__'):
            logger.info(f"  __dict__: {obj.__dict__}")
    
    def verify_vector_persistence(self, verification_samples):
        """
        Verify vector data persistence after batch operations.
        
        Args:
            verification_samples: List of sample objects to verify
            
        Returns:
            bool: True if verification succeeds for any sample
        """
        if not verification_samples:
            logger.warning("No verification samples provided")
            return False
        
        if self.verbose:
            logger.info(f"Verifying vector persistence for {len(verification_samples)} samples")
        else:
            logger.debug(f"Verifying vector persistence for {len(verification_samples)} samples")
        
        # Get EntityString collection
        try:
            collection = self.client.collections.get("EntityString")
        except Exception as e:
            logger.error(f"Failed to access EntityString collection: {str(e)}")
            return False
        
        success_count = 0
        
        for sample in verification_samples:
            hash_val = sample['hash_value']
            field_type = sample['field_type']
            
            try:
                # Query for object by hash value
                query_filter = Filter.by_property("hash_value").equal(hash_val)
                results = collection.query.fetch_objects(
                    filters=query_filter,
                    include_vector=True  # Explicitly request vector
                ).objects
                
                if not results:
                    logger.error(f"Verification failed: No object found for hash {hash_val}")
                    continue
                    
                obj = results[0]
                
                # Inspect vector property structure
                if not hasattr(obj, 'vector'):
                    logger.error(f"Verification failed: Object has no vector attribute")
                elif not obj.vector:
                    logger.error(f"Verification failed: Object has empty vector")
                    self.log_object_structure(obj)
                else:
                    vector_type = type(obj.vector).__name__
                    
                    if isinstance(obj.vector, dict):
                        if self.verbose:
                            logger.info(f"Vector stored as dictionary with keys: {list(obj.vector.keys())}")
                        if obj.vector:  # Check if dictionary is not empty
                            success_count += 1
                    elif isinstance(obj.vector, list):
                        vector_len = len(obj.vector)
                        if self.verbose:
                            logger.info(f"Vector stored as list with length: {vector_len}")
                        if vector_len > 0:
                            success_count += 1
                    else:
                        vector_len = getattr(obj.vector, "__len__", lambda: "unknown")()
                        if self.verbose:
                            logger.info(f"Vector stored as {vector_type} with length: {vector_len}")
                        success_count += 1
                        
                    if self.verbose:
                        logger.info(f"Verification success: Vector stored for {hash_val} (Type: {vector_type})")
            except Exception as e:
                logger.error(f"Error during vector verification for {hash_val}: {str(e)}")
        
        if success_count > 0:
            if self.verbose:
                logger.info(f"Vector persistence verified for {success_count}/{len(verification_samples)} samples")
            return True
        else:
            if self.verbose:
                logger.error("Vector persistence verification failed for all samples")
            return False
    
    def validate_weaviate_compatibility(self):
        """
        Validate Weaviate version and client compatibility.
        
        Returns:
            dict: Compatibility information
        """
        compatibility_info = {}
        
        try:
            # Check Weaviate version
            meta = self.client.get_meta()
            version = meta.get("version", "unknown")
            
            logger.info(f"Weaviate server version: {version}")
            compatibility_info["server_version"] = version
            
            # Check client version
            client_version = weaviate.__version__
            logger.info(f"Weaviate client version: {client_version}")
            compatibility_info["client_version"] = client_version
            
            # Verify compatibility
            if version != "unknown" and client_version != "unknown":
                server_parts = version.split('.')
                client_parts = client_version.split('.')
                
                if len(server_parts) >= 2 and len(client_parts) >= 2:
                    server_major, server_minor = server_parts[:2]
                    client_major, client_minor = client_parts[:2]
                    
                    if server_major != client_major:
                        logger.warning(f"Major version mismatch: Server {server_major} vs Client {client_major}")
                        compatibility_info["version_compatible"] = False
                    else:
                        compatibility_info["version_compatible"] = True
            
            # Verify module availability
            modules = meta.get("modules", {})
            logger.info(f"Available modules: {list(modules.keys())}")
            compatibility_info["modules"] = list(modules.keys())
            
            return compatibility_info
        except Exception as e:
            logger.error(f"Compatibility validation failed: {str(e)}")
            compatibility_info["error"] = str(e)
            return compatibility_info
    
    def run_full_diagnostics(self):
        """
        Run comprehensive vector persistence diagnostics.
        
        Returns:
            dict: Diagnostic results
        """
        logger.info("Starting comprehensive vector persistence diagnostics")
        
        self.results = {
            "compatibility": self.validate_weaviate_compatibility(),
            "direct_persistence_test": None,
            "timestamp": self._get_timestamp()
        }
        
        # Run direct vector persistence test
        test_result = self.test_direct_vector_persistence()
        self.results["direct_persistence_test"] = test_result
        
        # Log summary
        logger.info("Vector persistence diagnostic summary:")
        logger.info(f"  Compatibility check: {'Success' if self.results['compatibility'].get('version_compatible', False) else 'Warning'}")
        logger.info(f"  Direct persistence test: {'Success' if test_result else 'Failed'}")
        
        return self.results
    
    def _get_timestamp(self):
        """Get formatted timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Execute vector diagnostic tool."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Vector persistence diagnostic tool")
    parser.add_argument("--host", default="localhost", help="Weaviate host")
    parser.add_argument("--port", type=int, default=8080, help="Weaviate HTTP port")
    parser.add_argument("--grpc-port", type=int, default=50051, help="Weaviate gRPC port")
    parser.add_argument("--output", default="data/output/vector_diagnostic.json", help="Output file path")
    args = parser.parse_args()
    
    try:
        # Initialize Weaviate client
        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host=args.host,
                http_port=args.port,
                grpc_host=args.host,
                grpc_port=args.grpc_port
            )
        )
        
        # Connect to Weaviate
        client.connect()
        
        # Initialize and run diagnostic tool
        diagnostics = VectorDiagnosticTool(client, {
            "embedding_dimensions": 1536
        })
        
        results = diagnostics.run_full_diagnostics()
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Diagnostic results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Diagnostic execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
