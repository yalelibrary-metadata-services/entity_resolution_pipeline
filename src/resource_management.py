"""
Resource Management Module for Entity Resolution

This module provides enhanced resource management capabilities for the entity resolution pipeline,
ensuring proper cleanup of connections and resources during processing and shutdown.
"""

import logging
import time
import traceback
import gc
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages resources for entity resolution pipeline, ensuring proper cleanup
    to prevent leaks and improve reliability in production environments.
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        self.registered_resources = {}
        self.cleanup_callbacks = {}
        
    def register_resource(self, resource_id: str, resource: Any, 
                         cleanup_callback: Optional[Callable] = None) -> None:
        """
        Register a resource to be managed.
        
        Args:
            resource_id: Unique identifier for the resource
            resource: The resource object to manage
            cleanup_callback: Optional function to call when cleaning up the resource
        """
        self.registered_resources[resource_id] = resource
        if cleanup_callback:
            self.cleanup_callbacks[resource_id] = cleanup_callback
        logger.debug(f"Registered resource: {resource_id}")
        
    def cleanup_resource(self, resource_id: str) -> bool:
        """
        Clean up a specific resource.
        
        Args:
            resource_id: ID of the resource to clean up
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        if resource_id not in self.registered_resources:
            logger.warning(f"Cannot clean up unknown resource: {resource_id}")
            return False
        
        try:
            resource = self.registered_resources[resource_id]
            
            # Use custom cleanup callback if available
            if resource_id in self.cleanup_callbacks:
                self.cleanup_callbacks[resource_id](resource)
            # Try standard cleanup methods
            elif hasattr(resource, 'close') and callable(resource.close):
                resource.close()
            elif hasattr(resource, 'cleanup') and callable(resource.cleanup):
                resource.cleanup()
            elif hasattr(resource, '__exit__'):
                resource.__exit__(None, None, None)
            
            # Remove from tracking
            del self.registered_resources[resource_id]
            if resource_id in self.cleanup_callbacks:
                del self.cleanup_callbacks[resource_id]
                
            logger.debug(f"Successfully cleaned up resource: {resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up resource {resource_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def cleanup_all(self) -> Dict[str, bool]:
        """
        Clean up all registered resources.
        
        Returns:
            Dictionary mapping resource IDs to cleanup success status
        """
        results = {}
        # Make a copy of keys since we'll be modifying the dictionary during iteration
        resource_ids = list(self.registered_resources.keys())
        
        for resource_id in resource_ids:
            results[resource_id] = self.cleanup_resource(resource_id)
            
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Cleaned up {sum(results.values())}/{len(results)} resources")
        return results
    
    def get_resource_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all registered resources.
        
        Returns:
            Dictionary with resource status information
        """
        status = {}
        
        for resource_id, resource in self.registered_resources.items():
            resource_type = type(resource).__name__
            has_cleanup = resource_id in self.cleanup_callbacks
            
            status[resource_id] = {
                "type": resource_type,
                "has_custom_cleanup": has_cleanup,
                "registered_time": time.time()
            }
            
        return status

@contextmanager
def managed_resource(resource_manager: ResourceManager, resource_id: str, 
                    resource: Any, cleanup_callback: Optional[Callable] = None):
    """
    Context manager for automatically managing resources.
    
    Args:
        resource_manager: ResourceManager instance
        resource_id: Unique identifier for the resource
        resource: The resource object to manage
        cleanup_callback: Optional function to call when cleaning up the resource
        
    Yields:
        The resource object
    """
    try:
        resource_manager.register_resource(resource_id, resource, cleanup_callback)
        yield resource
    finally:
        resource_manager.cleanup_resource(resource_id)

def cleanup_weaviate_client(client):
    """
    Special cleanup function for Weaviate client.
    
    Args:
        client: Weaviate client instance
    """
    try:
        # First try direct close method
        if hasattr(client, 'close'):
            client.close()
        # Then try to access the grpc client and close it
        elif hasattr(client, '_connection'):
            connection = client._connection
            if hasattr(connection, 'close'):
                connection.close()
        # Finally try context exit
        elif hasattr(client, '__exit__'):
            client.__exit__(None, None, None)
    except Exception as e:
        logger.warning(f"Error during weaviate client cleanup: {str(e)}")

def optimize_memory_usage(threshold_percent: float = 80.0, 
                         force_gc: bool = True) -> Dict[str, float]:
    """
    Optimize memory usage if it exceeds the threshold.
    
    Args:
        threshold_percent: Memory usage percentage threshold
        force_gc: Whether to force garbage collection
        
    Returns:
        Dictionary with memory statistics
    """
    try:
        import psutil
        
        # Get current memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        stats = {
            'initial_usage_percent': memory_percent,
            'available_mb_before': memory.available / (1024 * 1024),
            'threshold_percent': threshold_percent,
            'optimization_applied': False
        }
        
        # Check if optimization is needed
        if memory_percent > threshold_percent or force_gc:
            # Force garbage collection
            gc.collect()
            
            # Get updated memory usage
            memory = psutil.virtual_memory()
            
            stats['optimization_applied'] = True
            stats['final_usage_percent'] = memory.percent
            stats['available_mb_after'] = memory.available / (1024 * 1024)
            stats['memory_freed_mb'] = stats['available_mb_after'] - stats['available_mb_before']
            
            logger.info(f"Memory optimization performed: {stats['initial_usage_percent']:.1f}% -> {stats['final_usage_percent']:.1f}%")
        
        return stats
    except ImportError:
        logger.warning("psutil not available, memory optimization skipped")
        return {'error': 'psutil not available', 'optimization_applied': False}
