"""
Integration Examples for Sprint 4.3 Scheduling Module

Demonstrates how to use GPU Memory Manager, Batch Scheduler, and Resource Allocator together.

Author: APEX, NEXUS
Date: January 7, 2026
"""

from src.scheduling import (
    create_memory_manager,
    create_scheduler,
    create_allocator,
    create_request,
    ResourceType,
    PriorityClass,
    create_alloc_request,
)
import time


# =============================================================================
# Example 1: Basic Scheduling with Memory Management
# =============================================================================

def example_1_basic_scheduling():
    """
    Basic example: Submit requests to scheduler and allocate GPU memory.
    
    Shows:
    - Creating scheduler and memory manager
    - Submitting requests
    - Forming batches
    - Allocating memory for batch execution
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Scheduling with Memory Management")
    print("="*80)
    
    # Create scheduler and memory manager
    scheduler = create_scheduler(max_batch_size=4, latency_target_p99_ms=500.0)
    memory = create_memory_manager(device_id=0, total_memory_gb=1.0)
    
    # Submit requests
    print("\n📤 Submitting 10 requests...")
    for i in range(10):
        req = create_request(
            request_id=f"req{i}",
            tenant_id=f"tenant{i % 3}",
            sequence_length=512,
            max_new_tokens=64 + (i % 3) * 32
        )
        scheduler.submit(req)
    
    print(f"   Queue length: {scheduler.queue_length}")
    
    # Form batches
    batch_num = 0
    while scheduler.queue_length > 0:
        decision = scheduler.schedule()
        batch_num += 1
        
        if decision:
            print(f"\n🔄 Batch {batch_num}:")
            print(f"   Requests: {len(decision.requests)}")
            print(f"   Policy: {decision.policy_used.name}")
            print(f"   Max latency: {decision.max_token_latency_ms:.1f}ms")
            
            # Calculate memory needed
            total_tokens = sum(r.max_new_tokens for r in decision.requests)
            memory_needed = total_tokens * 2048  # 2KB per token estimate
            
            # Allocate memory
            result = memory.allocate(
                size=memory_needed,
                tenant_id=decision.requests[0].tenant_id
            )
            
            if result.success:
                print(f"   💾 Memory: {memory_needed / 1024 / 1024:.1f}MB allocated")
            else:
                print(f"   ❌ Memory allocation failed")
            
            # Simulate execution
            time.sleep(0.1)
            for req in decision.requests:
                scheduler.record_completion(
                    request_id=req.request_id,
                    actual_latency_ms=150.0 + (req.sequence_length / 100),
                    tokens_generated=req.max_new_tokens
                )
    
    # Print metrics
    metrics = scheduler.metrics
    print(f"\n📊 Scheduler Metrics:")
    print(f"   Total requests: {metrics.total_requests_scheduled}")
    print(f"   Total batches: {metrics.total_batches_formed}")
    print(f"   Avg batch size: {metrics.total_requests_scheduled / metrics.total_batches_formed if metrics.total_batches_formed > 0 else 0:.1f}")


# =============================================================================
# Example 2: Multi-Tenant Resource Allocation
# =============================================================================

def example_2_multi_tenant_allocation():
    """
    Multi-tenant example: Fair resource allocation across tenants.
    
    Shows:
    - Registering nodes and tenants
    - Setting quotas
    - Allocating resources with fair sharing
    - Computing fairness metrics
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Tenant Resource Allocation with Fair Sharing")
    print("="*80)
    
    # Create allocator
    allocator = create_allocator(fair_share="drf")
    
    # Register nodes
    print("\n🖥️  Registering nodes...")
    for i in range(2):
        allocator.register_node(f"gpu{i}", {
            ResourceType.GPU_MEMORY: 8 * 1024 * 1024 * 1024,  # 8GB
            ResourceType.GPU_COMPUTE: 1.0
        })
    print(f"   Registered 2 nodes")
    
    # Register tenants with quotas
    print("\n👥 Registering tenants with quotas...")
    for i in range(3):
        tenant_id = f"tenant{i}"
        allocator.register_tenant(tenant_id, priority=PriorityClass.NORMAL)
        allocator.update_tenant_quota(
            tenant_id,
            ResourceType.GPU_MEMORY,
            guaranteed=2 * 1024 * 1024 * 1024,  # 2GB guaranteed
            limit=5 * 1024 * 1024 * 1024        # 5GB limit
        )
    print(f"   Registered 3 tenants")
    
    # Allocate resources
    print("\n💾 Allocating resources for 20 requests...")
    allocations = []
    for i in range(20):
        request = create_alloc_request(
            request_id=f"req{i}",
            tenant_id=f"tenant{i % 3}",
            gpu_memory=512 * 1024 * 1024  # 512MB per request
        )
        
        grant = allocator.allocate(request)
        if grant:
            allocations.append(grant)
            if i % 5 == 0:
                print(f"   [{i+1}/20] Allocated to {grant.node_id} " +
                      f"(quality: {grant.quality_score:.2f})")
    
    print(f"\n✅ Allocated {len(allocations)} out of 20 requests")
    
    # Compute fair shares
    print("\n📊 Fair Share Analysis:")
    fair_shares = allocator.compute_fair_shares()
    for tenant_id, shares in fair_shares.items():
        mem_share = shares.get(ResourceType.GPU_MEMORY, 0) / (1024 * 1024 * 1024)
        print(f"   {tenant_id}: {mem_share:.1f}GB fair share")
    
    # Get fairness metrics
    metrics = allocator.get_metrics()
    print(f"\n   Fairness Index: {metrics.fairness_index:.3f} (target: >0.9)")
    
    utilization = allocator.get_cluster_utilization()
    print(f"   Cluster Utilization: {utilization.get(ResourceType.GPU_MEMORY, 0)*100:.1f}%")


# =============================================================================
# Example 3: Memory Pressure Handling
# =============================================================================

def example_3_memory_pressure():
    """
    Memory pressure example: Handle memory pressure with defragmentation.
    
    Shows:
    - Monitoring memory pressure
    - Allocating to trigger pressure
    - Automatic defragmentation
    - Quota enforcement under pressure
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Memory Pressure Handling & Defragmentation")
    print("="*80)
    
    # Create memory manager with callbacks
    memory = create_memory_manager(device_id=0, total_memory_gb=0.5)  # Small for demo
    
    print("\n💾 Starting memory allocation to trigger pressure...")
    
    blocks = []
    for i in range(5):
        result = memory.allocate(
            size=60 * 1024 * 1024,  # 60MB
            tenant_id=f"tenant{i % 2}"
        )
        
        if result.success:
            blocks.append(result.block)
            stats = memory.get_stats()
            print(f"   Allocation {i+1}: {stats.utilization*100:.1f}% " +
                  f"({stats.pressure_level.name})")
        else:
            print(f"   Allocation {i+1}: FAILED (out of memory)")
            break
    
    # Check stats
    print("\n📊 Memory Statistics:")
    stats = memory.get_stats()
    print(f"   Total: {stats.total_bytes / 1024 / 1024:.0f}MB")
    print(f"   Allocated: {stats.allocated_bytes / 1024 / 1024:.0f}MB")
    print(f"   Available: {stats.available_bytes / 1024 / 1024:.0f}MB")
    print(f"   Utilization: {stats.utilization*100:.1f}%")
    print(f"   Pressure Level: {stats.pressure_level.name}")
    print(f"   Fragmentation: {stats.fragmentation_ratio*100:.1f}%")
    
    # Deallocate to lower pressure
    print("\n🧹 Deallocating memory...")
    for block in blocks[:3]:
        memory.deallocate(block)
    
    stats = memory.get_stats()
    print(f"   After deallocation:")
    print(f"   Utilization: {stats.utilization*100:.1f}%")
    print(f"   Pressure Level: {stats.pressure_level.name}")


# =============================================================================
# Example 4: Scheduling Policy Adaptation
# =============================================================================

def example_4_policy_adaptation():
    """
    Policy adaptation example: Show ML-based policy selection.
    
    Shows:
    - Scheduler adapting policies based on workload
    - Thompson Sampling for exploration/exploitation
    - Latency prediction improvement
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Scheduling Policy Adaptation (ML-Based)")
    print("="*80)
    
    # Create adaptive scheduler
    scheduler = create_scheduler(enable_adaptive=True, max_batch_size=8)
    
    print("\n🤖 Using ML-based adaptive scheduling...")
    
    # Simulate multiple workload patterns
    workload_patterns = [
        ("LIGHT", 5, 128),      # 5 reqs, 128 tokens each
        ("HEAVY", 20, 256),     # 20 reqs, 256 tokens each
        ("MIXED", 12, 192),     # 12 reqs, 192 tokens each
    ]
    
    for pattern_name, num_reqs, token_size in workload_patterns:
        print(f"\n📈 {pattern_name} Workload:")
        
        # Submit requests
        for i in range(num_reqs):
            req = create_request(
                request_id=f"{pattern_name}_{i}",
                tenant_id="tenant1",
                sequence_length=512,
                max_new_tokens=token_size
            )
            scheduler.submit(req)
        
        # Schedule until queue empty
        batches = 0
        while scheduler.queue_length > 0:
            decision = scheduler.schedule()
            if decision:
                batches += 1
                
                # Record fake execution
                for req in decision.requests:
                    actual_latency = 100.0 + (req.sequence_length / 50)
                    scheduler.record_completion(
                        request_id=req.request_id,
                        actual_latency_ms=actual_latency,
                        tokens_generated=req.max_new_tokens
                    )
        
        print(f"   Policy used: {scheduler._current_policy.name}")
        print(f"   Batches formed: {batches}")


# =============================================================================
# Example 5: End-to-End Integrated System
# =============================================================================

def example_5_integrated_system():
    """
    Complete integrated system: All components working together.
    
    Shows:
    - Creating a complete serving system
    - Handling multi-tenant requests
    - Resource allocation with fair sharing
    - Memory management with quota enforcement
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Integrated End-to-End Serving System")
    print("="*80)
    
    # Initialize all components
    print("\n🚀 Initializing serving system...")
    scheduler = create_scheduler(max_batch_size=8, enable_adaptive=True)
    allocator = create_allocator(fair_share="drf", overbooking=1.2)
    memory = create_memory_manager(device_id=0, total_memory_gb=2.0)
    
    # Register infrastructure
    allocator.register_node("gpu0", {
        ResourceType.GPU_MEMORY: 8 * 1024 * 1024 * 1024,
        ResourceType.GPU_COMPUTE: 1.0
    })
    
    # Register tenants
    print("   Registering tenants...")
    for i in range(2):
        allocator.register_tenant(f"app{i}")
        allocator.update_tenant_quota(
            f"app{i}",
            ResourceType.GPU_MEMORY,
            guaranteed=1 * 1024 * 1024 * 1024,
            limit=3 * 1024 * 1024 * 1024
        )
    
    # Simulate requests
    print("\n📨 Processing request stream...")
    request_stream = [
        (0, "app0", 128),
        (1, "app1", 64),
        (2, "app0", 256),
        (3, "app1", 128),
        (4, "app0", 192),
        (5, "app1", 96),
    ]
    
    for req_id, tenant, tokens in request_stream:
        # Create and submit request
        req = create_request(
            request_id=f"req{req_id}",
            tenant_id=tenant,
            sequence_length=512,
            max_new_tokens=tokens
        )
        scheduler.submit(req)
        
        # Try to form batch if queue building up
        if scheduler.queue_length >= 3:
            decision = scheduler.schedule()
            
            if decision:
                print(f"\n   🔄 Batch formed ({len(decision.requests)} reqs):")
                
                # Allocate resources
                for batch_req in decision.requests:
                    alloc_req = create_alloc_request(
                        request_id=batch_req.request_id,
                        tenant_id=batch_req.tenant_id,
                        gpu_memory=batch_req.max_new_tokens * 2048
                    )
                    
                    grant = allocator.allocate(alloc_req)
                    if grant:
                        print(f"      ✅ {batch_req.request_id} allocated")
                    else:
                        print(f"      ❌ {batch_req.request_id} failed")
                
                # Simulate inference
                time.sleep(0.05)
                
                # Record completions
                for batch_req in decision.requests:
                    scheduler.record_completion(
                        request_id=batch_req.request_id,
                        actual_latency_ms=150.0,
                        tokens_generated=batch_req.max_new_tokens
                    )
    
    # Final stats
    print("\n📊 System Statistics:")
    metrics = allocator.get_metrics()
    print(f"   Allocations: {metrics.successful_allocations} successful")
    print(f"   Fairness: {metrics.fairness_index:.3f}")
    
    sched_metrics = scheduler.metrics
    print(f"   Scheduler: {sched_metrics.total_batches_formed} batches formed")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPRINT 4.3 SCHEDULING MODULE - INTEGRATION EXAMPLES")
    print("="*80)
    
    try:
        example_1_basic_scheduling()
        example_2_multi_tenant_allocation()
        example_3_memory_pressure()
        example_4_policy_adaptation()
        example_5_integrated_system()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
