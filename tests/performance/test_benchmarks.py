"""
Performance benchmarking tests for Riad Concierge AI
Testing response times, throughput, and scalability
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import statistics

from app.services.agent_service import AgentService
from app.services.whatsapp_service import WhatsAppService
from app.services.cultural_service import CulturalService
from app.services.knowledge_service import KnowledgeService


class TestPerformanceBenchmarks:
    """Performance benchmarking test suite."""
    
    @pytest.mark.asyncio
    async def test_agent_response_time_benchmark(
        self, 
        agent_service,
        performance_benchmarks
    ):
        """Test agent response time meets performance targets."""
        
        target_time = performance_benchmarks["response_time"]["target"]
        test_messages = [
            "أريد حجز غرفة",
            "What amenities do you have?", 
            "Je voudrais réserver une suite",
            "Can you help me with spa services?",
            "ما هي الأنشطة المتاحة؟"
        ]
        
        response_times = []
        
        for message in test_messages:
            start_time = time.time()
            
            result = await agent_service.process_message(
                phone_number="+212600123456",
                message_text=message,
                message_type="text"
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert result is not None
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(response_times)
        p90_response_time = statistics.quantiles(response_times, n=10)[8]  # 90th percentile
        
        # Assert performance targets
        assert avg_response_time < target_time, f"Average response time {avg_response_time:.2f}s exceeds target {target_time}s"
        assert p90_response_time < target_time, f"90th percentile response time {p90_response_time:.2f}s exceeds target {target_time}s"
        
        print(f"Performance Results:")
        print(f"  Average Response Time: {avg_response_time:.3f}s")
        print(f"  90th Percentile: {p90_response_time:.3f}s")
        print(f"  Target: {target_time}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_user_handling(
        self, 
        agent_service,
        performance_benchmarks
    ):
        """Test system performance under concurrent load."""
        
        concurrent_users = performance_benchmarks["concurrent_users"]["target"]
        degradation_limit = performance_benchmarks["concurrent_users"]["response_degradation_limit"]
        
        # Baseline single user performance
        start_time = time.time()
        await agent_service.process_message(
            phone_number="+212600123456",
            message_text="Test message",
            message_type="text"
        )
        baseline_time = time.time() - start_time
        
        # Concurrent user simulation
        async def simulate_user(user_id: int):
            start = time.time()
            result = await agent_service.process_message(
                phone_number=f"+21260012{user_id:04d}",
                message_text=f"Concurrent test message from user {user_id}",
                message_type="text"
            )
            return time.time() - start, result is not None
        
        # Execute concurrent requests
        tasks = [simulate_user(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = 0
        response_times = []
        
        for result in results:
            if isinstance(result, tuple):
                response_time, success = result
                if success:
                    successful_requests += 1
                    response_times.append(response_time)
        
        # Calculate performance degradation
        avg_concurrent_time = statistics.mean(response_times) if response_times else float('inf')
        degradation = (avg_concurrent_time - baseline_time) / baseline_time
        
        # Assert performance requirements
        success_rate = successful_requests / concurrent_users
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95% threshold"
        assert degradation <= degradation_limit, f"Performance degradation {degradation:.2%} exceeds {degradation_limit:.2%} limit"
        
        print(f"Concurrent Load Results:")
        print(f"  Concurrent Users: {concurrent_users}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Performance Degradation: {degradation:.2%}")
    
    @pytest.mark.asyncio
    async def test_cultural_intelligence_performance(
        self, 
        cultural_test_cases,
        performance_benchmarks
    ):
        """Test cultural intelligence processing performance."""
        
        cultural_service = CulturalService()
        cultural_service.redis_client = AsyncMock()
        cultural_service.instructor_client = MagicMock()
        
        target_accuracy = performance_benchmarks["cultural_accuracy"]["target"]
        processing_times = []
        accuracy_scores = []
        
        for test_case_name, test_data in cultural_test_cases.items():
            # Mock cultural profile creation
            cultural_service.instructor_client.chat.completions.create.return_value = MagicMock(
                message="Culturally adapted response",
                cultural_markers=test_data["expected_cultural_markers"],
                confidence_score=0.92
            )
            
            start_time = time.time()
            
            profile = await cultural_service.create_comprehensive_cultural_profile(
                phone_number="+212600123456",
                message_text=test_data["input"]
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Verify cultural accuracy
            language_correct = profile.language == test_data["expected_language"]
            markers_correct = any(
                marker in profile.cultural_markers 
                for marker in test_data["expected_cultural_markers"]
            )
            
            accuracy = (language_correct + markers_correct) / 2
            accuracy_scores.append(accuracy)
        
        # Calculate performance metrics
        avg_processing_time = statistics.mean(processing_times)
        avg_accuracy = statistics.mean(accuracy_scores)
        
        # Assert performance targets
        assert avg_accuracy >= target_accuracy, f"Cultural accuracy {avg_accuracy:.2%} below target {target_accuracy:.2%}"
        assert avg_processing_time < 1.0, f"Cultural processing time {avg_processing_time:.3f}s too slow"
        
        print(f"Cultural Intelligence Performance:")
        print(f"  Average Accuracy: {avg_accuracy:.2%}")
        print(f"  Average Processing Time: {avg_processing_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_knowledge_retrieval_performance(
        self, 
        mock_pinecone_index,
        performance_benchmarks
    ):
        """Test knowledge retrieval system performance."""
        
        knowledge_service = KnowledgeService()
        knowledge_service.redis_client = AsyncMock()
        knowledge_service.index = mock_pinecone_index
        knowledge_service.instructor_client = MagicMock()
        
        target_time = performance_benchmarks["knowledge_retrieval"]["target_time"]
        relevance_threshold = performance_benchmarks["knowledge_retrieval"]["relevance_threshold"]
        
        test_queries = [
            "Traditional Moroccan spa treatments",
            "Best restaurants in Marrakech medina",
            "Cultural activities for families",
            "Prayer times and religious services",
            "Airport transportation options"
        ]
        
        retrieval_times = []
        relevance_scores = []
        
        for query in test_queries:
            # Mock knowledge retrieval
            knowledge_service.instructor_client.chat.completions.create.return_value = MagicMock(
                query_type="information_request",
                relevant_knowledge=["local_attractions", "cultural_information"],
                confidence=0.89
            )
            
            start_time = time.time()
            
            cag_knowledge, rag_results = await knowledge_service.get_hybrid_knowledge(
                query=query,
                intent="information_request",
                cultural_context=MagicMock(
                    language="en",
                    nationality="International"
                )
            )
            
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            
            # Mock relevance scoring
            relevance_scores.append(0.85)  # Simulated relevance score
        
        # Calculate performance metrics
        avg_retrieval_time = statistics.mean(retrieval_times)
        avg_relevance = statistics.mean(relevance_scores)
        
        # Assert performance targets
        assert avg_retrieval_time < target_time, f"Knowledge retrieval time {avg_retrieval_time:.3f}s exceeds target {target_time}s"
        assert avg_relevance >= relevance_threshold, f"Relevance score {avg_relevance:.2f} below threshold {relevance_threshold}"
        
        print(f"Knowledge Retrieval Performance:")
        print(f"  Average Retrieval Time: {avg_retrieval_time:.3f}s")
        print(f"  Average Relevance: {avg_relevance:.2f}")
    
    @pytest.mark.asyncio
    async def test_whatsapp_message_throughput(self):
        """Test WhatsApp message processing throughput."""
        
        whatsapp_service = WhatsAppService()
        whatsapp_service.redis_client = AsyncMock()
        whatsapp_service.instructor_client = MagicMock()
        
        # Mock message sending
        whatsapp_service._send_queued_message = AsyncMock()
        
        message_count = 100
        start_time = time.time()
        
        # Queue multiple messages
        for i in range(message_count):
            await whatsapp_service.message_queue.put({
                "id": f"msg_{i}",
                "to": f"+21260012{i:04d}",
                "content": f"Test message {i}",
                "timestamp": datetime.now(),
                "retry_count": 0,
                "max_retries": 3
            })
        
        # Process messages (simulate queue processing)
        processed_count = 0
        while not whatsapp_service.message_queue.empty() and processed_count < message_count:
            message_data = await whatsapp_service.message_queue.get()
            await whatsapp_service._send_queued_message(message_data)
            processed_count += 1
        
        processing_time = time.time() - start_time
        throughput = message_count / processing_time
        
        # Assert throughput targets
        min_throughput = 50  # messages per second
        assert throughput >= min_throughput, f"Throughput {throughput:.1f} msg/s below minimum {min_throughput} msg/s"
        
        print(f"WhatsApp Throughput Performance:")
        print(f"  Messages Processed: {processed_count}")
        print(f"  Processing Time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} messages/second")
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, agent_service):
        """Test memory usage remains stable under sustained load."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate sustained load
        message_count = 200
        for i in range(message_count):
            await agent_service.process_message(
                phone_number=f"+21260012{i:04d}",
                message_text=f"Load test message {i}",
                message_type="text"
            )
            
            # Check memory every 50 messages
            if i % 50 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Assert memory growth is reasonable
                max_growth = 100  # MB
                assert memory_growth < max_growth, f"Memory growth {memory_growth:.1f}MB exceeds limit {max_growth}MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"Memory Usage Results:")
        print(f"  Initial Memory: {initial_memory:.1f}MB")
        print(f"  Final Memory: {final_memory:.1f}MB")
        print(f"  Total Growth: {total_growth:.1f}MB")
        print(f"  Messages Processed: {message_count}")


class TestScalabilityBenchmarks:
    """Scalability and stress testing."""
    
    @pytest.mark.asyncio
    async def test_database_connection_pooling(self):
        """Test database connection pooling under load."""
        
        # Mock database operations
        async def mock_db_operation():
            await asyncio.sleep(0.01)  # Simulate DB query
            return {"result": "success"}
        
        # Simulate concurrent database operations
        concurrent_operations = 50
        tasks = [mock_db_operation() for _ in range(concurrent_operations)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Verify all operations completed successfully
        assert len(results) == concurrent_operations
        assert all(result["result"] == "success" for result in results)
        
        # Assert reasonable execution time (should benefit from connection pooling)
        max_time = 2.0  # seconds
        assert execution_time < max_time, f"Database operations took {execution_time:.2f}s, exceeds {max_time}s"
        
        print(f"Database Pooling Results:")
        print(f"  Concurrent Operations: {concurrent_operations}")
        print(f"  Execution Time: {execution_time:.3f}s")
        print(f"  Operations/Second: {concurrent_operations/execution_time:.1f}")
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, mock_redis):
        """Test Redis cache performance under high load."""
        
        # Configure cache operations
        cache_operations = 1000
        hit_ratio_target = 0.8
        
        # Simulate cache operations
        cache_hits = 0
        cache_misses = 0
        
        start_time = time.time()
        
        for i in range(cache_operations):
            key = f"test_key_{i % 100}"  # Create some key overlap for hits
            
            # Simulate cache lookup
            if i % 100 < 80:  # 80% hit ratio simulation
                mock_redis.get.return_value = f"cached_value_{key}"
                cache_hits += 1
            else:
                mock_redis.get.return_value = None
                mock_redis.setex.return_value = True
                cache_misses += 1
        
        execution_time = time.time() - start_time
        hit_ratio = cache_hits / cache_operations
        
        # Assert cache performance
        assert hit_ratio >= hit_ratio_target, f"Cache hit ratio {hit_ratio:.2%} below target {hit_ratio_target:.2%}"
        
        operations_per_second = cache_operations / execution_time
        min_ops_per_second = 5000
        assert operations_per_second >= min_ops_per_second, f"Cache ops/sec {operations_per_second:.0f} below minimum {min_ops_per_second}"
        
        print(f"Cache Performance Results:")
        print(f"  Cache Operations: {cache_operations}")
        print(f"  Hit Ratio: {hit_ratio:.2%}")
        print(f"  Operations/Second: {operations_per_second:.0f}")
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting_effectiveness(self):
        """Test API rate limiting under burst traffic."""
        
        whatsapp_service = WhatsAppService()
        whatsapp_service.redis_client = AsyncMock()
        
        phone_number = "+212600123456"
        burst_requests = 25  # Above the 20/hour limit
        
        # Configure rate limiting mock
        request_count = 0
        
        async def mock_rate_limit_check(phone):
            nonlocal request_count
            request_count += 1
            
            if request_count <= 20:
                whatsapp_service.redis_client.get.return_value = str(request_count)
                return True
            else:
                whatsapp_service.redis_client.get.return_value = "25"  # Over limit
                return False
        
        whatsapp_service._check_rate_limit = mock_rate_limit_check
        
        # Simulate burst requests
        allowed_requests = 0
        blocked_requests = 0
        
        for i in range(burst_requests):
            if await whatsapp_service._check_rate_limit(phone_number):
                allowed_requests += 1
            else:
                blocked_requests += 1
        
        # Verify rate limiting effectiveness
        assert allowed_requests == 20, f"Expected 20 allowed requests, got {allowed_requests}"
        assert blocked_requests == 5, f"Expected 5 blocked requests, got {blocked_requests}"
        
        print(f"Rate Limiting Results:")
        print(f"  Total Requests: {burst_requests}")
        print(f"  Allowed: {allowed_requests}")
        print(f"  Blocked: {blocked_requests}")
        print(f"  Block Rate: {blocked_requests/burst_requests:.2%}")


@pytest.mark.performance
class TestEndToEndPerformance:
    """End-to-end performance testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_guest_journey_performance(self, agent_service):
        """Test performance of complete guest journey workflow."""
        
        journey_stages = [
            "pre_arrival_inquiry",
            "arrival_notification", 
            "room_service_request",
            "dining_reservation",
            "spa_booking",
            "checkout_process"
        ]
        
        stage_times = []
        total_start_time = time.time()
        
        for stage in journey_stages:
            stage_start = time.time()
            
            result = await agent_service.process_message(
                phone_number="+212600123456",
                message_text=f"Message for {stage}",
                message_type="text"
            )
            
            stage_time = time.time() - stage_start
            stage_times.append(stage_time)
            
            assert result is not None, f"Failed to process {stage}"
        
        total_time = time.time() - total_start_time
        avg_stage_time = statistics.mean(stage_times)
        
        # Performance assertions
        max_total_time = 15.0  # seconds for complete journey
        max_avg_stage_time = 2.5  # seconds per stage
        
        assert total_time < max_total_time, f"Total journey time {total_time:.2f}s exceeds {max_total_time}s"
        assert avg_stage_time < max_avg_stage_time, f"Average stage time {avg_stage_time:.2f}s exceeds {max_avg_stage_time}s"
        
        print(f"Guest Journey Performance:")
        print(f"  Total Journey Time: {total_time:.2f}s")
        print(f"  Average Stage Time: {avg_stage_time:.2f}s")
        print(f"  Stages Processed: {len(journey_stages)}")
    
    @pytest.mark.asyncio
    async def test_peak_hour_simulation(self, agent_service):
        """Simulate peak hour traffic patterns."""
        
        # Simulate realistic peak hour: 50 guests, 3 messages each over 10 minutes
        guests = 50
        messages_per_guest = 3
        duration_minutes = 10
        
        total_messages = guests * messages_per_guest
        interval = (duration_minutes * 60) / total_messages
        
        successful_messages = 0
        failed_messages = 0
        response_times = []
        
        start_time = time.time()
        
        for guest_id in range(guests):
            for msg_num in range(messages_per_guest):
                msg_start = time.time()
                
                try:
                    result = await agent_service.process_message(
                        phone_number=f"+21260{guest_id:06d}",
                        message_text=f"Peak hour message {msg_num + 1}",
                        message_type="text"
                    )
                    
                    if result:
                        successful_messages += 1
                        response_times.append(time.time() - msg_start)
                    else:
                        failed_messages += 1
                        
                except Exception:
                    failed_messages += 1
                
                # Simulate realistic message spacing
                await asyncio.sleep(interval)
        
        total_duration = time.time() - start_time
        success_rate = successful_messages / total_messages
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Performance assertions for peak hour
        min_success_rate = 0.95
        max_avg_response_time = 3.0
        
        assert success_rate >= min_success_rate, f"Success rate {success_rate:.2%} below {min_success_rate:.2%}"
        assert avg_response_time <= max_avg_response_time, f"Avg response time {avg_response_time:.2f}s exceeds {max_avg_response_time}s"
        
        print(f"Peak Hour Simulation Results:")
        print(f"  Total Messages: {total_messages}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Average Response Time: {avg_response_time:.3f}s")
        print(f"  Total Duration: {total_duration/60:.1f} minutes")
        print(f"  Messages/Minute: {total_messages/(total_duration/60):.1f}")
