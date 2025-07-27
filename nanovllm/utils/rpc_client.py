import torch
import torch.distributed.rpc as rpc
from typing import Any, Dict, List, Optional, Union
import logging
import threading
import time
from dataclasses import dataclass
import pickle
import io

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


logger = logging.getLogger(__name__)


@dataclass
class RPCRequest:
    """RPC request structure."""
    method: str
    args: List[Any]
    kwargs: Dict[str, Any]
    request_id: str
    timestamp: float


@dataclass
class RPCResponse:
    """RPC response structure."""
    request_id: str
    result: Any
    error: Optional[str] = None
    processing_time: float = 0.0


class RPCServer:
    """RPC server for distributed inference."""
    
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.initialized = False
        self.worker_id = f"worker_{rank}"
        
    def initialize(self):
        """Initialize RPC server."""
        try:
            rpc.init_rpc(
                self.worker_id,
                rank=self.rank,
                world_size=self.config.world_size
            )
            self.initialized = True
            logger.info(f"RPC server initialized for {self.worker_id}")
        except Exception as e:
            logger.error(f"RPC initialization failed for {self.worker_id}: {e}")
            raise
    
    def shutdown(self):
        """Shutdown RPC server."""
        if self.initialized:
            rpc.shutdown()
            self.initialized = False
            logger.info(f"RPC server shutdown for {self.worker_id}")
    
    def process_inference_request(
        self,
        sequences: List[Sequence],
        is_prefill: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Process inference request via RPC."""
        try:
            start_time = time.time()
            
            # Import here to avoid circular imports
            from nanovllm.engine.model_runner import ModelRunner
            from nanovllm.engine.multimodal_model_runner import MultiModalModelRunner
            
            # Initialize model runner
            if self.config.enable_multimodal:
                model_runner = MultiModalModelRunner(self.config)
            else:
                model_runner = ModelRunner(self.config)
            
            # Process sequences
            token_ids = model_runner.run(sequences, is_prefill)
            
            # Prepare response
            response = {
                'token_ids': token_ids,
                'rank': self.rank,
                'model_stats': self._get_model_stats(model_runner),
                'processing_time': time.time() - start_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"RPC inference failed: {e}")
            return {'error': str(e), 'rank': self.rank}
    
    def _get_model_stats(self, model_runner) -> Dict[str, Any]:
        """Get model statistics."""
        stats = {
            'device': str(next(model_runner.model.parameters()).device),
            'memory_usage': self._get_memory_usage()
        }
        
        if hasattr(model_runner, 'get_vision_stats'):
            stats.update(model_runner.get_vision_stats())
        
        return stats
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """Get GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        cached = torch.cuda.memory_reserved(device)
        
        return {
            'total_memory': total,
            'allocated_memory': allocated,
            'cached_memory': cached,
            'free_memory': total - cached
        }
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get worker information."""
        return {
            'rank': self.rank,
            'world_size': self.config.world_size,
            'tensor_parallel_size': self.config.tensor_parallel_size,
            'pipeline_parallel_size': self.config.pipeline_parallel_size,
            'is_multimodal': self.config.enable_multimodal
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for this worker."""
        return {
            'status': 'healthy',
            'rank': self.rank,
            'timestamp': time.time()
        }


class RPCClient:
    """RPC client for communicating with distributed workers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.initialized = False
        self.workers = {}
        
    def initialize(self):
        """Initialize RPC client."""
        try:
            rpc.init_rpc(
                "client",
                rank=0,
                world_size=self.config.world_size
            )
            self.initialized = True
            logger.info("RPC client initialized")
        except Exception as e:
            logger.error(f"RPC client initialization failed: {e}")
            raise
    
    def shutdown(self):
        """Shutdown RPC client."""
        if self.initialized:
            rpc.shutdown()
            self.initialized = False
            logger.info("RPC client shutdown")
    
    def call_worker(
        self,
        worker_rank: int,
        method: str,
        *args,
        **kwargs
    ) -> Any:
        """Call method on specific worker."""
        if not self.initialized:
            self.initialize()
        
        worker_id = f"worker_{worker_rank}"
        try:
            # Create remote reference
            remote_ref = rpc.remote(worker_id, RPCServer)
            
            # Call method
            result = remote_ref.rpc_sync().forward(
                method,
                args,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"RPC call to worker {worker_rank} failed: {e}")
            raise
    
    def broadcast_request(
        self,
        method: str,
        *args,
        **kwargs
    ) -> Dict[int, Any]:
        """Broadcast request to all workers."""
        if not self.initialized:
            self.initialize()
        
        results = {}
        
        for rank in range(1, self.config.world_size):
            try:
                result = self.call_worker(rank, method, *args, **kwargs)
                results[rank] = result
            except Exception as e:
                logger.warning(f"Worker {rank} failed: {e}")
                results[rank] = {'error': str(e)}
        
        return results
    
    def gather_results(self, results: Dict[int, Any]) -> List[Any]:
        """Gather results from all workers."""
        if not self.initialized:
            self.initialize()
        
        gathered = []
        for rank, result in results.items():
            if 'error' not in result:
                gathered.append(result)
        
        return gathered
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the distributed cluster."""
        if not self.initialized:
            self.initialize()
        
        info = {
            'world_size': self.config.world_size,
            'workers': {}
        }
        
        for rank in range(1, self.config.world_size):
            try:
                worker_info = self.call_worker(rank, 'get_worker_info')
                info['workers'][rank] = worker_info
            except Exception as e:
                info['workers'][rank] = {'error': str(e)}
        
        return info
    
    def health_check_cluster(self) -> Dict[int, Dict[str, Any]]:
        """Health check all workers in cluster."""
        if not self.initialized:
            self.initialize()
        
        health_status = {}
        
        for rank in range(1, self.config.world_size):
            try:
                health = self.call_worker(rank, 'health_check')
                health_status[rank] = health
            except Exception as e:
                health_status[rank] = {'status': 'unhealthy', 'error': str(e)}
        
        return health_status
    
    def load_balance_request(
        self,
        sequences: List[Sequence],
        is_prefill: bool
    ) -> Dict[int, List[Sequence]]:
        """Distribute sequences across workers using load balancing."""
        if not self.initialized:
            self.initialize()
        
        # Simple round-robin load balancing
        worker_assignments = {}
        
        for i, seq in enumerate(sequences):
            worker_rank = (i % (self.config.world_size - 1)) + 1
            if worker_rank not in worker_assignments:
                worker_assignments[worker_rank] = []
            worker_assignments[worker_rank].append(seq)
        
        return worker_assignments
    
    def submit_distributed_task(
        self,
        sequences: List[Sequence],
        is_prefill: bool
    ) -> Dict[int, Any]:
        """Submit task to distributed workers."""
        if not self.initialized:
            self.initialize()
        
        # Load balance sequences
        worker_assignments = self.load_balance_request(sequences, is_prefill)
        
        # Submit tasks to workers
        results = {}
        
        for worker_rank, worker_seqs in worker_assignments.items():
            try:
                result = self.call_worker(
                    worker_rank,
                    'process_inference_request',
                    worker_seqs,
                    is_prefill
                )
                results[worker_rank] = result
            except Exception as e:
                logger.error(f"Task submission to worker {worker_rank} failed: {e}")
                results[worker_rank] = {'error': str(e)}
        
        return results


class SimpleRPCServer:
    """Simple RPC server without torch.distributed.rpc dependency."""
    
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.server_socket = None
        
    def start_server(self, port: int):
        """Start simple RPC server."""
        import socket
        import threading
        import json
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen(5)
        
        def handle_client(client_socket):
            try:
                data = client_socket.recv(4096).decode()
                request = json.loads(data)
                
                # Process request
                if request['method'] == 'process_inference':
                    result = self.process_inference_request(
                        request['sequences'],
                        request['is_prefill']
                    )
                elif request['method'] == 'health_check':
                    result = self.health_check()
                else:
                    result = {'error': 'Unknown method'}
                
                response = json.dumps(result).encode()
                client_socket.send(response)
                
            except Exception as e:
                client_socket.send(json.dumps({'error': str(e)}).encode())
            finally:
                client_socket.close()
        
        def server_loop():
            while True:
                client_socket, addr = self.server_socket.accept()
                client_thread = threading.Thread(target=handle_client, args=(client_socket,))
                client_thread.start()
        
        server_thread = threading.Thread(target=server_loop, daemon=True)
        server_thread.start()
        logger.info(f"Simple RPC server started on port {port}")


class RPCManager:
    """High-level RPC manager for distributed inference."""
    
    def __init__(self, config: Config):
        self.config = config
        self.rpc_client = RPCClient(config)
        self.rpc_servers = {}
        self.active_workers = []
        
    def initialize_cluster(self):
        """Initialize entire RPC cluster."""
        if not self.config.enable_distributed:
            return
        
        # Initialize RPC client
        self.rpc_client.initialize()
        
        # Start RPC servers on workers
        for rank in range(1, self.config.world_size):
            try:
                server = RPCServer(self.config, rank)
                server.initialize()
                self.rpc_servers[rank] = server
                self.active_workers.append(rank)
            except Exception as e:
                logger.error(f"Failed to initialize server for rank {rank}: {e}")
        
        logger.info(f"RPC cluster initialized with {len(self.active_workers)} workers")
    
    def shutdown_cluster(self):
        """Shutdown entire RPC cluster."""
        for server in self.rpc_servers.values():
            server.shutdown()
        
        self.rpc_client.shutdown()
        logger.info("RPC cluster shutdown completed")
    
    def distribute_and_execute(
        self,
        sequences: List[Sequence],
        is_prefill: bool
    ) -> Dict[int, List[int]]:
        """Distribute sequences and execute across cluster."""
        if not self.active_workers:
            return {}
        
        # Use RPC client for distribution
        results = self.rpc_client.submit_distributed_task(sequences, is_prefill)
        
        # Collect token IDs
        token_ids_map = {}
        for worker_rank, result in results.items():
            if 'error' not in result:
                for i, token_id in enumerate(result['token_ids']):
                    seq_id = sequences[i].seq_id
                    token_ids_map[seq_id] = token_id
        
        return token_ids_map