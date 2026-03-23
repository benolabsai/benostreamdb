"""
MinIO setup and configuration for benchmarks.
"""
import subprocess
import time
import os
from typing import Optional


class MinIOManager:
    """Manage MinIO instance for benchmarks."""
    
    def __init__(
        self,
        port: int = 9000,
        console_port: int = 9001,
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        data_dir: Optional[str] = None
    ):
        self.port = port
        self.console_port = console_port
        self.access_key = access_key
        self.secret_key = secret_key
        self.data_dir = data_dir or "/tmp/minio-benchmark-data"
        self.process = None
        self.container_name = "hyperstreamdb-benchmark-minio"
    
    def start(self, use_docker: bool = True):
        """Start MinIO server."""
        if use_docker:
            self._start_docker()
        else:
            self._start_binary()
        
        # Wait for MinIO to be ready
        self._wait_for_ready()
        
        print(f"✓ MinIO started at http://localhost:{self.port}")
        print(f"  Access key: {self.access_key}")
        print(f"  Secret key: {self.secret_key}")
    
    def _start_docker(self):
        """Start MinIO using Docker."""
        # Stop existing container if running
        subprocess.run(
            ["docker", "stop", self.container_name],
            capture_output=True
        )
        subprocess.run(
            ["docker", "rm", self.container_name],
            capture_output=True
        )
        
        # Start new container
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", f"{self.port}:9000",
            "-p", f"{self.console_port}:9001",
            "-e", f"MINIO_ROOT_USER={self.access_key}",
            "-e", f"MINIO_ROOT_PASSWORD={self.secret_key}",
            "-v", f"{self.data_dir}:/data",
            "minio/minio",
            "server", "/data",
            "--console-address", ":9001"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start MinIO: {result.stderr}")
        
        print(f"Started MinIO container: {self.container_name}")
    
    def _start_binary(self):
        """Start MinIO using binary (if installed)."""
        os.makedirs(self.data_dir, exist_ok=True)
        
        env = os.environ.copy()
        env["MINIO_ROOT_USER"] = self.access_key
        env["MINIO_ROOT_PASSWORD"] = self.secret_key
        
        cmd = [
            "minio", "server", self.data_dir,
            "--address", f":{self.port}",
            "--console-address", f":{self.console_port}"
        ]
        
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"Started MinIO process (PID: {self.process.pid})")
    
    def _wait_for_ready(self, timeout: int = 30):
        """Wait for MinIO to be ready."""
        import requests
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/minio/health/live")
                if response.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.5)
        
        raise RuntimeError(f"MinIO did not start within {timeout} seconds")
    
    def stop(self, use_docker: bool = True):
        """Stop MinIO server."""
        if use_docker:
            subprocess.run(["docker", "stop", self.container_name], capture_output=True)
            subprocess.run(["docker", "rm", self.container_name], capture_output=True)
            print(f"✓ Stopped MinIO container")
        elif self.process:
            self.process.terminate()
            self.process.wait(timeout=10)
            print(f"✓ Stopped MinIO process")
    
    def get_endpoint(self) -> str:
        """Get MinIO endpoint URL."""
        return f"http://localhost:{self.port}"
    
    def get_credentials(self) -> tuple:
        """Get access credentials."""
        return (self.access_key, self.secret_key)
    
    def create_bucket(self, bucket_name: str):
        """Create a bucket in MinIO."""
        import boto3
        
        s3_client = boto3.client(
            's3',
            endpoint_url=self.get_endpoint(),
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"✓ Created bucket: {bucket_name}")
        except s3_client.exceptions.BucketAlreadyOwnedByYou:
            print(f"  Bucket already exists: {bucket_name}")


def setup_minio_for_benchmarks(bucket_name: str = "hyperstreamdb-benchmarks") -> MinIOManager:
    """
    Setup MinIO for benchmark tests.
    
    Returns:
        MinIOManager instance
    """
    print("Setting up MinIO for benchmarks...")
    
    minio = MinIOManager()
    minio.start(use_docker=True)
    minio.create_bucket(bucket_name)
    
    # Set environment variables for HyperStreamDB
    os.environ["AWS_ENDPOINT_URL"] = minio.get_endpoint()
    os.environ["AWS_ACCESS_KEY_ID"] = minio.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio.secret_key
    os.environ["AWS_REGION"] = "us-east-1"
    
    return minio
