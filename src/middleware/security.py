from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers
import time
from typing import Callable, List
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security Headers
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "img-src 'self' data:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline';"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }
        
        for key, value in headers.items():
            response.headers[key] = value
            
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app: FastAPI, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        current_time = time.time()
        if client_ip in self.requests:
            requests = [req for req in self.requests[client_ip] 
                       if req > current_time - 60]
            
            if len(requests) >= self.requests_per_minute:
                return Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": "60"}
                )
            
            self.requests[client_ip] = requests + [current_time]
        else:
            self.requests[client_ip] = [current_time]
        
        return await call_next(request)

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate incoming requests"""
    
    def __init__(self, app: FastAPI, max_content_length: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_content_length = max_content_length
        self.blocked_patterns = [
            r"(?i)(<|%3C)script",  # XSS patterns
            r"(?i)alert\s*\(",
            r"(?i)eval\s*\(",
            r"(?i)javascript:",
            r"(?i)onload=",
            r"(?i)union\s+select",  # SQL injection patterns
            r"(?i)drop\s+table",
            r"(?i)--",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length", 0)
        if int(content_length) > self.max_content_length:
            return Response(
                content="Request too large",
                status_code=413
            )
        
        # Check for malicious patterns
        body = await request.body()
        body_str = body.decode()
        for pattern in self.blocked_patterns:
            if re.search(pattern, body_str):
                logger.warning(f"Blocked malicious request pattern: {pattern}")
                return Response(
                    content="Invalid request",
                    status_code=400
                )
        
        return await call_next(request)

class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log all requests for audit purposes"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} "
            f"Duration: {duration:.2f}s"
        )
        
        return response

def setup_security_middleware(
    app: FastAPI,
    allowed_hosts: List[str] = None,
    cors_origins: List[str] = None,
    requests_per_minute: int = 60,
    max_content_length: int = 10 * 1024 * 1024
) -> None:
    """Setup all security middleware"""
    
    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=requests_per_minute
    )
    
    # Add request validation
    app.add_middleware(
        RequestValidationMiddleware,
        max_content_length=max_content_length
    )
    
    # Add audit logging
    app.add_middleware(AuditLogMiddleware)
    
    # Add CORS middleware
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add trusted host middleware
    if allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )
    
    # Add compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
