from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter('orchestrator_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('orchestrator_request_latency_seconds', 'Request latency seconds')
