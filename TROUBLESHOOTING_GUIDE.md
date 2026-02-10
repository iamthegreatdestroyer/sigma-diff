# Troubleshooting Guide - Desktop Client System

**Date:** January 7, 2026  
**Sprint:** Sprint 6, Week 2, Days 4-5  
**Component:** Desktop Client Support

## Common Issues and Solutions

### Issue 1: Configuration File Not Found

**Symptom:**

```
Error: configuration file not found at config.yaml
```

**Root Cause:**

- File path is incorrect
- Working directory is wrong
- File doesn't exist

**Solutions:**

1. **Verify file exists:**

   ```bash
   ls -la config.yaml
   ```

2. **Use absolute path:**

   ```bash
   export RYOT_CONFIG_PATH=/absolute/path/to/config.yaml
   ```

3. **Create default config:**
   ```go
   cfg := config.NewDefaultConfig()
   // Use defaults instead of file
   ```

---

### Issue 2: Connection Refused

**Symptom:**

```
Error: connection refused (127.0.0.1:8080)
```

**Root Cause:**

- Service not running
- Wrong host/port in config
- Firewall blocking connection

**Solutions:**

1. **Verify service is running:**

   ```bash
   curl http://localhost:8080/api/health
   ```

2. **Check configuration:**

   ```yaml
   # Verify in config.yaml
   server:
     host: "127.0.0.1"
     port: 8080
   ```

3. **Check firewall:**
   ```bash
   # Linux: check if port is listening
   netstat -tlnp | grep 8080
   ```

---

### Issue 3: Model Loading Timeout

**Symptom:**

```
Error: context deadline exceeded while loading model
```

**Root Cause:**

- Model is large and takes time to load
- Timeout too short
- Insufficient memory

**Solutions:**

1. **Increase timeout:**

   ```go
   ctx, cancel := context.WithTimeout(
       context.Background(),
       5*time.Minute, // Increase from 30s
   )
   defer cancel()
   ```

2. **Check available memory:**

   ```bash
   free -h  # Linux
   ```

3. **Load model asynchronously:**
   ```go
   go func() {
       ms.LoadModel(ctx, "model-id")
   }()
   ```

---

### Issue 4: Out of Memory

**Symptom:**

```
Error: cannot allocate memory
Fatal error: runtime: out of memory
```

**Root Cause:**

- Too many models loaded
- Cache size too large
- Memory leak in application

**Solutions:**

1. **Reduce loaded models:**

   ```go
   ms.UnloadModel(ctx, "unused-model-id")
   ```

2. **Reduce cache size:**

   ```yaml
   cache:
     size: 512 # Reduce from 1024
   ```

3. **Clear cache periodically:**

   ```go
   ticker := time.NewTicker(1*time.Hour)
   for range ticker.C {
       ms.ClearCache()
   }
   ```

4. **Monitor memory:**
   ```bash
   # Linux: watch memory usage
   watch -n 1 free -h
   ```

---

### Issue 5: High Latency

**Symptom:**

```
Inference takes 30+ seconds
Expected: <1 second
```

**Root Cause:**

- Model not loaded
- Network latency
- Slow hardware
- Large batch size

**Solutions:**

1. **Pre-load models:**

   ```go
   ms.LoadModel(ctx, "model-id")
   // Now inference is faster
   ```

2. **Use connection pooling:**

   ```yaml
   client:
     connection_pool_size: 20
   ```

3. **Check network:**

   ```bash
   ping api.server.com
   ```

4. **Monitor latency:**
   ```go
   start := time.Now()
   result := is.Infer(ctx, "model-id", "prompt")
   duration := time.Since(start)
   fmt.Printf("Latency: %v\n", duration)
   ```

---

### Issue 6: Cache Not Working

**Symptom:**

```
Every request hits the API (no caching)
```

**Root Cause:**

- Cache disabled
- TTL expired
- Cache cleared

**Solutions:**

1. **Enable cache:**

   ```yaml
   cache:
     enabled: true
   ```

2. **Check TTL:**

   ```yaml
   cache:
     ttl: 3600s # 1 hour
   ```

3. **Verify cache hits:**
   ```go
   metrics := ms.GetMetrics()
   fmt.Printf("Cache hits: %d\n", metrics.CacheHits)
   ```

---

### Issue 7: Context Cancellation Errors

**Symptom:**

```
Error: context cancelled
Error: context deadline exceeded
```

**Root Cause:**

- Timeout too short
- Context cancelled externally
- Operation took longer than expected

**Solutions:**

1. **Increase timeout:**

   ```go
   ctx, cancel := context.WithTimeout(
       context.Background(),
       60*time.Second, // Longer timeout
   )
   defer cancel()
   ```

2. **Handle cancellation gracefully:**

   ```go
   select {
   case <-ctx.Done():
       return ctx.Err()
   case result := <-resultChan:
       return result
   }
   ```

3. **Don't cancel prematurely:**

   ```go
   // Wrong - cancels too early
   ctx, cancel := context.WithCancel(context.Background())
   cancel() // Oops!

   // Correct - use timeout instead
   ctx, cancel := context.WithTimeout(
       context.Background(),
       30*time.Second,
   )
   defer cancel()
   ```

---

### Issue 8: Concurrent Request Errors

**Symptom:**

```
Errors with 10+ concurrent requests
Works fine with 1-2 requests
```

**Root Cause:**

- Race condition
- Not enough connections
- Connection pool exhausted

**Solutions:**

1. **Increase connection pool:**

   ```yaml
   server:
     max_connections: 100
   ```

2. **Use proper synchronization:**

   ```go
   var mu sync.RWMutex
   // Protect shared data
   mu.Lock()
   defer mu.Unlock()
   ```

3. **Test concurrency:**
   ```bash
   go test -race ./...
   ```

---

### Issue 9: Invalid Configuration

**Symptom:**

```
Error: invalid configuration
Error: validation failed
```

**Root Cause:**

- Wrong data types
- Missing required fields
- Invalid values

**Solutions:**

1. **Check YAML syntax:**

   ```yaml
   # Wrong - string instead of number
   server:
     port: "8080"

   # Correct
   server:
     port: 8080
   ```

2. **Validate configuration:**

   ```go
   cfg, err := config.LoadConfig("config.yaml")
   if err != nil {
       fmt.Printf("Config error: %v\n", err)
   }
   ```

3. **Use environment variables:**
   ```bash
   export RYOT_SERVER_PORT=8080
   ```

---

### Issue 10: Performance Degradation

**Symptom:**

```
System starts fast, then slows down
Latency increases over time
```

**Root Cause:**

- Memory leak
- Cache growing unbounded
- Garbage collection pressure

**Solutions:**

1. **Monitor memory growth:**

   ```bash
   # Linux: monitor over time
   watch -n 5 'ps aux | grep ryot'
   ```

2. **Clear cache periodically:**

   ```go
   ticker := time.NewTicker(10*time.Minute)
   for range ticker.C {
       ms.ClearCache()
   }
   ```

3. **Limit model loading:**
   ```go
   if ms.GetModelCount() > 5 {
       return fmt.Errorf("too many models loaded")
   }
   ```

---

## Debug Tips

### Enable Debug Logging

```yaml
logging:
  level: "debug"
  format: "text"
  output: "stdout"
```

### Check Metrics

```go
// Model service metrics
fmt.Printf("Loaded models: %d\n", ms.GetModelCount())
fmt.Printf("Cache hits: %d\n", ms.GetMetrics().CacheHits)

// Inference service metrics
fmt.Printf("Total inferences: %d\n", is.GetMetrics().TotalRequests)
fmt.Printf("Errors: %d\n", is.GetMetrics().Errors)
```

### Trace Requests

```bash
# Use network trace
tcpdump -i lo -A 'tcp port 8080'
```

---

## Performance Tuning

### For High Throughput

```yaml
server:
  max_connections: 200
  timeout: 60s

cache:
  size: 2048
  ttl: 3600s

client:
  connection_pool_size: 50
```

### For Low Latency

```yaml
server:
  timeout: 10s

cache:
  enabled: true
  ttl: 300s

logging:
  level: "warn"
```

### For Resource Constrained

```yaml
server:
  max_connections: 10
  timeout: 30s

cache:
  size: 256
  ttl: 300s

logging:
  level: "error"
```

---

## Getting Help

### Check Logs

```bash
# View recent logs
tail -100f app.log

# Search for errors
grep ERROR app.log
```

### Run Diagnostics

```bash
# Test configuration
go run cmd/config-test/main.go

# Run benchmarks
go test -bench=. ./...
```

### Report Issues

When reporting issues, include:

- Error message (full text)
- Configuration (sanitized)
- Logs (relevant section)
- Steps to reproduce
- System info (OS, Go version, etc.)

---

**For more help, see:**

- CONFIGURATION_GUIDE.md
- INTEGRATION_GUIDE.md
- Code documentation (godoc)
