# Configuration Guide - Desktop Client System

**Date:** January 7, 2026  
**Sprint:** Sprint 6, Week 2, Days 4-5  
**Component:** Configuration Management System

## Overview

This guide provides detailed instructions for configuring the desktop client system. The system uses a hierarchical configuration approach with environment variables, YAML configuration files, and programmatic configuration loading.

## Configuration Hierarchy

The system follows this priority order (highest to lowest):

1. **Environment Variables** - Runtime overrides
2. **YAML Config File** - Application configuration file
3. **Default Values** - Built-in defaults

## Configuration File Structure

### Basic YAML Format

```yaml
# config.yaml
server:
  host: "127.0.0.1"
  port: 8080
  timeout: 30s
  max_connections: 100

client:
  api_key: "your-api-key"
  model: "sigmalang-v2"
  max_retries: 3
  timeout: 15s

logging:
  level: "info"
  output: "stdout"
  format: "json"

cache:
  enabled: true
  type: "memory"
  ttl: 3600s
  size: 1024
```

### Environment Variables

Override any config value with environment variables:

```bash
# Server configuration
export RYOT_SERVER_HOST="0.0.0.0"
export RYOT_SERVER_PORT=9000
export RYOT_SERVER_TIMEOUT=45s

# Client configuration
export RYOT_CLIENT_API_KEY="your-api-key"
export RYOT_CLIENT_MODEL="sigmalang-v2"

# Logging configuration
export RYOT_LOGGING_LEVEL="debug"
```

## Configuration Loading

### Programmatic Loading

```go
import "github.com/iamthegreatdestroyer/ryot/desktop/internal/config"

// Load with defaults
cfg := config.NewDefaultConfig()

// Load from file and environment
cfg, err := config.LoadConfig("config.yaml")
if err != nil {
    log.Fatal(err)
}

// Access configuration
fmt.Println(cfg.Server.Host)
fmt.Println(cfg.Client.APIKey)
```

### Default Configuration

The system provides sensible defaults:

- **Server:** localhost:8080, 30s timeout, 100 max connections
- **Client:** Default model, 3 retries, 15s timeout
- **Logging:** INFO level, stdout output, JSON format
- **Cache:** Enabled, in-memory, 1 hour TTL, 1024 entries

## Common Configuration Scenarios

### Local Development

```yaml
# config.dev.yaml
server:
  host: "127.0.0.1"
  port: 8080
  timeout: 30s

logging:
  level: "debug"
  format: "text"

cache:
  enabled: true
  type: "memory"
```

### Production Deployment

```yaml
# config.prod.yaml
server:
  host: "0.0.0.0"
  port: 8080
  timeout: 60s
  max_connections: 1000

client:
  max_retries: 5
  timeout: 30s

logging:
  level: "warn"
  output: "file"
  file: "/var/log/ryot/app.log"

cache:
  enabled: true
  type: "redis"
  size: 10000
```

### Testing

```yaml
# config.test.yaml
server:
  host: "127.0.0.1"
  port: 8081

client:
  max_retries: 1

logging:
  level: "error"

cache:
  enabled: false
```

## Configuration Validation

The system validates configuration on load:

```go
// Validation is automatic during Load
cfg, err := config.LoadConfig("config.yaml")
if err != nil {
    // Configuration invalid
    // Possible errors:
    // - Missing required fields
    // - Invalid values
    // - Network configuration issues
}
```

## Troubleshooting Configuration

### Configuration Not Loading

**Problem:** Config file not found  
**Solution:** Verify file path and ensure file exists in correct location

```bash
# Check if config exists
ls -la config.yaml

# Explicitly specify path
RYOT_CONFIG_PATH=/etc/ryot/config.yaml
```

### Environment Variables Not Applied

**Problem:** Environment variables not overriding config  
**Solution:** Ensure variable names follow pattern RYOT_SECTION_KEY

```bash
# Correct
export RYOT_SERVER_HOST=localhost

# Incorrect (won't work)
export SERVER_HOST=localhost
```

### Invalid Configuration Values

**Problem:** Configuration fails validation  
**Solution:** Check value types and ranges

```yaml
# WRONG - strings for numbers
server:
  port: "8080" # Should be: 8080

# WRONG - invalid timeout format
timeout: "30" # Should be: 30s
```

## Performance Tuning

### Cache Configuration

```yaml
cache:
  # Enable for frequently accessed data
  enabled: true

  # Type: "memory" for speed, "redis" for distribution
  type: "memory"

  # TTL: balance freshness vs performance
  ttl: 3600s

  # Size: larger = more memory, fewer cache misses
  size: 2048
```

### Connection Pooling

```yaml
server:
  # Balance throughput vs resource usage
  max_connections: 100

  # Timeout: balance responsiveness vs resource cleanup
  timeout: 30s
```

## Security Configuration

### API Key Management

```yaml
client:
  # Use environment variable for secrets
  # api_key: "your-secret-key"  # DON'T DO THIS
  # Instead use:
  # export RYOT_CLIENT_API_KEY="your-secret-key"
```

### HTTPS Configuration

For production, enable HTTPS:

```yaml
server:
  tls:
    enabled: true
    cert_file: "/etc/ryot/cert.pem"
    key_file: "/etc/ryot/key.pem"
```

## Configuration Monitoring

### Log Configuration Changes

```yaml
logging:
  level: "info"
  format: "json"
  output: "stdout"
```

Monitor logs for configuration-related issues:

```bash
# View configuration-related logs
grep -i "config" app.log
```

---

**Next Steps:** See INTEGRATION_GUIDE.md for usage patterns
