# Ryzanstein LLM API Documentation

## Overview

The Ryzanstein LLM API provides OpenAI-compatible endpoints for chat completions, embeddings, and model management. This allows seamless integration with existing tools and workflows.

## Base URL

```
http://localhost:8000
```

## Authentication

(TODO: Authentication will be implemented in future versions)

## Endpoints

### 1. List Models

Get a list of available models.

**Endpoint:** `GET /v1/models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "bitnet-7b",
      "object": "model",
      "created": 1704067200,
      "owned_by": "ryzanstein-llm"
    },
    {
      "id": "mamba-2.8b",
      "object": "model",
      "created": 1704067200,
      "owned_by": "ryzanstein-llm"
    }
  ]
}
```

### 2. Chat Completions

Generate chat completions with streaming support.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**
```json
{
  "model": "bitnet-7b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Write a Python function to calculate fibonacci numbers."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

**Parameters:**
- `model` (string, required): Model identifier
- `messages` (array, required): Array of message objects with `role` and `content`
- `temperature` (number, optional): Sampling temperature (0.0-2.0, default: 0.7)
- `max_tokens` (integer, optional): Maximum tokens to generate
- `stream` (boolean, optional): Enable streaming (default: false)
- `top_p` (number, optional): Nucleus sampling parameter (0.0-1.0, default: 1.0)

**Response (Non-Streaming):**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "bitnet-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here's a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

**Response (Streaming):**

When `stream: true`, responses are sent as Server-Sent Events (SSE):

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"choices":[{"index":0,"delta":{"content":"Here"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704067200,"choices":[{"index":0,"delta":{"content":"'s"},"finish_reason":null}]}

...

data: [DONE]
```

### 3. Embeddings

Generate embeddings for text.

**Endpoint:** `POST /v1/embeddings`

**Request Body:**
```json
{
  "model": "default",
  "input": "The food was delicious and the waiter was friendly."
}
```

**Parameters:**
- `model` (string, optional): Embedding model (default: "default")
- `input` (string or array, required): Text or array of texts to embed

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, -0.3, ...],
      "index": 0
    }
  ],
  "model": "default",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

## Model Context Protocol (MCP)

Ryzanstein LLM supports the Model Context Protocol for tool use and agent capabilities.

### Tool Registration

Tools can be registered via the MCP bridge for dynamic invocation during generation.

(TODO: Complete MCP documentation)

## Error Responses

All errors follow OpenAI's error format:

```json
{
  "error": {
    "message": "Invalid model specified",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

**Common Error Codes:**
- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Authentication failed
- `404`: Not Found - Model or resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error
- `501`: Not Implemented - Feature not yet available

## Rate Limiting

(TODO: Rate limiting will be implemented in future versions)

## Streaming Best Practices

When using streaming mode:

1. Set appropriate connection timeouts
2. Handle reconnection logic for network interruptions
3. Parse SSE messages incrementally
4. Check for the `[DONE]` message to detect completion

## Client Examples

### Python (OpenAI SDK)

```python
import openai

openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(
    model="bitnet-7b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bitnet-7b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'bitnet-7b',
    messages: [{ role: 'user', content: 'Hello!' }]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## Performance Tips

1. **Use Streaming**: For interactive applications, streaming reduces perceived latency
2. **Batch Requests**: Use batch embedding requests when processing multiple texts
3. **Model Selection**: Choose smaller models (Mamba 2.8B) for simple tasks
4. **Context Management**: Leverage the token recycling system for long conversations

## Compatibility

The API is designed to be compatible with:
- OpenAI Python SDK
- OpenAI Node.js SDK
- LangChain
- LlamaIndex
- Cursor IDE
- VS Code extensions using OpenAI API

## Support

For issues and questions, please refer to the project documentation or open an issue on GitHub.
