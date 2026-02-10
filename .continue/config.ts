/**
 * Continue.dev Configuration for Ryzanstein LLM
 *
 * This configuration integrates the Ryzanstein LLM inference engine
 * with Continue.dev for in-editor AI assistance.
 *
 * Features:
 * - OpenAI-compatible API endpoint
 * - Streaming response support
 * - Custom model configuration
 * - Advanced context management
 */

import { IdeProtocolServer } from "continue";

const config: IdeProtocolServer = {
  models: [
    {
      title: "Ryzanstein LLM (Local)",
      provider: "openai",
      model: "ryzanstein-7b",
      apiBase: process.env.RYZANSTEIN_API_URL || "http://localhost:8000",
      apiKey: process.env.RYZANSTEIN_API_KEY || "local-development",
      systemPrompt: `You are Ryzanstein, a CPU-optimized LLM assistant integrated into VS Code.
You provide accurate, concise code assistance with deep understanding of software engineering principles.
You leverage the Elite Agent Collective for specialized expertise across domains.`,
      contextLength: 4096,
      maxTokens: 2048,
      temperature: 0.7,
      topP: 0.9,
      streaming: true,
      requestOptions: {
        timeout: 30000,
        retries: 3,
        retryDelay: 1000,
      },
    },
  ],

  slashCommands: [
    // Core Commands (Elite Agent: @APEX)
    {
      name: "inference",
      description: "Run direct model inference on selected code",
      prompt: "Analyze the selected code and provide detailed insights",
    },
    {
      name: "chat",
      description: "Start a multi-turn conversation",
      prompt: "Engage in a detailed discussion about the selected code",
    },

    // Code Analysis (Elite Agent: @MENTOR)
    {
      name: "explain",
      description: "Explain the selected code in detail",
      prompt: `Explain the selected code thoroughly:
1. What does it do?
2. Why is this approach used?
3. What are the key algorithms or patterns?
4. How does it integrate with the rest of the system?`,
    },
    {
      name: "review",
      description: "Code review with constructive feedback",
      prompt: `Provide a professional code review:
1. Code quality assessment
2. Best practices alignment
3. Potential issues or improvements
4. Performance considerations
5. Maintainability suggestions`,
    },

    // Performance & Optimization (Elite Agent: @VELOCITY)
    {
      name: "optimize",
      description: "Optimize the selected code for performance",
      prompt: `Optimize the selected code:
1. Identify performance bottlenecks
2. Suggest algorithmic improvements
3. Recommend data structure changes
4. Provide memory optimization tips
5. Show before/after comparisons`,
    },
    {
      name: "bench",
      description: "Add benchmarking code",
      prompt: `Add comprehensive benchmarking:
1. Create timing instrumentation
2. Add profiling hooks
3. Generate performance reports
4. Show optimization opportunities`,
    },

    // Testing & Quality (Elite Agent: @ECLIPSE)
    {
      name: "test",
      description: "Generate unit tests",
      prompt: `Generate comprehensive unit tests:
1. Test all functions/methods
2. Cover edge cases
3. Add integration tests
4. Include mocking/fixtures
5. Ensure >90% coverage`,
    },
    {
      name: "doctest",
      description: "Add docstring tests",
      prompt: `Add thorough docstring tests:
1. Document function behavior
2. Include usage examples
3. Add parameter descriptions
4. Document return values
5. Include exception cases`,
    },

    // Security (Elite Agent: @CIPHER)
    {
      name: "security",
      description: "Security audit and hardening",
      prompt: `Perform security analysis:
1. Identify vulnerabilities
2. Check for injection risks
3. Validate authentication/authorization
4. Review cryptographic usage
5. Suggest hardening measures`,
    },
    {
      name: "sanitize",
      description: "Sanitize user input handling",
      prompt: `Improve input validation:
1. Add input sanitization
2. Implement validation rules
3. Add error handling
4. Prevent injection attacks
5. Log security events`,
    },

    // Architecture & Design (Elite Agent: @ARCHITECT)
    {
      name: "arch",
      description: "Architecture review and suggestions",
      prompt: `Review and improve architecture:
1. Component decomposition
2. Interface design
3. Dependency analysis
4. Scalability assessment
5. Alternative architectures`,
    },
    {
      name: "refactor",
      description: "Refactor for maintainability",
      prompt: `Refactor for better maintainability:
1. Apply design patterns
2. Improve naming conventions
3. Reduce complexity
4. Extract methods/components
5. Improve modularity`,
    },

    // Documentation (Elite Agent: @SCRIBE)
    {
      name: "doc",
      description: "Generate comprehensive documentation",
      prompt: `Generate documentation:
1. Module overview
2. API documentation
3. Usage examples
4. Architecture diagrams
5. Integration guides`,
    },
    {
      name: "comment",
      description: "Add meaningful comments",
      prompt: `Add helpful comments:
1. Explain complex logic
2. Document assumptions
3. Note performance implications
4. Flag potential issues
5. Suggest improvements`,
    },

    // API & Integration (Elite Agent: @SYNAPSE)
    {
      name: "api",
      description: "Design REST/GraphQL API",
      prompt: `Design API endpoints:
1. Resource modeling
2. HTTP methods and status codes
3. Request/response schemas
4. Error handling
5. Versioning strategy`,
    },
    {
      name: "integrate",
      description: "Integration implementation",
      prompt: `Implement integration:
1. API client generation
2. Error handling
3. Retry logic
4. Caching strategy
5. Testing approach`,
    },

    // Data & Database (Elite Agent: @VERTEX)
    {
      name: "query",
      description: "Database query optimization",
      prompt: `Optimize database operations:
1. Query analysis
2. Index suggestions
3. Normalization review
4. Performance tuning
5. Caching strategy`,
    },
    {
      name: "migrate",
      description: "Database migration design",
      prompt: `Design database migration:
1. Schema changes
2. Data transformation
3. Rollback strategy
4. Performance impact
5. Compatibility checks`,
    },

    // DevOps & Deployment (Elite Agent: @FLUX)
    {
      name: "deploy",
      description: "Deployment configuration",
      prompt: `Create deployment setup:
1. Container configuration
2. Infrastructure as Code
3. CI/CD pipeline
4. Monitoring setup
5. Rollback strategy`,
    },
    {
      name: "ci",
      description: "CI/CD pipeline design",
      prompt: `Design CI/CD pipeline:
1. Build steps
2. Test stages
3. Deployment gates
4. Monitoring integration
5. Failure handling`,
    },

    // ML & AI (Elite Agent: @TENSOR)
    {
      name: "ml",
      description: "Machine learning implementation",
      prompt: `Design ML solution:
1. Model selection
2. Data pipeline
3. Training strategy
4. Evaluation metrics
5. Deployment approach`,
    },
    {
      name: "train",
      description: "Training optimization",
      prompt: `Optimize model training:
1. Hyperparameter tuning
2. Data augmentation
3. Loss functions
4. Optimization strategy
5. Validation approach`,
    },

    // Cryptography & Security (Elite Agent: @CIPHER)
    {
      name: "encrypt",
      description: "Encryption implementation",
      prompt: `Implement encryption:
1. Algorithm selection
2. Key management
3. Implementation details
4. Error handling
5. Performance considerations`,
    },
    {
      name: "auth",
      description: "Authentication & authorization",
      prompt: `Design auth system:
1. Authentication method
2. Session management
3. Authorization rules
4. Token handling
5. Security best practices`,
    },

    // Cloud & Infrastructure (Elite Agent: @ATLAS)
    {
      name: "cloud",
      description: "Cloud infrastructure design",
      prompt: `Design cloud architecture:
1. Service selection
2. Scalability planning
3. Cost optimization
4. Security setup
5. Disaster recovery`,
    },
    {
      name: "infra",
      description: "Infrastructure automation",
      prompt: `Automate infrastructure:
1. IaC templating
2. Environment management
3. Scaling policies
4. Monitoring setup
5. Disaster recovery`,
    },

    // Performance & Optimization Advanced
    {
      name: "profile",
      description: "Performance profiling",
      prompt: `Add performance profiling:
1. CPU profiling
2. Memory tracking
3. I/O monitoring
4. Bottleneck identification
5. Optimization recommendations`,
    },
    {
      name: "cache",
      description: "Caching strategy",
      prompt: `Design caching layer:
1. Cache key strategy
2. TTL configuration
3. Eviction policy
4. Hit rate optimization
5. Invalidation strategy`,
    },

    // Concurrency (Elite Agent: @APEX)
    {
      name: "async",
      description: "Async/await patterns",
      prompt: `Implement async patterns:
1. Promise/Future usage
2. Error handling
3. Timeout management
4. Cancellation tokens
5. Performance tuning`,
    },
    {
      name: "thread",
      description: "Threading & synchronization",
      prompt: `Design threading approach:
1. Thread pool sizing
2. Synchronization primitives
3. Deadlock prevention
4. Race condition handling
5. Performance optimization`,
    },

    // Research & Analysis (Elite Agent: @VANGUARD)
    {
      name: "research",
      description: "Literature and research assistance",
      prompt: `Research and analyze:
1. Topic overview
2. State-of-the-art approaches
3. Implementation options
4. Trade-off analysis
5. Recommendations`,
    },
    {
      name: "compare",
      description: "Compare approaches and frameworks",
      prompt: `Comparative analysis:
1. Feature comparison
2. Performance benchmarks
3. Ease of use
4. Community support
5. Cost analysis`,
    },

    // Innovation & Design (Elite Agent: @GENESIS)
    {
      name: "design",
      description: "Design new features or systems",
      prompt: `Design new solution:
1. Requirements analysis
2. Design options
3. Pros and cons
4. Implementation plan
5. Risk assessment`,
    },
    {
      name: "novel",
      description: "Novel approaches and breakthroughs",
      prompt: `Explore novel approaches:
1. Challenge assumptions
2. Brainstorm alternatives
3. Evaluate innovations
4. Implementation strategy
5. Performance predictions`,
    },

    // Debugging & Troubleshooting
    {
      name: "debug",
      description: "Debugging assistance",
      prompt: `Help with debugging:
1. Analyze error messages
2. Identify root cause
3. Suggest fixes
4. Add debugging instrumentation
5. Prevent future issues`,
    },
    {
      name: "trace",
      description: "Add tracing and logging",
      prompt: `Add comprehensive tracing:
1. Log critical points
2. Include context information
3. Add performance metrics
4. Structure log output
5. Set appropriate levels`,
    },

    // Accessibility & UX (Elite Agent: @CANVAS)
    {
      name: "a11y",
      description: "Accessibility improvements",
      prompt: `Improve accessibility:
1. WCAG compliance
2. Keyboard navigation
3. Screen reader support
4. Color contrast
5. ARIA attributes`,
    },
    {
      name: "ux",
      description: "User experience enhancement",
      prompt: `Enhance user experience:
1. Usability assessment
2. Navigation improvement
3. Error messaging
4. Performance optimization
5. Accessibility checks`,
    },

    // Help & Meta
    {
      name: "help",
      description: "Get help with slash commands",
      prompt: `Explain available commands:
1. List all commands
2. Explain each command
3. Show usage examples
4. Provide tips and tricks
5. Link to documentation`,
    },
    {
      name: "context",
      description: "Manage conversation context",
      prompt: `Manage context:
1. Summarize current context
2. Remove irrelevant context
3. Expand context window
4. Focus on specific files
5. Clear conversation history`,
    },
  ],

  contextProviders: [
    {
      name: "file",
      class: FileContextProvider,
    },
    {
      name: "code",
      class: CodeContextProvider,
    },
    {
      name: "terminal",
      class: TerminalContextProvider,
    },
    {
      name: "git",
      class: GitContextProvider,
    },
  ],

  customCommands: [
    {
      name: "ryzanstein-status",
      prompt: "Show Ryzanstein LLM status and configuration",
      description: "Display current Ryzanstein LLM connection status",
    },
    {
      name: "ryzanstein-benchmark",
      prompt: "Run performance benchmark",
      description: "Benchmark Ryzanstein LLM inference performance",
    },
  ],

  autocompletionProviders: [
    {
      name: "ryzanstein",
      description: "Ryzanstein-specific completions",
    },
  ],
};

export default config;

// Context Providers (Placeholder implementations)
class FileContextProvider {}
class CodeContextProvider {}
class TerminalContextProvider {}
class GitContextProvider {}
