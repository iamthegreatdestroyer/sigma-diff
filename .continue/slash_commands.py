"""
Continue.dev Slash Commands Implementation
Sprint 5: Complete slash command handler system

Implements 40 slash commands for the Elite Agent Collective,
providing specialized assistance across all software engineering domains.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Command Types & Data Classes
# =============================================================================

class CommandCategory(Enum):
    """Slash command categories."""
    INFERENCE = "inference"
    CODE_ANALYSIS = "code_analysis"
    OPTIMIZATION = "optimization"
    TESTING = "testing"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    API_DESIGN = "api_design"
    DATABASE = "database"
    DEVOPS = "devops"
    MACHINE_LEARNING = "machine_learning"
    PERFORMANCE = "performance"
    CONCURRENCY = "concurrency"
    RESEARCH = "research"
    INNOVATION = "innovation"
    DEBUGGING = "debugging"
    ACCESSIBILITY = "accessibility"
    META = "meta"


@dataclass
class SlashCommand:
    """Slash command definition."""
    name: str
    description: str
    category: CommandCategory
    elite_agent: str  # Which Elite Agent handles this
    prompt_template: str
    context_required: List[str]
    streaming_enabled: bool = True
    timeout: int = 30000  # milliseconds


@dataclass
class CommandContext:
    """Context for command execution."""
    selected_code: Optional[str] = None
    current_file: Optional[str] = None
    file_language: Optional[str] = None
    git_context: Optional[Dict[str, str]] = None
    conversation_history: List[Dict[str, str]] = None
    workspace_root: Optional[str] = None


@dataclass
class CommandResponse:
    """Response from slash command execution."""
    command_name: str
    content: str
    is_streaming: bool
    metadata: Dict[str, Any]
    agent_used: str


# =============================================================================
# Slash Command Registry
# =============================================================================

class SlashCommandRegistry:
    """Registry and handler for all slash commands."""

    def __init__(self):
        self.commands: Dict[str, SlashCommand] = {}
        self._register_all_commands()

    def _register_all_commands(self):
        """Register all 40+ slash commands."""
        
        # Core Commands (Elite Agent: @APEX)
        self._register_command(SlashCommand(
            name="inference",
            description="Run direct model inference on selected code",
            category=CommandCategory.INFERENCE,
            elite_agent="@APEX",
            prompt_template="Analyze the selected code and provide detailed insights",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="chat",
            description="Start a multi-turn conversation",
            category=CommandCategory.INFERENCE,
            elite_agent="@APEX",
            prompt_template="Engage in a detailed discussion about the selected code",
            context_required=[],
        ))

        # Code Analysis (Elite Agent: @MENTOR)
        self._register_command(SlashCommand(
            name="explain",
            description="Explain the selected code in detail",
            category=CommandCategory.CODE_ANALYSIS,
            elite_agent="@MENTOR",
            prompt_template="""Explain the selected code thoroughly:
1. What does it do?
2. Why is this approach used?
3. What are the key algorithms or patterns?
4. How does it integrate with the rest of the system?""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="review",
            description="Code review with constructive feedback",
            category=CommandCategory.CODE_ANALYSIS,
            elite_agent="@MENTOR",
            prompt_template="""Provide a professional code review:
1. Code quality assessment
2. Best practices alignment
3. Potential issues or improvements
4. Performance considerations
5. Maintainability suggestions""",
            context_required=["selected_code"],
        ))

        # Performance & Optimization (Elite Agent: @VELOCITY)
        self._register_command(SlashCommand(
            name="optimize",
            description="Optimize the selected code for performance",
            category=CommandCategory.OPTIMIZATION,
            elite_agent="@VELOCITY",
            prompt_template="""Optimize the selected code:
1. Identify performance bottlenecks
2. Suggest algorithmic improvements
3. Recommend data structure changes
4. Provide memory optimization tips
5. Show before/after comparisons""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="bench",
            description="Add benchmarking code",
            category=CommandCategory.OPTIMIZATION,
            elite_agent="@VELOCITY",
            prompt_template="""Add comprehensive benchmarking:
1. Create timing instrumentation
2. Add profiling hooks
3. Generate performance reports
4. Show optimization opportunities""",
            context_required=["selected_code", "file_language"],
        ))

        # Testing & Quality (Elite Agent: @ECLIPSE)
        self._register_command(SlashCommand(
            name="test",
            description="Generate unit tests",
            category=CommandCategory.TESTING,
            elite_agent="@ECLIPSE",
            prompt_template="""Generate comprehensive unit tests:
1. Test all functions/methods
2. Cover edge cases
3. Add integration tests
4. Include mocking/fixtures
5. Ensure >90% coverage""",
            context_required=["selected_code", "file_language"],
        ))

        self._register_command(SlashCommand(
            name="doctest",
            description="Add docstring tests",
            category=CommandCategory.TESTING,
            elite_agent="@ECLIPSE",
            prompt_template="""Add thorough docstring tests:
1. Document function behavior
2. Include usage examples
3. Add parameter descriptions
4. Document return values
5. Include exception cases""",
            context_required=["selected_code"],
        ))

        # Security (Elite Agent: @CIPHER)
        self._register_command(SlashCommand(
            name="security",
            description="Security audit and hardening",
            category=CommandCategory.SECURITY,
            elite_agent="@CIPHER",
            prompt_template="""Perform security analysis:
1. Identify vulnerabilities
2. Check for injection risks
3. Validate authentication/authorization
4. Review cryptographic usage
5. Suggest hardening measures""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="sanitize",
            description="Sanitize user input handling",
            category=CommandCategory.SECURITY,
            elite_agent="@CIPHER",
            prompt_template="""Improve input validation:
1. Add input sanitization
2. Implement validation rules
3. Add error handling
4. Prevent injection attacks
5. Log security events""",
            context_required=["selected_code"],
        ))

        # Architecture & Design (Elite Agent: @ARCHITECT)
        self._register_command(SlashCommand(
            name="arch",
            description="Architecture review and suggestions",
            category=CommandCategory.ARCHITECTURE,
            elite_agent="@ARCHITECT",
            prompt_template="""Review and improve architecture:
1. Component decomposition
2. Interface design
3. Dependency analysis
4. Scalability assessment
5. Alternative architectures""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="refactor",
            description="Refactor for maintainability",
            category=CommandCategory.ARCHITECTURE,
            elite_agent="@ARCHITECT",
            prompt_template="""Refactor for better maintainability:
1. Apply design patterns
2. Improve naming conventions
3. Reduce complexity
4. Extract methods/components
5. Improve modularity""",
            context_required=["selected_code"],
        ))

        # Documentation (Elite Agent: @SCRIBE)
        self._register_command(SlashCommand(
            name="doc",
            description="Generate comprehensive documentation",
            category=CommandCategory.DOCUMENTATION,
            elite_agent="@SCRIBE",
            prompt_template="""Generate documentation:
1. Module overview
2. API documentation
3. Usage examples
4. Architecture diagrams
5. Integration guides""",
            context_required=["selected_code", "current_file"],
        ))

        self._register_command(SlashCommand(
            name="comment",
            description="Add meaningful comments",
            category=CommandCategory.DOCUMENTATION,
            elite_agent="@SCRIBE",
            prompt_template="""Add helpful comments:
1. Explain complex logic
2. Document assumptions
3. Note performance implications
4. Flag potential issues
5. Suggest improvements""",
            context_required=["selected_code"],
        ))

        # API & Integration (Elite Agent: @SYNAPSE)
        self._register_command(SlashCommand(
            name="api",
            description="Design REST/GraphQL API",
            category=CommandCategory.API_DESIGN,
            elite_agent="@SYNAPSE",
            prompt_template="""Design API endpoints:
1. Resource modeling
2. HTTP methods and status codes
3. Request/response schemas
4. Error handling
5. Versioning strategy""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="integrate",
            description="Integration implementation",
            category=CommandCategory.API_DESIGN,
            elite_agent="@SYNAPSE",
            prompt_template="""Implement integration:
1. API client generation
2. Error handling
3. Retry logic
4. Caching strategy
5. Testing approach""",
            context_required=["selected_code"],
        ))

        # Data & Database (Elite Agent: @VERTEX)
        self._register_command(SlashCommand(
            name="query",
            description="Database query optimization",
            category=CommandCategory.DATABASE,
            elite_agent="@VERTEX",
            prompt_template="""Optimize database operations:
1. Query analysis
2. Index suggestions
3. Normalization review
4. Performance tuning
5. Caching strategy""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="migrate",
            description="Database migration design",
            category=CommandCategory.DATABASE,
            elite_agent="@VERTEX",
            prompt_template="""Design database migration:
1. Schema changes
2. Data transformation
3. Rollback strategy
4. Performance impact
5. Compatibility checks""",
            context_required=["selected_code"],
        ))

        # DevOps & Deployment (Elite Agent: @FLUX)
        self._register_command(SlashCommand(
            name="deploy",
            description="Deployment configuration",
            category=CommandCategory.DEVOPS,
            elite_agent="@FLUX",
            prompt_template="""Create deployment setup:
1. Container configuration
2. Infrastructure as Code
3. CI/CD pipeline
4. Monitoring setup
5. Rollback strategy""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="ci",
            description="CI/CD pipeline design",
            category=CommandCategory.DEVOPS,
            elite_agent="@FLUX",
            prompt_template="""Design CI/CD pipeline:
1. Build steps
2. Test stages
3. Deployment gates
4. Monitoring integration
5. Failure handling""",
            context_required=["selected_code"],
        ))

        # ML & AI (Elite Agent: @TENSOR)
        self._register_command(SlashCommand(
            name="ml",
            description="Machine learning implementation",
            category=CommandCategory.MACHINE_LEARNING,
            elite_agent="@TENSOR",
            prompt_template="""Design ML solution:
1. Model selection
2. Data pipeline
3. Training strategy
4. Evaluation metrics
5. Deployment approach""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="train",
            description="Training optimization",
            category=CommandCategory.MACHINE_LEARNING,
            elite_agent="@TENSOR",
            prompt_template="""Optimize model training:
1. Hyperparameter tuning
2. Data augmentation
3. Loss functions
4. Optimization strategy
5. Validation approach""",
            context_required=["selected_code"],
        ))

        # Encryption & Auth (Elite Agent: @CIPHER)
        self._register_command(SlashCommand(
            name="encrypt",
            description="Encryption implementation",
            category=CommandCategory.SECURITY,
            elite_agent="@CIPHER",
            prompt_template="""Implement encryption:
1. Algorithm selection
2. Key management
3. Implementation details
4. Error handling
5. Performance considerations""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="auth",
            description="Authentication & authorization",
            category=CommandCategory.SECURITY,
            elite_agent="@CIPHER",
            prompt_template="""Design auth system:
1. Authentication method
2. Session management
3. Authorization rules
4. Token handling
5. Security best practices""",
            context_required=["selected_code"],
        ))

        # Cloud & Infrastructure (Elite Agent: @ATLAS)
        self._register_command(SlashCommand(
            name="cloud",
            description="Cloud infrastructure design",
            category=CommandCategory.DEVOPS,
            elite_agent="@ATLAS",
            prompt_template="""Design cloud architecture:
1. Service selection
2. Scalability planning
3. Cost optimization
4. Security setup
5. Disaster recovery""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="infra",
            description="Infrastructure automation",
            category=CommandCategory.DEVOPS,
            elite_agent="@ATLAS",
            prompt_template="""Automate infrastructure:
1. IaC templating
2. Environment management
3. Scaling policies
4. Monitoring setup
5. Disaster recovery""",
            context_required=["selected_code"],
        ))

        # Performance Advanced
        self._register_command(SlashCommand(
            name="profile",
            description="Performance profiling",
            category=CommandCategory.PERFORMANCE,
            elite_agent="@VELOCITY",
            prompt_template="""Add performance profiling:
1. CPU profiling
2. Memory tracking
3. I/O monitoring
4. Bottleneck identification
5. Optimization recommendations""",
            context_required=["selected_code", "file_language"],
        ))

        self._register_command(SlashCommand(
            name="cache",
            description="Caching strategy",
            category=CommandCategory.PERFORMANCE,
            elite_agent="@VELOCITY",
            prompt_template="""Design caching layer:
1. Cache key strategy
2. TTL configuration
3. Eviction policy
4. Hit rate optimization
5. Invalidation strategy""",
            context_required=["selected_code"],
        ))

        # Concurrency (Elite Agent: @APEX)
        self._register_command(SlashCommand(
            name="async",
            description="Async/await patterns",
            category=CommandCategory.CONCURRENCY,
            elite_agent="@APEX",
            prompt_template="""Implement async patterns:
1. Promise/Future usage
2. Error handling
3. Timeout management
4. Cancellation tokens
5. Performance tuning""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="thread",
            description="Threading & synchronization",
            category=CommandCategory.CONCURRENCY,
            elite_agent="@APEX",
            prompt_template="""Design threading approach:
1. Thread pool sizing
2. Synchronization primitives
3. Deadlock prevention
4. Race condition handling
5. Performance optimization""",
            context_required=["selected_code"],
        ))

        # Research & Analysis (Elite Agent: @VANGUARD)
        self._register_command(SlashCommand(
            name="research",
            description="Literature and research assistance",
            category=CommandCategory.RESEARCH,
            elite_agent="@VANGUARD",
            prompt_template="""Research and analyze:
1. Topic overview
2. State-of-the-art approaches
3. Implementation options
4. Trade-off analysis
5. Recommendations""",
            context_required=[],
        ))

        self._register_command(SlashCommand(
            name="compare",
            description="Compare approaches and frameworks",
            category=CommandCategory.RESEARCH,
            elite_agent="@VANGUARD",
            prompt_template="""Comparative analysis:
1. Feature comparison
2. Performance benchmarks
3. Ease of use
4. Community support
5. Cost analysis""",
            context_required=[],
        ))

        # Innovation & Design (Elite Agent: @GENESIS)
        self._register_command(SlashCommand(
            name="design",
            description="Design new features or systems",
            category=CommandCategory.INNOVATION,
            elite_agent="@GENESIS",
            prompt_template="""Design new solution:
1. Requirements analysis
2. Design options
3. Pros and cons
4. Implementation plan
5. Risk assessment""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="novel",
            description="Novel approaches and breakthroughs",
            category=CommandCategory.INNOVATION,
            elite_agent="@GENESIS",
            prompt_template="""Explore novel approaches:
1. Challenge assumptions
2. Brainstorm alternatives
3. Evaluate innovations
4. Implementation strategy
5. Performance predictions""",
            context_required=["selected_code"],
        ))

        # Debugging & Troubleshooting
        self._register_command(SlashCommand(
            name="debug",
            description="Debugging assistance",
            category=CommandCategory.DEBUGGING,
            elite_agent="@APEX",
            prompt_template="""Help with debugging:
1. Analyze error messages
2. Identify root cause
3. Suggest fixes
4. Add debugging instrumentation
5. Prevent future issues""",
            context_required=["selected_code"],
        ))

        self._register_command(SlashCommand(
            name="trace",
            description="Add tracing and logging",
            category=CommandCategory.DEBUGGING,
            elite_agent="@APEX",
            prompt_template="""Add comprehensive tracing:
1. Log critical points
2. Include context information
3. Add performance metrics
4. Structure log output
5. Set appropriate levels""",
            context_required=["selected_code"],
        ))

        # Accessibility & UX (Elite Agent: @CANVAS)
        self._register_command(SlashCommand(
            name="a11y",
            description="Accessibility improvements",
            category=CommandCategory.ACCESSIBILITY,
            elite_agent="@CANVAS",
            prompt_template="""Improve accessibility:
1. WCAG compliance
2. Keyboard navigation
3. Screen reader support
4. Color contrast
5. ARIA attributes""",
            context_required=["selected_code", "file_language"],
        ))

        self._register_command(SlashCommand(
            name="ux",
            description="User experience enhancement",
            category=CommandCategory.ACCESSIBILITY,
            elite_agent="@CANVAS",
            prompt_template="""Enhance user experience:
1. Usability assessment
2. Navigation improvement
3. Error messaging
4. Performance optimization
5. Accessibility checks""",
            context_required=["selected_code"],
        ))

        # Help & Meta
        self._register_command(SlashCommand(
            name="help",
            description="Get help with slash commands",
            category=CommandCategory.META,
            elite_agent="@OMNISCIENT",
            prompt_template="""Explain available commands:
1. List all commands
2. Explain each command
3. Show usage examples
4. Provide tips and tricks
5. Link to documentation""",
            context_required=[],
        ))

        self._register_command(SlashCommand(
            name="context",
            description="Manage conversation context",
            category=CommandCategory.META,
            elite_agent="@OMNISCIENT",
            prompt_template="""Manage context:
1. Summarize current context
2. Remove irrelevant context
3. Expand context window
4. Focus on specific files
5. Clear conversation history""",
            context_required=[],
        ))

    def _register_command(self, command: SlashCommand):
        """Register a single command."""
        self.commands[command.name] = command
        logger.info(f"Registered slash command: /{command.name} ({command.elite_agent})")

    def get_command(self, name: str) -> Optional[SlashCommand]:
        """Get a command by name."""
        return self.commands.get(name)

    def list_commands(self, category: Optional[CommandCategory] = None) -> List[SlashCommand]:
        """List all commands, optionally filtered by category."""
        commands = list(self.commands.values())
        if category:
            commands = [c for c in commands if c.category == category]
        return sorted(commands, key=lambda c: c.name)

    def list_by_agent(self, agent: str) -> List[SlashCommand]:
        """List all commands handled by a specific agent."""
        return sorted(
            [c for c in self.commands.values() if c.elite_agent == agent],
            key=lambda c: c.name
        )


# =============================================================================
# Command Handler
# =============================================================================

class SlashCommandHandler:
    """Handles execution of slash commands."""

    def __init__(self, ryzanstein_api_url: str = "http://localhost:8000"):
        self.registry = SlashCommandRegistry()
        self.ryzanstein_api_url = ryzanstein_api_url
        self.command_handlers: Dict[str, Callable] = {}

    async def execute_command(
        self,
        command_name: str,
        context: CommandContext,
        user_input: Optional[str] = None
    ) -> CommandResponse:
        """Execute a slash command."""
        command = self.registry.get_command(command_name)
        if not command:
            raise ValueError(f"Unknown command: /{command_name}")

        # Validate context
        for required in command.context_required:
            if not getattr(context, required, None):
                raise ValueError(f"Command /{command_name} requires: {required}")

        # Get handler
        handler = self.command_handlers.get(command_name)
        if not handler:
            handler = self._get_default_handler(command)

        # Execute
        logger.info(f"Executing command: /{command_name} (handler: {command.elite_agent})")
        
        response = await handler(command, context, user_input)
        return response

    def _get_default_handler(self, command: SlashCommand) -> Callable:
        """Get default handler for a command."""
        async def default_handler(
            cmd: SlashCommand,
            ctx: CommandContext,
            user_input: Optional[str]
        ) -> CommandResponse:
            # Build prompt
            prompt = cmd.prompt_template
            if user_input:
                prompt = f"{prompt}\n\nUser input: {user_input}"
            
            if ctx.selected_code:
                prompt = f"{prompt}\n\nCode:\n{ctx.selected_code}"

            # Call Ryzanstein API
            response_content = await self._call_ryzanstein_api(
                prompt,
                streaming=cmd.streaming_enabled,
                timeout=cmd.timeout
            )

            return CommandResponse(
                command_name=cmd.name,
                content=response_content,
                is_streaming=cmd.streaming_enabled,
                metadata={
                    "agent": cmd.elite_agent,
                    "category": cmd.category.value,
                },
                agent_used=cmd.elite_agent
            )

        return default_handler

    async def _call_ryzanstein_api(
        self,
        prompt: str,
        streaming: bool = True,
        timeout: int = 30000
    ) -> str:
        """Call Ryzanstein LLM API."""
        # This would be implemented with actual API calls
        # For now, return a placeholder
        logger.info(f"Calling Ryzanstein API: streaming={streaming}, timeout={timeout}ms")
        return f"[Response from {self.ryzanstein_api_url}]"


# =============================================================================
# Convenience Functions
# =============================================================================

def get_command_registry() -> SlashCommandRegistry:
    """Get the slash command registry."""
    return SlashCommandRegistry()


def list_all_commands() -> List[SlashCommand]:
    """List all available slash commands."""
    registry = get_command_registry()
    return registry.list_commands()


def list_commands_by_category(category: CommandCategory) -> List[SlashCommand]:
    """List commands by category."""
    registry = get_command_registry()
    return registry.list_by_category(category)


def list_commands_by_agent(agent: str) -> List[SlashCommand]:
    """List commands handled by a specific Elite Agent."""
    registry = get_command_registry()
    return registry.list_by_agent(agent)


if __name__ == "__main__":
    # Display all commands
    registry = get_command_registry()
    
    print(f"Total commands: {len(registry.commands)}")
    print("\nCommands by category:")
    
    for category in CommandCategory:
        cmds = registry.list_commands(category)
        if cmds:
            print(f"\n{category.value.upper()} ({len(cmds)}):")
            for cmd in cmds:
                print(f"  /{cmd.name:15} - {cmd.description:40} ({cmd.elite_agent})")
