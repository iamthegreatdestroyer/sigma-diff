"""
Continue.dev Integration Tests
Sprint 5: Comprehensive testing suite

Tests for:
- Slash command registration and execution
- Streaming response handling
- Context management
- API integration with Ryzanstein LLM
- Elite Agent routing
"""

import pytest
import asyncio
from typing import Optional, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import logging

# Assuming imports from the slash_commands module
from slash_commands import (
    SlashCommand,
    CommandCategory,
    CommandContext,
    CommandResponse,
    SlashCommandRegistry,
    SlashCommandHandler,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def command_registry():
    """Provide a slash command registry."""
    return SlashCommandRegistry()


@pytest.fixture
def command_handler():
    """Provide a command handler."""
    return SlashCommandHandler(ryzanstein_api_url="http://localhost:8000")


@pytest.fixture
def sample_code():
    """Provide sample Python code for testing."""
    return '''
def fibonacci(n):
    """Calculate fibonacci number at position n."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Result: {result}")
'''


@pytest.fixture
def command_context_with_code(sample_code):
    """Provide command context with sample code."""
    return CommandContext(
        selected_code=sample_code,
        current_file="test.py",
        file_language="python",
        git_context={
            "branch": "main",
            "commit": "abc123",
        },
        conversation_history=[],
        workspace_root="/home/user/project",
    )


# =============================================================================
# Test: Command Registry
# =============================================================================

class TestCommandRegistry:
    """Test the slash command registry."""

    def test_registry_initialization(self, command_registry):
        """Test that registry initializes with all commands."""
        assert len(command_registry.commands) > 0
        assert len(command_registry.commands) >= 40  # At least 40 commands

    def test_registry_has_core_commands(self, command_registry):
        """Test that registry has core commands."""
        core_commands = ["inference", "chat", "explain", "optimize", "test"]
        for cmd in core_commands:
            assert cmd in command_registry.commands

    def test_get_command(self, command_registry):
        """Test getting a command from registry."""
        command = command_registry.get_command("optimize")
        assert command is not None
        assert command.name == "optimize"
        assert command.elite_agent == "@VELOCITY"

    def test_get_nonexistent_command(self, command_registry):
        """Test getting non-existent command returns None."""
        command = command_registry.get_command("nonexistent")
        assert command is None

    def test_list_commands_all(self, command_registry):
        """Test listing all commands."""
        commands = command_registry.list_commands()
        assert len(commands) >= 40
        assert all(isinstance(c, SlashCommand) for c in commands)

    def test_list_commands_by_category(self, command_registry):
        """Test listing commands by category."""
        testing_commands = command_registry.list_commands(CommandCategory.TESTING)
        assert len(testing_commands) > 0
        assert all(c.category == CommandCategory.TESTING for c in testing_commands)

    def test_list_commands_by_agent(self, command_registry):
        """Test listing commands by Elite Agent."""
        velocity_commands = command_registry.list_by_agent("@VELOCITY")
        assert len(velocity_commands) > 0
        assert all(c.elite_agent == "@VELOCITY" for c in velocity_commands)

    def test_command_has_required_fields(self, command_registry):
        """Test that all commands have required fields."""
        for cmd in command_registry.list_commands():
            assert cmd.name
            assert cmd.description
            assert cmd.category
            assert cmd.elite_agent
            assert cmd.prompt_template
            assert isinstance(cmd.context_required, list)
            assert isinstance(cmd.streaming_enabled, bool)


# =============================================================================
# Test: Command Handler
# =============================================================================

class TestCommandHandler:
    """Test the command execution handler."""

    @pytest.mark.asyncio
    async def test_execute_valid_command(self, command_handler, command_context_with_code):
        """Test executing a valid command."""
        with patch.object(command_handler, "_call_ryzanstein_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "Mock response from Ryzanstein API"

            response = await command_handler.execute_command(
                "explain",
                command_context_with_code
            )

            assert response.command_name == "explain"
            assert response.content == "Mock response from Ryzanstein API"
            assert response.is_streaming is True
            assert response.agent_used == "@MENTOR"
            mock_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_command(self, command_handler, command_context_with_code):
        """Test executing an invalid command raises error."""
        with pytest.raises(ValueError, match="Unknown command"):
            await command_handler.execute_command(
                "nonexistent",
                command_context_with_code
            )

    @pytest.mark.asyncio
    async def test_execute_command_missing_context(self, command_handler):
        """Test executing command without required context raises error."""
        context = CommandContext()  # Empty context
        
        with pytest.raises(ValueError, match="requires"):
            await command_handler.execute_command(
                "explain",  # Requires selected_code
                context
            )

    @pytest.mark.asyncio
    async def test_execute_with_user_input(self, command_handler, command_context_with_code):
        """Test executing command with user input."""
        with patch.object(command_handler, "_call_ryzanstein_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "Response with user input incorporated"

            response = await command_handler.execute_command(
                "chat",
                command_context_with_code,
                user_input="What does this function do?"
            )

            assert response.content == "Response with user input incorporated"
            # Verify user input was passed to API
            call_args = mock_api.call_args
            assert "What does this function do?" in str(call_args)

    @pytest.mark.asyncio
    async def test_streaming_enabled_for_command(self, command_handler, command_context_with_code):
        """Test that streaming is enabled where appropriate."""
        with patch.object(command_handler, "_call_ryzanstein_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "Streaming response"

            response = await command_handler.execute_command(
                "optimize",
                command_context_with_code
            )

            # Check that streaming was requested
            call_args = mock_api.call_args
            assert call_args[1]["streaming"] is True


# =============================================================================
# Test: Specific Slash Commands
# =============================================================================

class TestSpecificCommands:
    """Test specific slash command behaviors."""

    def test_explain_command_exists(self, command_registry):
        """Test explain command."""
        cmd = command_registry.get_command("explain")
        assert cmd is not None
        assert cmd.elite_agent == "@MENTOR"
        assert CommandCategory.CODE_ANALYSIS in [cmd.category]

    def test_optimize_command_exists(self, command_registry):
        """Test optimize command."""
        cmd = command_registry.get_command("optimize")
        assert cmd is not None
        assert cmd.elite_agent == "@VELOCITY"
        assert cmd.category == CommandCategory.OPTIMIZATION

    def test_test_command_exists(self, command_registry):
        """Test test generation command."""
        cmd = command_registry.get_command("test")
        assert cmd is not None
        assert cmd.elite_agent == "@ECLIPSE"
        assert cmd.category == CommandCategory.TESTING

    def test_security_command_exists(self, command_registry):
        """Test security audit command."""
        cmd = command_registry.get_command("security")
        assert cmd is not None
        assert cmd.elite_agent == "@CIPHER"
        assert cmd.category == CommandCategory.SECURITY

    def test_arch_command_exists(self, command_registry):
        """Test architecture review command."""
        cmd = command_registry.get_command("arch")
        assert cmd is not None
        assert cmd.elite_agent == "@ARCHITECT"
        assert cmd.category == CommandCategory.ARCHITECTURE

    def test_doc_command_exists(self, command_registry):
        """Test documentation generation command."""
        cmd = command_registry.get_command("doc")
        assert cmd is not None
        assert cmd.elite_agent == "@SCRIBE"
        assert cmd.category == CommandCategory.DOCUMENTATION

    def test_api_command_exists(self, command_registry):
        """Test API design command."""
        cmd = command_registry.get_command("api")
        assert cmd is not None
        assert cmd.elite_agent == "@SYNAPSE"
        assert cmd.category == CommandCategory.API_DESIGN

    def test_query_command_exists(self, command_registry):
        """Test database query optimization command."""
        cmd = command_registry.get_command("query")
        assert cmd is not None
        assert cmd.elite_agent == "@VERTEX"
        assert cmd.category == CommandCategory.DATABASE

    def test_deploy_command_exists(self, command_registry):
        """Test deployment command."""
        cmd = command_registry.get_command("deploy")
        assert cmd is not None
        assert cmd.elite_agent == "@FLUX"
        assert cmd.category == CommandCategory.DEVOPS

    def test_ml_command_exists(self, command_registry):
        """Test ML implementation command."""
        cmd = command_registry.get_command("ml")
        assert cmd is not None
        assert cmd.elite_agent == "@TENSOR"
        assert cmd.category == CommandCategory.MACHINE_LEARNING


# =============================================================================
# Test: Context Management
# =============================================================================

class TestContextManagement:
    """Test command context handling."""

    def test_create_empty_context(self):
        """Test creating empty command context."""
        context = CommandContext()
        assert context.selected_code is None
        assert context.current_file is None
        assert context.conversation_history is None

    def test_create_context_with_code(self, sample_code):
        """Test creating context with code."""
        context = CommandContext(selected_code=sample_code)
        assert context.selected_code == sample_code

    def test_context_preserves_metadata(self, command_context_with_code):
        """Test that context preserves all metadata."""
        assert command_context_with_code.file_language == "python"
        assert command_context_with_code.current_file == "test.py"
        assert command_context_with_code.workspace_root is not None
        assert command_context_with_code.git_context is not None

    def test_conversation_history_management(self):
        """Test conversation history in context."""
        history = [
            {"role": "user", "content": "Explain this code"},
            {"role": "assistant", "content": "This code does..."},
        ]
        context = CommandContext(conversation_history=history)
        assert len(context.conversation_history) == 2
        assert context.conversation_history[0]["role"] == "user"


# =============================================================================
# Test: API Integration
# =============================================================================

class TestAPIIntegration:
    """Test integration with Ryzanstein API."""

    @pytest.mark.asyncio
    async def test_api_call_success(self, command_handler):
        """Test successful API call."""
        with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"text": "Response from API"})
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await command_handler._call_ryzanstein_api(
                "Test prompt",
                streaming=False
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_api_call_with_timeout(self, command_handler):
        """Test API call respects timeout."""
        with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
            # Simulate timeout
            mock_post.side_effect = asyncio.TimeoutError()

            # This should be handled gracefully
            with pytest.raises(asyncio.TimeoutError):
                await command_handler._call_ryzanstein_api(
                    "Test prompt",
                    timeout=1000
                )

    @pytest.mark.asyncio
    async def test_api_call_with_streaming(self, command_handler):
        """Test API call with streaming enabled."""
        with patch.object(command_handler, "_call_ryzanstein_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "Streamed response"

            # Would verify streaming behavior
            # This is a placeholder for actual streaming tests
            pass


# =============================================================================
# Test: Elite Agent Routing
# =============================================================================

class TestEliteAgentRouting:
    """Test routing to correct Elite Agent."""

    def test_velocity_commands_routed_correctly(self, command_registry):
        """Test that optimization commands go to @VELOCITY."""
        commands = command_registry.list_by_agent("@VELOCITY")
        assert len(commands) > 0
        assert all(c.elite_agent == "@VELOCITY" for c in commands)

    def test_eclipse_commands_routed_correctly(self, command_registry):
        """Test that testing commands go to @ECLIPSE."""
        commands = command_registry.list_by_agent("@ECLIPSE")
        assert len(commands) > 0
        assert all(c.elite_agent == "@ECLIPSE" for c in commands)

    def test_cipher_commands_routed_correctly(self, command_registry):
        """Test that security commands go to @CIPHER."""
        commands = command_registry.list_by_agent("@CIPHER")
        assert len(commands) > 0
        assert all(c.elite_agent == "@CIPHER" for c in commands)

    def test_architect_commands_routed_correctly(self, command_registry):
        """Test that architecture commands go to @ARCHITECT."""
        commands = command_registry.list_by_agent("@ARCHITECT")
        assert len(commands) > 0
        assert all(c.elite_agent == "@ARCHITECT" for c in commands)


# =============================================================================
# Test: Response Handling
# =============================================================================

class TestResponseHandling:
    """Test response handling and formatting."""

    def test_command_response_structure(self):
        """Test command response has proper structure."""
        response = CommandResponse(
            command_name="test",
            content="Test content",
            is_streaming=True,
            metadata={"agent": "@ECLIPSE"},
            agent_used="@ECLIPSE"
        )

        assert response.command_name == "test"
        assert response.content == "Test content"
        assert response.is_streaming is True
        assert response.agent_used == "@ECLIPSE"

    def test_response_with_metadata(self):
        """Test response can include metadata."""
        metadata = {
            "agent": "@VELOCITY",
            "category": "optimization",
            "execution_time_ms": 245,
            "tokens_used": 1024,
        }
        response = CommandResponse(
            command_name="optimize",
            content="Optimized code",
            is_streaming=False,
            metadata=metadata,
            agent_used="@VELOCITY"
        )

        assert response.metadata["execution_time_ms"] == 245
        assert response.metadata["tokens_used"] == 1024


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in command execution."""

    @pytest.mark.asyncio
    async def test_handle_invalid_command_gracefully(self, command_handler, command_context_with_code):
        """Test handling of invalid commands."""
        with pytest.raises(ValueError) as exc_info:
            await command_handler.execute_command("invalid_cmd", command_context_with_code)

        assert "Unknown command" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_missing_context_gracefully(self, command_handler):
        """Test handling of missing required context."""
        context = CommandContext()  # No code
        
        with pytest.raises(ValueError) as exc_info:
            await command_handler.execute_command("explain", context)

        assert "requires" in str(exc_info.value).lower()

    def test_logging_of_errors(self, caplog):
        """Test that errors are logged."""
        registry = SlashCommandRegistry()
        
        with caplog.at_level(logging.INFO):
            # Logging happens during registration
            pass
        
        # Check that some registration was logged
        assert "Registered slash command" in caplog.text


# =============================================================================
# Test: Performance & Load
# =============================================================================

class TestPerformance:
    """Test performance and load handling."""

    def test_registry_initialization_performance(self):
        """Test that registry initializes quickly."""
        import time
        start = time.time()
        registry = SlashCommandRegistry()
        elapsed = time.time() - start
        
        # Should initialize in under 100ms
        assert elapsed < 0.1
        assert len(registry.commands) >= 40

    @pytest.mark.asyncio
    async def test_concurrent_command_execution(self, command_handler, command_context_with_code):
        """Test executing multiple commands concurrently."""
        with patch.object(command_handler, "_call_ryzanstein_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "Response"

            # Execute multiple commands concurrently
            tasks = [
                command_handler.execute_command("explain", command_context_with_code),
                command_handler.execute_command("optimize", command_context_with_code),
                command_handler.execute_command("test", command_context_with_code),
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(isinstance(r, CommandResponse) for r in results)


# =============================================================================
# Pytest Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
