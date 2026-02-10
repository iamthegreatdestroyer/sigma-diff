<script lang="ts">
  import { onMount } from "svelte";
  import { SendMessage, GetHistory } from "../../wailsjs/go/main/App";
  import { EventsOn } from "../../wailsjs/runtime/runtime";

  export let config = null;

  let messages = [];
  let inputValue = "";
  let selectedModel = config?.defaultModel || "ryzanstein-7b";
  let selectedAgent = config?.defaultAgent || "@APEX";
  let isLoading = false;
  let messageContainer;

  onMount(async () => {
    // Load chat history
    try {
      messages = await GetHistory(50);
    } catch (error) {
      console.error("Error loading history:", error);
    }

    // Listen for incoming messages
    EventsOn("chat:message", (message) => {
      messages = [...messages, message];
      scrollToBottom();
    });

    EventsOn("chat:response", (message) => {
      messages = [...messages, message];
      scrollToBottom();
    });
  });

  const scrollToBottom = () => {
    if (messageContainer) {
      messageContainer.scrollTop = messageContainer.scrollHeight;
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const message = inputValue;
    inputValue = "";
    isLoading = true;

    try {
      const response = await SendMessage(message, selectedModel, selectedAgent);
      console.log("Response:", response);
    } catch (error) {
      console.error("Error sending message:", error);
      messages = [
        ...messages,
        {
          id: `error_${Date.now()}`,
          role: "system",
          content: `Error: ${error}`,
          timestamp: Date.now() / 1000,
        },
      ];
    } finally {
      isLoading = false;
      scrollToBottom();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
</script>

<div class="chat-panel">
  <div class="chat-header">
    <h2>Chat Interface</h2>
    <div class="controls">
      <select bind:value={selectedModel} class="select">
        <option value="ryzanstein-7b">Ryzanstein 7B</option>
        <option value="ryzanstein-13b">Ryzanstein 13B</option>
      </select>
      <select bind:value={selectedAgent} class="select">
        <option value="@APEX">@APEX - Engineering</option>
        <option value="@CIPHER">@CIPHER - Security</option>
        <option value="@ARCHITECT">@ARCHITECT - Architecture</option>
      </select>
    </div>
  </div>

  <div class="messages-container" bind:this={messageContainer}>
    {#each messages as message (message.id)}
      <div class="message message-{message.role}">
        <div class="message-badge">{message.role}</div>
        <div class="message-content">{message.content}</div>
        <div class="message-time">
          {new Date(message.timestamp * 1000).toLocaleTimeString()}
        </div>
      </div>
    {/each}
    {#if isLoading}
      <div class="message message-assistant loading">
        <div class="typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    {/if}
  </div>

  <div class="input-area">
    <textarea
      bind:value={inputValue}
      placeholder="Type your message... (Shift+Enter for new line)"
      on:keypress={handleKeyPress}
      disabled={isLoading}
      class="message-input"
    ></textarea>
    <button
      on:click={handleSendMessage}
      disabled={isLoading || !inputValue.trim()}
      class="send-button"
    >
      Send
    </button>
  </div>
</div>

<style>
  .chat-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    gap: 12px;
  }

  .chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .chat-header h2 {
    margin: 0;
    font-size: 18px;
  }

  .controls {
    display: flex;
    gap: 12px;
  }

  .select {
    padding: 6px 12px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 12px;
    cursor: pointer;
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px 0;
  }

  .message {
    padding: 12px;
    border-radius: 8px;
    margin: 0 12px;
    animation: slideIn 0.3s ease;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .message-user {
    background: rgba(0, 212, 255, 0.1);
    border-left: 3px solid #00d4ff;
    margin-left: 40px;
  }

  .message-assistant {
    background: rgba(124, 58, 237, 0.1);
    border-left: 3px solid #7c3aed;
    margin-right: 40px;
  }

  .message-system {
    background: rgba(255, 100, 100, 0.1);
    border-left: 3px solid #ff6464;
  }

  .message-badge {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 4px;
  }

  .message-content {
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
  }

  .message-time {
    font-size: 11px;
    color: #666;
    margin-top: 6px;
  }

  .message.loading {
    display: flex;
    align-items: center;
  }

  .typing-indicator {
    display: flex;
    gap: 4px;
  }

  .typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #7c3aed;
    animation: typing 1.4s infinite;
  }

  .typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
  }

  .typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
  }

  @keyframes typing {
    0%,
    60%,
    100% {
      opacity: 0.3;
    }
    30% {
      opacity: 1;
    }
  }

  .input-area {
    display: flex;
    gap: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .message-input {
    flex: 1;
    padding: 12px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 14px;
    resize: none;
    max-height: 120px;
  }

  .message-input:focus {
    outline: none;
    border-color: #00d4ff;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1);
  }

  .message-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .send-button {
    padding: 12px 24px;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    border: none;
    border-radius: 4px;
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .send-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
  }

  .send-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
