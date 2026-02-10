<script lang="ts">
  import { onMount } from "svelte";
  import ChatPanel from "./components/ChatPanel.svelte";
  import ModelSelector from "./components/ModelSelector.svelte";
  import AgentPanel from "./components/AgentPanel.svelte";
  import SettingsPanel from "./components/SettingsPanel.svelte";
  import {
    Greet,
    SendMessage,
    ListModels,
    ListAgents,
    GetConfig,
  } from "../wailsjs/go/main/App";
  import { EventsOn } from "../wailsjs/runtime/runtime";

  let activeTab = "chat";
  let config = null;
  let isLoading = true;

  onMount(async () => {
    try {
      // Load initial configuration
      config = await GetConfig();

      // Load models
      await ListModels();

      // Load agents
      await ListAgents();

      // Listen for app events
      EventsOn("app:ready", (data) => {
        console.log("App ready:", data);
      });

      EventsOn("model:loaded", (data) => {
        console.log("Model loaded:", data);
      });

      EventsOn("chat:message", (message) => {
        console.log("Message sent:", message);
      });

      isLoading = false;
    } catch (error) {
      console.error("Error initializing app:", error);
      isLoading = false;
    }
  });
</script>

<main class="app-container">
  <header class="app-header">
    <div class="header-left">
      <h1>Ryzanstein</h1>
      <p class="subtitle">Elite AI Agent Desktop Environment</p>
    </div>
    <div class="header-right">
      <span class="status">Ready</span>
    </div>
  </header>

  <div class="app-content">
    <!-- Sidebar Navigation -->
    <nav class="sidebar">
      <button
        class="nav-item {activeTab === 'chat' ? 'active' : ''}"
        on:click={() => (activeTab = "chat")}
      >
        💬 Chat
      </button>
      <button
        class="nav-item {activeTab === 'models' ? 'active' : ''}"
        on:click={() => (activeTab = "models")}
      >
        🤖 Models
      </button>
      <button
        class="nav-item {activeTab === 'agents' ? 'active' : ''}"
        on:click={() => (activeTab = "agents")}
      >
        🧠 Agents
      </button>
      <button
        class="nav-item {activeTab === 'settings' ? 'active' : ''}"
        on:click={() => (activeTab = "settings")}
      >
        ⚙️ Settings
      </button>
    </nav>

    <!-- Main Content -->
    <div class="main-content">
      {#if isLoading}
        <div class="loading">
          <p>Loading Ryzanstein...</p>
        </div>
      {:else if activeTab === "chat"}
        <ChatPanel {config} />
      {:else if activeTab === "models"}
        <ModelSelector {config} />
      {:else if activeTab === "agents"}
        <AgentPanel {config} />
      {:else if activeTab === "settings"}
        <SettingsPanel {config} />
      {/if}
    </div>
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
      Ubuntu, Cantarell, sans-serif;
    background-color: #1a1e27;
    color: #e0e0e0;
  }

  :global(*) {
    box-sizing: border-box;
  }

  .app-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: linear-gradient(135deg, #1a1e27 0%, #2d1f3a 100%);
  }

  .app-header {
    background: rgba(0, 0, 0, 0.3);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(10px);
  }

  .header-left h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .subtitle {
    margin: 4px 0 0 0;
    font-size: 12px;
    color: #888;
  }

  .status {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(0, 200, 100, 0.2);
    color: #00c864;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
  }

  .app-content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .sidebar {
    width: 200px;
    background: rgba(0, 0, 0, 0.2);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    padding: 16px 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    overflow-y: auto;
  }

  .nav-item {
    padding: 12px 16px;
    background: none;
    border: none;
    color: #aaa;
    text-align: left;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s ease;
    border-left: 3px solid transparent;
  }

  .nav-item:hover {
    color: #e0e0e0;
    background: rgba(255, 255, 255, 0.05);
  }

  .nav-item.active {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
    border-left-color: #00d4ff;
  }

  .main-content {
    flex: 1;
    overflow: auto;
    padding: 24px;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    font-size: 18px;
    color: #888;
  }
</style>
