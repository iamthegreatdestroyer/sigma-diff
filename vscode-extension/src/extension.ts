import * as vscode from "vscode";
import { ExtensionContext, window, commands, ViewColumn } from "vscode";
import { AgentTreeProvider } from "./providers/AgentTreeProvider";
import { ModelTreeProvider } from "./providers/ModelTreeProvider";
import { ChatWebviewProvider } from "./providers/ChatWebviewProvider";
import {
  RyzansteinChatModelProvider,
  RyzansteinChatResponseProvider,
} from "./providers/RyzansteinChatModelProvider";
import { RyzansteinClient } from "./client/RyzansteinClient";
import { MCPClient } from "./client/MCPClient";
import { CommandHandler } from "./commands/CommandHandler";

let extensionContext: ExtensionContext;
let ryzansteinClient: RyzansteinClient;
let mcpClient: MCPClient;
let commandHandler: CommandHandler;

export async function activate(context: ExtensionContext) {
  console.log("Ryzanstein extension activating...");

  extensionContext = context;

  // Initialize clients
  const config = vscode.workspace.getConfiguration("ryzanstein");

  ryzansteinClient = new RyzansteinClient(
    config.get("ryzansteinApiUrl") || "http://localhost:8000"
  );

  mcpClient = new MCPClient(config.get("mcpServerUrl") || "localhost:50051");

  // Auto-connect to MCP if configured
  if (config.get("autoConnect")) {
    try {
      await mcpClient.connect();
      window.showInformationMessage("✅ Connected to Ryzanstein MCP server");
    } catch (error) {
      console.error("Failed to connect to MCP server:", error);
      window.showWarningMessage(
        "⚠️ Could not connect to Ryzanstein MCP server"
      );
    }
  }

  // Initialize command handler
  commandHandler = new CommandHandler(context, ryzansteinClient, mcpClient);

  // Register tree view providers
  const agentProvider = new AgentTreeProvider(mcpClient);
  const modelProvider = new ModelTreeProvider(ryzansteinClient);

  vscode.window.registerTreeDataProvider("ryzanstein.agents", agentProvider);
  vscode.window.registerTreeDataProvider("ryzanstein.models", modelProvider);

  // Register Copilot Chat model provider
  const chatModelProvider = new RyzansteinChatModelProvider(ryzansteinClient);
  const chatResponseProvider = new RyzansteinChatResponseProvider(
    ryzansteinClient
  );

  context.subscriptions.push(
    vscode.chat.registerChatModelProvider("ryzanstein", chatModelProvider),
    vscode.chat.registerChatResponseProvider(
      { vendor: "ryzanstein" },
      chatResponseProvider
    )
  );

  // Register chat webview provider
  const chatProvider = new ChatWebviewProvider(
    extensionContext,
    ryzansteinClient,
    mcpClient
  );
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("ryzanstein.chat", chatProvider)
  );

  // Register all commands
  commandHandler.registerCommands();

  // Status bar
  const statusBar = window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.command = "ryzanstein.openChat";
  statusBar.text = "$(robot) Ryzanstein";
  statusBar.tooltip = "Click to open Ryzanstein chat";
  statusBar.show();
  context.subscriptions.push(statusBar);

  console.log("Ryzanstein extension activated successfully");
}

export function deactivate() {
  console.log("Ryzanstein extension deactivating...");
  if (mcpClient) {
    mcpClient.disconnect();
  }
}
