import * as vscode from "vscode";
import { ExtensionContext, window, commands, ViewColumn } from "vscode";
import { AgentTreeProvider } from "./providers/AgentTreeProvider";
import { ModelTreeProvider } from "./providers/ModelTreeProvider";
import { ChatWebviewProvider } from "./providers/ChatWebviewProvider";
// NOTE (build fix, 2026-07-08): "./providers/RyzansteinChatModelProvider" never existed in
// this repo, and the API it targeted (vscode.chat.registerChatModelProvider /
// registerChatResponseProvider) has been removed from the VS Code extension API this
// project targets (engines.vscode ^1.85.0 only exposes vscode.chat.createChatParticipant
// with a ChatRequestHandler callback - a different, participant-based model). Porting to
// createChatParticipant requires a new request-handler implementation backed by a real
// chat participant contribution point in package.json, which is a feature undertaking
// beyond a build fix. Disabled below rather than silently dropped - the in-house
// ChatWebviewProvider view (registered further down) still provides chat functionality.
// import {
//   RyzansteinChatModelProvider,
//   RyzansteinChatResponseProvider,
// } from "./providers/RyzansteinChatModelProvider";
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

  // Initialize command handler (constructor registers commands against context)
  commandHandler = new CommandHandler(context, ryzansteinClient, mcpClient);

  // Register tree view providers
  const agentProvider = new AgentTreeProvider(mcpClient);
  const modelProvider = new ModelTreeProvider(ryzansteinClient);

  vscode.window.registerTreeDataProvider("ryzanstein.agents", agentProvider);
  vscode.window.registerTreeDataProvider("ryzanstein.models", modelProvider);

  // Copilot Chat model/response provider registration disabled - see note above the
  // commented-out import. vscode.chat no longer exposes registerChatModelProvider /
  // registerChatResponseProvider on the targeted API version (^1.85.0); the current
  // equivalent is vscode.chat.createChatParticipant, which needs a real
  // ChatRequestHandler implementation to port to (not just a mechanical rename).
  //
  // const chatModelProvider = new RyzansteinChatModelProvider(ryzansteinClient);
  // const chatResponseProvider = new RyzansteinChatResponseProvider(
  //   ryzansteinClient
  // );
  //
  // context.subscriptions.push(
  //   vscode.chat.registerChatModelProvider("ryzanstein", chatModelProvider),
  //   vscode.chat.registerChatResponseProvider(
  //     { vendor: "ryzanstein" },
  //     chatResponseProvider
  //   )
  // );

  // Register chat webview provider
  const chatProvider = new ChatWebviewProvider(
    extensionContext.extensionUri,
    ryzansteinClient
  );
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("ryzanstein.chat", chatProvider)
  );

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
