import * as vscode from "vscode";
import { ExtensionContext } from "vscode";
import { RyzansteinClient } from "../client/RyzansteinClient";
import { MCPClient } from "../client/MCPClient";
import { ChatWebviewProvider } from "../providers/ChatWebviewProvider";

export class CommandHandler {
  constructor(
    context: ExtensionContext,
    private ryzansteinClient: RyzansteinClient,
    private mcpClient: MCPClient
  ) {
    this.registerCommands(context);
  }

  private registerCommands(context: ExtensionContext) {
    // Open Chat Command
    const openChatCommand = vscode.commands.registerCommand(
      "ryzanstein.openChat",
      () => {
        vscode.window.showInformationMessage("💬 Ryzanstein Chat View Ready");
      }
    );

    // Select Agent Command
    const selectAgentCommand = vscode.commands.registerCommand(
      "ryzanstein.selectAgent",
      async () => {
        const agents = await this.ryzansteinClient.listAgents();
        const agentNames = agents.map((a) => a.name);
        const selected = await vscode.window.showQuickPick(agentNames);
        if (selected) {
          vscode.workspace
            .getConfiguration("ryzanstein")
            .update("selectedAgent", selected);
          vscode.window.showInformationMessage(`✓ Selected agent: ${selected}`);
        }
      }
    );

    // Load Model Command
    const loadModelCommand = vscode.commands.registerCommand(
      "ryzanstein.loadModel",
      async (modelId: string) => {
        try {
          await this.ryzansteinClient.loadModel(modelId);
          vscode.window.showInformationMessage(`✓ Loaded model: ${modelId}`);
        } catch (error) {
          vscode.window.showErrorMessage(
            `Failed to load model: ${
              error instanceof Error ? error.message : "Unknown error"
            }`
          );
        }
      }
    );

    // Generate Code Command
    const generateCodeCommand = vscode.commands.registerCommand(
      "ryzanstein.generateCode",
      async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
          vscode.window.showErrorMessage("No active editor");
          return;
        }

        const prompt = await vscode.window.showInputBox({
          prompt: "Enter your code generation prompt",
          placeHolder: "e.g., Generate a TypeScript function to...",
        });

        if (!prompt) return;

        try {
          const code = await this.ryzansteinClient.generateCode(prompt);
          editor.edit((editBuilder) => {
            editBuilder.insert(editor.selection.active, code);
          });
        } catch (error) {
          vscode.window.showErrorMessage(
            `Code generation failed: ${
              error instanceof Error ? error.message : "Unknown error"
            }`
          );
        }
      }
    );

    context.subscriptions.push(
      openChatCommand,
      selectAgentCommand,
      loadModelCommand,
      generateCodeCommand
    );
  }
}
