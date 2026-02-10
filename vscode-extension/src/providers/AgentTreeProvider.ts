import * as vscode from "vscode";
import { MCPClient } from "../client/MCPClient";

export class AgentTreeProvider
  implements vscode.TreeDataProvider<AgentTreeItem>
{
  private _onDidChangeTreeData: vscode.EventEmitter<
    AgentTreeItem | undefined | null | void
  > = new vscode.EventEmitter<AgentTreeItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    AgentTreeItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  private agents: AgentTreeItem[] = [];

  constructor(private mcpClient: MCPClient) {
    this.loadAgents();
  }

  async loadAgents() {
    try {
      // This will be populated when MCP client connects
      this.agents = [
        new AgentTreeItem(
          "APEX",
          "Computer Science Engineering",
          vscode.TreeItemCollapsibleState.None
        ),
        new AgentTreeItem(
          "ARCHITECT",
          "Systems Architecture",
          vscode.TreeItemCollapsibleState.None
        ),
        new AgentTreeItem(
          "CIPHER",
          "Cryptography & Security",
          vscode.TreeItemCollapsibleState.None
        ),
        new AgentTreeItem(
          "TENSOR",
          "Machine Learning",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
      this._onDidChangeTreeData.fire();
    } catch (error) {
      console.error("Failed to load agents:", error);
    }
  }

  getTreeItem(
    element: AgentTreeItem
  ): vscode.TreeItem | Thenable<vscode.TreeItem> {
    return element;
  }

  getChildren(element?: AgentTreeItem): vscode.ProviderResult<AgentTreeItem[]> {
    if (element === undefined) {
      return this.agents;
    }
    return [];
  }
}

class AgentTreeItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly description: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState
  ) {
    super(label, collapsibleState);
    this.description = description;
    this.tooltip = `${label}: ${description}`;
    this.iconPath = new vscode.ThemeIcon("robot");
  }
}
