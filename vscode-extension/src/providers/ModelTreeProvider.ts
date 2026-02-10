import * as vscode from "vscode";
import { RyzansteinClient } from "../client/RyzansteinClient";

export class ModelTreeProvider
  implements vscode.TreeDataProvider<ModelTreeItem>
{
  private _onDidChangeTreeData: vscode.EventEmitter<
    ModelTreeItem | undefined | null | void
  > = new vscode.EventEmitter<ModelTreeItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    ModelTreeItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  private models: ModelTreeItem[] = [];

  constructor(private ryzansteinClient: RyzansteinClient) {
    this.loadModels();
  }

  async loadModels() {
    try {
      const models = await this.ryzansteinClient.listModels();
      this.models = models.map(
        (model: any) =>
          new ModelTreeItem(
            model.name,
            model.id,
            vscode.TreeItemCollapsibleState.None
          )
      );
      this._onDidChangeTreeData.fire();
    } catch (error) {
      console.error("Failed to load models:", error);
      this.models = [
        new ModelTreeItem(
          "Llama 2",
          "llama-2",
          vscode.TreeItemCollapsibleState.None
        ),
        new ModelTreeItem(
          "Mistral",
          "mistral",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
      this._onDidChangeTreeData.fire();
    }
  }

  getTreeItem(
    element: ModelTreeItem
  ): vscode.TreeItem | Thenable<vscode.TreeItem> {
    return element;
  }

  getChildren(element?: ModelTreeItem): vscode.ProviderResult<ModelTreeItem[]> {
    if (element === undefined) {
      return this.models;
    }
    return [];
  }

  refresh() {
    this._onDidChangeTreeData.fire();
    this.loadModels();
  }
}

class ModelTreeItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly modelId: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState
  ) {
    super(label, collapsibleState);
    this.tooltip = `Model: ${label} (${modelId})`;
    this.iconPath = new vscode.ThemeIcon("package");
    this.command = {
      title: "Load Model",
      command: "ryzanstein.loadModel",
      arguments: [modelId],
    };
  }
}
