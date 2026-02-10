import * as vscode from "vscode";
import { RyzansteinClient } from "../client/RyzansteinClient";

export class ChatWebviewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "ryzanstein.chatView";

  private view?: vscode.WebviewView;
  private disposables: vscode.Disposable[] = [];

  constructor(
    private readonly _extensionUri: vscode.Uri,
    private ryzansteinClient: RyzansteinClient
  ) {}

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    this.view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(
      (data) => {
        this._handleWebviewMessage(data);
      },
      null,
      this.disposables
    );
  }

  private _getHtmlForWebview(webview: vscode.Webview): string {
    const nonce = this._getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ryzanstein Chat</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--vscode-editor-background);
      color: var(--vscode-editor-foreground);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    #messages { 
      flex: 1; 
      overflow-y: auto; 
      padding: 1rem; 
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .message { 
      padding: 0.75rem; 
      border-radius: 4px;
      max-width: 90%;
      word-wrap: break-word;
    }
    .message.user { 
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      align-self: flex-end;
    }
    .message.assistant { 
      background: var(--vscode-editor-lineHighlightBackground);
      align-self: flex-start;
    }
    #input-container {
      display: flex;
      gap: 0.5rem;
      padding: 1rem;
      border-top: 1px solid var(--vscode-panel-border);
      background: var(--vscode-panel-background);
    }
    #message-input {
      flex: 1;
      padding: 0.75rem;
      border: 1px solid var(--vscode-input-border);
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border-radius: 4px;
      font-size: 14px;
    }
    #send-btn {
      padding: 0.75rem 1.5rem;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.2s;
    }
    #send-btn:hover {
      background: var(--vscode-button-hoverBackground);
    }
    #send-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <div id="messages"></div>
  <div id="input-container">
    <input type="text" id="message-input" placeholder="Ask Ryzanstein...">
    <button id="send-btn">Send</button>
  </div>
  
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const messagesDiv = document.getElementById('messages');
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMessage();
    });

    function sendMessage() {
      const text = input.value.trim();
      if (!text) return;

      // Add user message
      const userMsg = document.createElement('div');
      userMsg.className = 'message user';
      userMsg.textContent = text;
      messagesDiv.appendChild(userMsg);

      vscode.postMessage({
        command: 'sendMessage',
        text: text
      });

      input.value = '';
      input.focus();
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    window.addEventListener('message', (event) => {
      const message = event.data;
      switch (message.command) {
        case 'addMessage':
          const assistantMsg = document.createElement('div');
          assistantMsg.className = 'message assistant';
          assistantMsg.textContent = message.content;
          messagesDiv.appendChild(assistantMsg);
          messagesDiv.scrollTop = messagesDiv.scrollHeight;
          break;
      }
    });
  </script>
</body>
</html>`;
  }

  private _getNonce(): string {
    let text = "";
    const possible =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (let i = 0; i < 32; i++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
  }

  private async _handleWebviewMessage(data: any) {
    switch (data.command) {
      case "sendMessage":
        try {
          const response = await this.ryzansteinClient.chat(
            data.text,
            "default"
          );
          this.view?.webview.postMessage({
            command: "addMessage",
            content: response.response,
          });
        } catch (error) {
          this.view?.webview.postMessage({
            command: "addMessage",
            content: `Error: ${
              error instanceof Error ? error.message : "Unknown error"
            }`,
          });
        }
        break;
    }
  }

  public dispose() {
    this.disposables.forEach((d) => d.dispose());
  }
}
