import * as vscode from 'vscode';
import { RyzansteinAPI } from '../services/ryzansteinAPI';

export class ChatPanel {
    constructor(
        private panel: vscode.WebviewPanel,
        private api: RyzansteinAPI
    ) {
        this.panel.webview.html = this.getHtmlContent();
        this.setupMessageHandlers();
    }

    private setupMessageHandlers() {
        this.panel.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'sendMessage':
                    await this.handleMessage(message.text);
                    break;
                case 'selectAgent':
                    await this.selectAgent(message.agentId);
                    break;
                case 'getAgents':
                    await this.getAgents();
                    break;
            }
        });
    }

    private async handleMessage(text: string) {
        const config = vscode.workspace.getConfiguration('ryzanstein');
        const agentId = config.get<string>('selectedAgent') || 'default';

        try {
            const response = await this.api.chat(text, agentId);
            this.panel.webview.postMessage({
                command: 'addMessage',
                role: 'assistant',
                content: response.response,
            });
        } catch (error) {
            this.panel.webview.postMessage({
                command: 'error',
                message: error instanceof Error ? error.message : 'Unknown error',
            });
        }
    }

    private async selectAgent(agentId: string) {
        vscode.workspace.getConfiguration('ryzanstein').update('selectedAgent', agentId);
    }

    private async getAgents() {
        try {
            const agents = await this.api.listAgents();
            this.panel.webview.postMessage({
                command: 'setAgents',
                agents: agents,
            });
        } catch (error) {
            console.error('Error loading agents:', error);
        }
    }

    private getHtmlContent(): string {
        const nonce = this.getNonce();
        return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ryzanstein Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
        #chat-container { display: flex; flex-direction: column; height: 100vh; }
        #messages { flex: 1; overflow-y: auto; padding: 1rem; }
        .message { margin: 0.5rem 0; padding: 0.75rem; border-radius: 4px; }
        .message.user { background: var(--vscode-button-background); color: var(--vscode-button-foreground); margin-left: 2rem; }
        .message.assistant { background: var(--vscode-editor-lineHighlightBackground); margin-right: 2rem; }
        #input-area { display: flex; gap: 0.5rem; padding: 1rem; border-top: 1px solid var(--vscode-panel-border); }
        #message-input { flex: 1; padding: 0.75rem; border: 1px solid var(--vscode-input-border); background: var(--vscode-input-background); color: var(--vscode-input-foreground); border-radius: 4px; font-size: 14px; }
        #send-button { padding: 0.75rem 1.5rem; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; border-radius: 4px; cursor: pointer; }
        #send-button:hover { background: var(--vscode-button-hoverBackground); }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="message-input" placeholder="Type a message...">
            <button id="send-button">Send</button>
        </div>
    </div>
    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        sendButton.addEventListener('click', () => {
            if (input.value.trim()) {
                vscode.postMessage({
                    command: 'sendMessage',
                    text: input.value,
                });
                input.value = '';
            }
        });

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });

        window.addEventListener('message', (event) => {
            const message = event.data;
            switch (message.command) {
                case 'addMessage':
                    const div = document.createElement('div');
                    div.className = \`message \${message.role}\`;
                    div.textContent = message.content;
                    messagesDiv.appendChild(div);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    break;
            }
        });

        vscode.postMessage({ command: 'getAgents' });
    </script>
</body>
</html>`;
    }

    private getNonce(): string {
        let text = '';
        const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        for (let i = 0; i < 32; i++) {
            text += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        return text;
    }

    reveal() {
        this.panel.reveal(vscode.ViewColumn.Beside);
    }
}
