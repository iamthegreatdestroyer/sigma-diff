import * as vscode from "vscode";
import { RyzansteinClient } from "../client/RyzansteinClient";

export class RyzansteinChatModelProvider implements vscode.ChatModelProvider {
  constructor(private ryzansteinClient: RyzansteinClient) {}

  async provideChatModels(
    token: vscode.CancellationToken
  ): Promise<vscode.ChatModel[]> {
    const models: vscode.ChatModel[] = [
      {
        id: "ryzanstein",
        vendor: "ryzanstein",
        family: "ryzanstein",
        name: "Ryzanstein",
        label: "Ryzanstein",
        version: "1.0.0",
        maxInputTokens: 4096,
        maxOutputTokens: 2048,
      },
      {
        id: "ryzanstein-apex",
        vendor: "ryzanstein",
        family: "ryzanstein-agent",
        name: "APEX",
        label: "Ryzanstein (@APEX - Elite CS Engineering)",
        version: "1.0.0",
        maxInputTokens: 4096,
        maxOutputTokens: 2048,
      },
      {
        id: "ryzanstein-architect",
        vendor: "ryzanstein",
        family: "ryzanstein-agent",
        name: "ARCHITECT",
        label: "Ryzanstein (@ARCHITECT - Systems Design)",
        version: "1.0.0",
        maxInputTokens: 4096,
        maxOutputTokens: 2048,
      },
      {
        id: "ryzanstein-tensor",
        vendor: "ryzanstein",
        family: "ryzanstein-agent",
        name: "TENSOR",
        label: "Ryzanstein (@TENSOR - Machine Learning)",
        version: "1.0.0",
        maxInputTokens: 4096,
        maxOutputTokens: 2048,
      },
      {
        id: "ryzanstein-cipher",
        vendor: "ryzanstein",
        family: "ryzanstein-agent",
        name: "CIPHER",
        label: "Ryzanstein (@CIPHER - Security)",
        version: "1.0.0",
        maxInputTokens: 4096,
        maxOutputTokens: 2048,
      },
    ];

    return models;
  }
}

export class RyzansteinChatResponseProvider
  implements vscode.ChatResponseProvider
{
  constructor(private ryzansteinClient: RyzansteinClient) {}

  async provideUsageDetails(
    chatModel: vscode.ChatModel,
    options: vscode.ChatRequestOptions,
    token: vscode.CancellationToken
  ): Promise<vscode.ChatUsageDetails | undefined> {
    return {
      inputTokenCount: 0,
      outputTokenCount: 0,
    };
  }

  async provideChatResponse(
    chat: vscode.ChatMessage[],
    model: vscode.ChatModel,
    options: vscode.ChatRequestOptions,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken
  ): Promise<void> {
    const lastMessage = chat[chat.length - 1];
    const userInput =
      lastMessage.role === vscode.ChatRole.User
        ? lastMessage.content
        : "Help me";

    // Extract agent name from model ID
    let agent = "default";
    if (model.id.includes("apex")) agent = "APEX";
    else if (model.id.includes("architect")) agent = "ARCHITECT";
    else if (model.id.includes("tensor")) agent = "TENSOR";
    else if (model.id.includes("cipher")) agent = "CIPHER";

    try {
      const response = await this.ryzansteinClient.chat(userInput, agent);

      // Stream the response
      stream.markdown(response.response);

      if (response.traceId) {
        stream.markdown(`\n\n_(Trace: ${response.traceId})_`);
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Unknown error";
      stream.markdown(`Error: ${errorMsg}`);
    }
  }
}
