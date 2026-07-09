import axios, { AxiosInstance } from "axios";

export interface RyzansteinAgent {
  id: string;
  name: string;
}

export interface RyzansteinModel {
  id: string;
  name: string;
}

export interface RyzansteinChatResponse {
  response: string;
}

/**
 * HTTP client for the Ryzanstein API server (see ryzanstein.ryzansteinApiUrl
 * setting, default http://localhost:8000). Mirrors the minimal surface used
 * by CommandHandler, ModelTreeProvider and ChatWebviewProvider.
 *
 * This module was missing entirely from the repo (not a broken import path -
 * there was no prior implementation to restore). Reconstructed from the call
 * sites that depend on it.
 */
export class RyzansteinClient {
  private http: AxiosInstance;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.http = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
    });
  }

  async listAgents(): Promise<RyzansteinAgent[]> {
    try {
      const { data } = await this.http.get<RyzansteinAgent[]>("/agents");
      return data;
    } catch (error) {
      throw new Error(
        `Failed to list agents: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  async listModels(): Promise<RyzansteinModel[]> {
    try {
      const { data } = await this.http.get<RyzansteinModel[]>("/models");
      return data;
    } catch (error) {
      throw new Error(
        `Failed to list models: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  async loadModel(modelId: string): Promise<void> {
    try {
      await this.http.post("/models/load", { modelId });
    } catch (error) {
      throw new Error(
        `Failed to load model: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  async generateCode(prompt: string): Promise<string> {
    try {
      const { data } = await this.http.post<{ code: string }>("/generate", {
        prompt,
      });
      return data.code;
    } catch (error) {
      throw new Error(
        `Failed to generate code: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  async chat(text: string, agentId: string): Promise<RyzansteinChatResponse> {
    try {
      const { data } = await this.http.post<RyzansteinChatResponse>("/chat", {
        text,
        agentId,
      });
      return data;
    } catch (error) {
      throw new Error(
        `Failed to chat: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }
}
