import axios, { AxiosInstance } from "axios";

export interface ChatResponse {
  response: string;
  traceId?: string;
}

export interface Model {
  id: string;
  name: string;
}

export interface Agent {
  id: string;
  name: string;
  type: string;
}

export class RyzansteinClient {
  private client: AxiosInstance;

  constructor(baseURL: string = "http://localhost:8000") {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: { "Content-Type": "application/json" },
    });
  }

  async chat(
    message: string,
    agentId: string = "default"
  ): Promise<ChatResponse> {
    try {
      const response = await this.client.post("/chat", {
        message,
        agent_id: agentId,
      });
      return response.data;
    } catch (error) {
      throw new Error(
        `Chat request failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  async listModels(): Promise<Model[]> {
    try {
      const response = await this.client.get("/models");
      return response.data;
    } catch (error) {
      console.error("Failed to list models:", error);
      return [];
    }
  }

  async listAgents(): Promise<Agent[]> {
    try {
      const response = await this.client.get("/agents");
      return response.data;
    } catch (error) {
      console.error("Failed to list agents:", error);
      return [];
    }
  }

  async loadModel(modelId: string): Promise<void> {
    try {
      await this.client.post("/models/load", { model_id: modelId });
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
      const response = await this.client.post("/generate", {
        prompt,
        type: "code",
      });
      return response.data.generated;
    } catch (error) {
      throw new Error(
        `Code generation failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }
}
