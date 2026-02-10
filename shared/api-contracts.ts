/**
 * API Interface Contracts
 * Defines the contract between Desktop App, VS Code Extension, and Backend Services
 */

// ============================================================================
// Ryzanstein LLM API Contracts
// ============================================================================

export interface InferenceRequest {
  prompt: string;
  model: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stream?: boolean;
  metadata?: Record<string, any>;
}

export interface InferenceResponse {
  id: string;
  model: string;
  completion: string;
  finishReason: "stop" | "length" | "error";
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  metadata?: Record<string, any>;
}

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  contextLength: number;
  loaded: boolean;
  status: "ready" | "loading" | "unloading" | "error";
  capabilities?: string[];
}

export interface RyzansteinAPI {
  // Inference
  infer(request: InferenceRequest): Promise<InferenceResponse>;
  inferStream(request: InferenceRequest): AsyncIterable<string>;

  // Models
  listModels(): Promise<ModelInfo[]>;
  loadModel(modelId: string): Promise<void>;
  unloadModel(modelId: string): Promise<void>;
  getModelInfo(modelId: string): Promise<ModelInfo>;

  // Health
  health(): Promise<{ status: string; version: string }>;
}

// ============================================================================
// MCP Server Contracts (gRPC)
// ============================================================================

export interface MCPMessage {
  id: string;
  type: "request" | "response" | "event";
  agent: string;
  tool: string;
  payload: any;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface MCPRequest {
  agent: string;
  tool: string;
  parameters: Record<string, any>;
  context?: {
    modelId: string;
    sessionId: string;
    userId?: string;
  };
}

export interface MCPResponse {
  requestId: string;
  success: boolean;
  result?: any;
  error?: {
    code: string;
    message: string;
  };
  executionTime: number;
}

export interface AgentCapability {
  codename: string;
  name: string;
  tier: number;
  philosophy: string;
  capabilities: string[];
  masteryDomains: string[];
  tools: ToolInfo[];
}

export interface ToolInfo {
  name: string;
  description: string;
  category: string;
  parameters: ParameterSchema[];
  returns?: {
    type: string;
    description: string;
  };
}

export interface ParameterSchema {
  name: string;
  type: "string" | "number" | "boolean" | "object" | "array";
  description: string;
  required: boolean;
  default?: any;
}

export interface MCPAPI {
  // Agent Management
  listAgents(): Promise<AgentCapability[]>;
  getAgent(codename: string): Promise<AgentCapability>;
  invokeAgent(request: MCPRequest): Promise<MCPResponse>;
  invokeAgentStream(request: MCPRequest): AsyncIterable<MCPResponse>;

  // Memory
  storeExperience(experience: ExperienceTuple): Promise<void>;
  retrieveExperience(query: string): Promise<ExperienceTuple[]>;
  getMemoryStats(): Promise<MemoryStats>;

  // Health
  health(): Promise<{ status: string; agents: number }>;
}

export interface ExperienceTuple {
  id: string;
  input: string;
  output: string;
  agent: string;
  timestamp: number;
  fitness?: number;
  embedding?: number[];
}

export interface MemoryStats {
  totalExperiences: number;
  breakthroughs: number;
  avgFitness: number;
  lastUpdate: number;
}

// ============================================================================
// Continue.dev Integration Contracts
// ============================================================================

export interface ContinueProviderRequest {
  code: string;
  selection?: {
    start: number;
    end: number;
  };
  context?: {
    file: string;
    language: string;
    line: number;
  };
  action: "explain" | "refactor" | "generate" | "test" | "debug";
}

export interface ContinueProviderResponse {
  text: string;
  code?: string;
  description?: string;
  suggestions?: string[];
}

export interface ContinueAPI {
  processRequest(
    request: ContinueProviderRequest
  ): Promise<ContinueProviderResponse>;
  streamResponse(request: ContinueProviderRequest): AsyncIterable<string>;
}

// ============================================================================
// Chat Service Contracts
// ============================================================================

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  metadata?: {
    agent?: string;
    model?: string;
    tokens?: number;
  };
}

export interface ChatSession {
  id: string;
  title: string;
  model: string;
  agent: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

export interface ChatAPI {
  sendMessage(sessionId: string, message: string): Promise<ChatMessage>;
  getSession(sessionId: string): Promise<ChatSession>;
  listSessions(): Promise<ChatSession[]>;
  createSession(
    title: string,
    model: string,
    agent: string
  ): Promise<ChatSession>;
  deleteSession(sessionId: string): Promise<void>;
  clearHistory(sessionId: string): Promise<void>;
}

// ============================================================================
// Configuration Contracts
// ============================================================================

export interface AppConfig {
  theme: "light" | "dark" | "auto";
  defaultModel: string;
  defaultAgent: string;
  ryzansteinApiUrl: string;
  mcpServerUrl: string;
  autoLoadLastModel: boolean;
  enableSystemTray: boolean;
  minimizeToTray: boolean;
  // VS Code Extension specific
  enableInlineChat?: boolean;
  enableCodeLens?: boolean;
  autoConnect?: boolean;
}

export interface ConfigAPI {
  getConfig(): Promise<AppConfig>;
  saveConfig(config: Partial<AppConfig>): Promise<void>;
  resetConfig(): Promise<void>;
}

// ============================================================================
// Error Handling
// ============================================================================

export class RyzansteinError extends Error {
  constructor(
    public code: string,
    message: string,
    public statusCode?: number,
    public details?: any
  ) {
    super(message);
    this.name = "RyzansteinError";
  }
}

export const ErrorCodes = {
  // Connection errors
  CONNECTION_FAILED: "CONNECTION_FAILED",
  TIMEOUT: "TIMEOUT",
  SERVER_UNAVAILABLE: "SERVER_UNAVAILABLE",

  // Model errors
  MODEL_NOT_FOUND: "MODEL_NOT_FOUND",
  MODEL_LOAD_FAILED: "MODEL_LOAD_FAILED",
  INFERENCE_FAILED: "INFERENCE_FAILED",

  // Agent errors
  AGENT_NOT_FOUND: "AGENT_NOT_FOUND",
  TOOL_NOT_FOUND: "TOOL_NOT_FOUND",
  AGENT_INVOCATION_FAILED: "AGENT_INVOCATION_FAILED",

  // Configuration errors
  INVALID_CONFIG: "INVALID_CONFIG",
  CONFIG_SAVE_FAILED: "CONFIG_SAVE_FAILED",

  // Generic errors
  INTERNAL_ERROR: "INTERNAL_ERROR",
  INVALID_REQUEST: "INVALID_REQUEST",
};
