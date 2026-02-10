import axios from 'axios';

export class RyzansteinAPI {
    private client;

    constructor(baseURL: string) {
        this.client = axios.create({
            baseURL,
            timeout: 30000,
            headers: { 'Content-Type': 'application/json' },
        });
    }

    async chat(message: string, agentId: string): Promise<{ response: string }> {
        const response = await this.client.post('/chat', {
            message,
            agent_id: agentId,
        });
        return response.data;
    }

    async listAgents(): Promise<any[]> {
        const response = await this.client.get('/agents');
        return response.data;
    }

    async listModels(): Promise<any[]> {
        const response = await this.client.get('/models');
        return response.data;
    }

    async generateCode(prompt: string): Promise<string> {
        const response = await this.client.post('/generate', {
            prompt,
            type: 'code',
        });
        return response.data.generated;
    }
}
