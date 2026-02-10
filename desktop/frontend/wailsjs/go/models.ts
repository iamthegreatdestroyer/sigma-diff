export namespace main {
	
	export class ConfigData {
	    theme: string;
	    defaultModel: string;
	    defaultAgent: string;
	    ryzansteinApiUrl: string;
	    mcpServerUrl: string;
	    autoLoadLastModel: boolean;
	    enableSystemTray: boolean;
	    minimizeToTray: boolean;
	
	    static createFrom(source: any = {}) {
	        return new ConfigData(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.theme = source["theme"];
	        this.defaultModel = source["defaultModel"];
	        this.defaultAgent = source["defaultAgent"];
	        this.ryzansteinApiUrl = source["ryzansteinApiUrl"];
	        this.mcpServerUrl = source["mcpServerUrl"];
	        this.autoLoadLastModel = source["autoLoadLastModel"];
	        this.enableSystemTray = source["enableSystemTray"];
	        this.minimizeToTray = source["minimizeToTray"];
	    }
	}
	export class Message {
	    id: string;
	    role: string;
	    content: string;
	    timestamp: number;
	    metadata?: Record<string, any>;
	
	    static createFrom(source: any = {}) {
	        return new Message(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.role = source["role"];
	        this.content = source["content"];
	        this.timestamp = source["timestamp"];
	        this.metadata = source["metadata"];
	    }
	}
	export class ModelInfo {
	    id: string;
	    name: string;
	    size: string;
	    contextLength: number;
	    loaded: boolean;
	    status: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.name = source["name"];
	        this.size = source["size"];
	        this.contextLength = source["contextLength"];
	        this.loaded = source["loaded"];
	        this.status = source["status"];
	    }
	}

}

