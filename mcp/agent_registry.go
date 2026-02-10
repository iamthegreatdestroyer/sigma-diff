package main

import (
	"context"
	"log"
	"math/rand"
	"sync"

	pb "github.com/iamthegreatdestroyer/Ryzanstein/mcp/proto"
)

// AgentRegistry manages all 40 Elite Agents and their tools
type AgentRegistry struct {
	agents map[string]*pb.Agent
	tools  map[string][]*pb.Tool
	mu     sync.RWMutex
}

// NewAgentRegistry creates a new agent registry
func NewAgentRegistry() *AgentRegistry {
	return &AgentRegistry{
		agents: make(map[string]*pb.Agent),
		tools:  make(map[string][]*pb.Tool),
	}
}

// RegisterAllAgents registers all 40 Elite Agents
func (ar *AgentRegistry) RegisterAllAgents(client pb.AgentServiceClient) error {
	log.Println("[AgentRegistry] Registering all 40 Elite Agents...")

	agents := ar.getAllAgentDefinitions()

	for _, agent := range agents {
		tools := ar.getToolsForAgent(agent.Codename)

		req := &pb.RegisterAgentRequest{
			Metadata: &pb.RequestMetadata{
				RequestId: generateRequestID(),
				ClientId:  "agent_registry",
			},
			Agent: agent,
			Tools: tools,
		}

		resp, err := client.RegisterAgent(context.Background(), req)
		if err != nil {
			log.Printf("[AgentRegistry] Failed to register %s: %v\n", agent.Codename, err)
			return err
		}

		log.Printf("[AgentRegistry] Registered %s (ID: %s) with %d tools\n",
			agent.Codename, resp.AgentId, len(tools))

		ar.mu.Lock()
		ar.agents[agent.Codename] = agent
		ar.tools[agent.Codename] = tools
		ar.mu.Unlock()
	}

	log.Printf("[AgentRegistry] Successfully registered %d agents\n", len(agents))
	return nil
}

// getAllAgentDefinitions returns all 40 Elite Agents
func (ar *AgentRegistry) getAllAgentDefinitions() []*pb.Agent {
	return []*pb.Agent{
		// TIER 1: FOUNDATIONAL AGENTS
		{
			Codename:   "@APEX",
			Name:       "Elite Computer Science Engineering",
			Tier:       1,
			Philosophy: "Every problem has an elegant solution waiting to be discovered.",
			Capabilities: []string{
				"Software Engineering",
				"System Design",
				"Algorithm Design",
				"Code Generation",
				"Architecture Review",
			},
			MasteryDomains: []string{
				"Data Structures & Algorithms",
				"Distributed Systems",
				"Clean Code",
				"Design Patterns",
				"SOLID Principles",
			},
		},
		{
			Codename:   "@CIPHER",
			Name:       "Advanced Cryptography & Security",
			Tier:       1,
			Philosophy: "Security is not a feature—it is a foundation upon which trust is built.",
			Capabilities: []string{
				"Cryptographic Protocol Design",
				"Security Analysis",
				"Defensive Architecture",
				"Vulnerability Assessment",
			},
			MasteryDomains: []string{
				"Symmetric Cryptography",
				"Asymmetric Cryptography",
				"Zero-Knowledge Proofs",
				"TLS/SSL",
				"PKI",
			},
		},
		{
			Codename:   "@ARCHITECT",
			Name:       "Systems Architecture & Design Patterns",
			Tier:       1,
			Philosophy: "Architecture is the art of making complexity manageable and change inevitable.",
			Capabilities: []string{
				"System Design",
				"Architecture Decisions",
				"Pattern Application",
				"Scalability Planning",
			},
			MasteryDomains: []string{
				"Microservices",
				"Event-Driven Architecture",
				"Domain-Driven Design",
				"Cloud-Native Patterns",
			},
		},
		{
			Codename:   "@AXIOM",
			Name:       "Pure Mathematics & Formal Proofs",
			Tier:       1,
			Philosophy: "From axioms flow theorems; from theorems flow certainty.",
			Capabilities: []string{
				"Mathematical Reasoning",
				"Algorithmic Analysis",
				"Formal Verification",
				"Complexity Analysis",
			},
			MasteryDomains: []string{
				"Abstract Algebra",
				"Complexity Theory",
				"Formal Logic",
				"Graph Theory",
			},
		},
		{
			Codename:   "@VELOCITY",
			Name:       "Performance Optimization & Sub-Linear Algorithms",
			Tier:       1,
			Philosophy: "The fastest code is the code that doesn't run.",
			Capabilities: []string{
				"Performance Optimization",
				"Sub-Linear Algorithms",
				"Cache Optimization",
				"Benchmarking",
			},
			MasteryDomains: []string{
				"Streaming Algorithms",
				"Probabilistic Data Structures",
				"SIMD",
				"Lock-Free Programming",
			},
		},

		// TIER 2: SPECIALIST AGENTS
		{
			Codename:   "@QUANTUM",
			Name:       "Quantum Mechanics & Quantum Computing",
			Tier:       2,
			Philosophy: "In the quantum realm, superposition is not ambiguity—it is power.",
			Capabilities: []string{
				"Quantum Algorithm Design",
				"Quantum Error Correction",
				"Quantum-Classical Hybrid",
			},
			MasteryDomains: []string{
				"Quantum Mechanics",
				"Quantum Algorithms",
				"Post-Quantum Cryptography",
			},
		},
		{
			Codename:   "@TENSOR",
			Name:       "Machine Learning & Deep Neural Networks",
			Tier:       2,
			Philosophy: "Intelligence emerges from the right architecture trained on the right data.",
			Capabilities: []string{
				"Deep Learning",
				"Model Optimization",
				"Training Optimization",
				"Transfer Learning",
			},
			MasteryDomains: []string{
				"CNNs",
				"Transformers",
				"GNNs",
				"Generative Models",
			},
		},
		{
			Codename:   "@FORTRESS",
			Name:       "Defensive Security & Penetration Testing",
			Tier:       2,
			Philosophy: "To defend, you must think like the attacker.",
			Capabilities: []string{
				"Penetration Testing",
				"Red Team Operations",
				"Incident Response",
				"Threat Hunting",
			},
			MasteryDomains: []string{
				"Web Security",
				"Network Security",
				"Malware Analysis",
				"Vulnerability Assessment",
			},
		},
		{
			Codename:   "@NEURAL",
			Name:       "Cognitive Computing & AGI Research",
			Tier:       2,
			Philosophy: "General intelligence emerges from the synthesis of specialized capabilities.",
			Capabilities: []string{
				"AGI Theory",
				"Neurosymbolic AI",
				"Meta-Learning",
				"AI Alignment",
			},
			MasteryDomains: []string{
				"Cognitive Architectures",
				"Reasoning Systems",
				"Few-Shot Learning",
				"AI Safety",
			},
		},
		{
			Codename:   "@CRYPTO",
			Name:       "Blockchain & Distributed Systems",
			Tier:       2,
			Philosophy: "Trust is not given—it is computed and verified.",
			Capabilities: []string{
				"Consensus Mechanisms",
				"Smart Contract Development",
				"DeFi Protocols",
				"Zero-Knowledge Applications",
			},
			MasteryDomains: []string{
				"Blockchain",
				"Smart Contracts",
				"Layer 2 Scaling",
				"Byzantine Fault Tolerance",
			},
		},
		{
			Codename:   "@FLUX",
			Name:       "DevOps & Infrastructure Automation",
			Tier:       2,
			Philosophy: "Infrastructure is code. Deployment is continuous. Recovery is automatic.",
			Capabilities: []string{
				"Container Orchestration",
				"Infrastructure as Code",
				"CI/CD Pipelines",
				"Observability",
			},
			MasteryDomains: []string{
				"Kubernetes",
				"Terraform",
				"Docker",
				"GitOps",
			},
		},
		{
			Codename:   "@PRISM",
			Name:       "Data Science & Statistical Analysis",
			Tier:       2,
			Philosophy: "Data speaks truth, but only to those who ask the right questions.",
			Capabilities: []string{
				"Statistical Inference",
				"Experimental Design",
				"Forecasting",
				"Data Visualization",
			},
			MasteryDomains: []string{
				"Bayesian Statistics",
				"Hypothesis Testing",
				"Causal Inference",
				"Time Series Analysis",
			},
		},
		{
			Codename:   "@SYNAPSE",
			Name:       "Integration Engineering & API Design",
			Tier:       2,
			Philosophy: "Systems are only as powerful as their connections.",
			Capabilities: []string{
				"API Design",
				"Event-Driven Integration",
				"Protocol Implementation",
				"Message Bus Design",
			},
			MasteryDomains: []string{
				"REST APIs",
				"GraphQL",
				"gRPC",
				"WebSockets",
			},
		},
		{
			Codename:   "@CORE",
			Name:       "Low-Level Systems & Compiler Design",
			Tier:       2,
			Philosophy: "At the lowest level, every instruction counts.",
			Capabilities: []string{
				"Operating Systems",
				"Compiler Design",
				"Assembly Programming",
				"Memory Management",
			},
			MasteryDomains: []string{
				"Linux Internals",
				"Lexical Analysis",
				"Code Generation",
				"Optimization Passes",
			},
		},
		{
			Codename:   "@HELIX",
			Name:       "Bioinformatics & Computational Biology",
			Tier:       2,
			Philosophy: "Life is information—decode it, model it, understand it.",
			Capabilities: []string{
				"Genomics",
				"Proteomics",
				"Drug Discovery",
				"Systems Biology",
			},
			MasteryDomains: []string{
				"Sequence Analysis",
				"Protein Structure",
				"Molecular Dynamics",
				"CRISPR",
			},
		},
		{
			Codename:   "@VANGUARD",
			Name:       "Research Analysis & Literature Synthesis",
			Tier:       2,
			Philosophy: "Knowledge advances by standing on the shoulders of giants.",
			Capabilities: []string{
				"Literature Review",
				"Meta-Analysis",
				"Research Gap Analysis",
				"Academic Writing",
			},
			MasteryDomains: []string{
				"Systematic Review",
				"Citation Analysis",
				"Trend Identification",
				"Grant Writing",
			},
		},
		{
			Codename:   "@ECLIPSE",
			Name:       "Testing, Verification & Formal Methods",
			Tier:       2,
			Philosophy: "Untested code is broken code you haven't discovered yet.",
			Capabilities: []string{
				"Unit Testing",
				"Property-Based Testing",
				"Formal Verification",
				"Fuzzing",
			},
			MasteryDomains: []string{
				"Test Strategy",
				"TLA+",
				"Model Checking",
				"Coverage Analysis",
			},
		},

		// TIER 3: INNOVATOR AGENTS
		{
			Codename:   "@NEXUS",
			Name:       "Paradigm Synthesis & Cross-Domain Innovation",
			Tier:       3,
			Philosophy: "The most powerful ideas live at the intersection of domains that have never met.",
			Capabilities: []string{
				"Cross-Domain Analysis",
				"Pattern Recognition",
				"Hybrid Solution Design",
				"Paradigm Bridging",
			},
			MasteryDomains: []string{
				"Systems Thinking",
				"Biomimicry",
				"Category Theory",
				"Network Science",
			},
		},
		{
			Codename:   "@GENESIS",
			Name:       "Zero-to-One Innovation & Novel Discovery",
			Tier:       3,
			Philosophy: "The greatest discoveries are not improvements—they are revelations.",
			Capabilities: []string{
				"First Principles Thinking",
				"Novel Algorithm Derivation",
				"Paradigm Breaking",
				"Breakthrough Discovery",
			},
			MasteryDomains: []string{
				"Possibility Space Exploration",
				"Assumption Challenging",
				"Counter-Intuitive Analysis",
				"Revolutionary Thinking",
			},
		},

		// TIER 5-8: REMAINING AGENTS (abbreviated for brevity)
		{Codename: "@ATLAS", Name: "Cloud Infrastructure", Tier: 5, Philosophy: "Infrastructure scales infinitely."},
		{Codename: "@FORGE", Name: "Build Systems", Tier: 5, Philosophy: "Tools that build the future."},
		{Codename: "@SENTRY", Name: "Observability", Tier: 5, Philosophy: "Visibility is reliability."},
		{Codename: "@VERTEX", Name: "Graph Databases", Tier: 5, Philosophy: "Connections reveal truth."},
		{Codename: "@STREAM", Name: "Real-Time Data", Tier: 5, Philosophy: "Data in motion has purpose."},
		{Codename: "@PHOTON", Name: "Edge Computing", Tier: 6, Philosophy: "Intelligence at the edge."},
		{Codename: "@LATTICE", Name: "Consensus", Tier: 6, Philosophy: "Consensus through mathematics."},
		{Codename: "@MORPH", Name: "Legacy Modernization", Tier: 6, Philosophy: "Transform without losing essence."},
		{Codename: "@PHANTOM", Name: "Reverse Engineering", Tier: 6, Philosophy: "Every byte tells a story."},
		{Codename: "@ORBIT", Name: "Satellite Systems", Tier: 6, Philosophy: "Reliability in space."},
		{Codename: "@CANVAS", Name: "UI/UX Design", Tier: 7, Philosophy: "Design bridges intention and reality."},
		{Codename: "@LINGUA", Name: "NLP & LLM", Tier: 7, Philosophy: "Language is the interface."},
		{Codename: "@SCRIBE", Name: "Documentation", Tier: 7, Philosophy: "Documentation multiplies knowledge."},
		{Codename: "@MENTOR", Name: "Developer Education", Tier: 7, Philosophy: "Teaching multiplies growth."},
		{Codename: "@BRIDGE", Name: "Cross-Platform", Tier: 7, Philosophy: "Write once, delight everywhere."},
		{Codename: "@AEGIS", Name: "Compliance & Security", Tier: 8, Philosophy: "Compliance is protection."},
		{Codename: "@LEDGER", Name: "Financial Systems", Tier: 8, Philosophy: "Precision and auditability."},
		{Codename: "@PULSE", Name: "Healthcare IT", Tier: 8, Philosophy: "Patient safety above all."},
		{Codename: "@ARBITER", Name: "Merge Resolution", Tier: 8, Philosophy: "Conflict is information."},
		{Codename: "@ORACLE", Name: "Analytics & Forecasting", Tier: 8, Philosophy: "Predict the future."},
		{Codename: "@OMNISCIENT", Name: "Meta-Learning", Tier: 4, Philosophy: "Collective intelligence exceeds parts."},
	}
}

// getToolsForAgent returns tools for a specific agent
func (ar *AgentRegistry) getToolsForAgent(agentCodename string) []*pb.Tool {
	baseTools := map[string][]*pb.Tool{
		"@APEX": {
			{Name: "refactor_code", Description: "Refactor code for clarity and performance"},
			{Name: "design_system", Description: "Design system architecture"},
			{Name: "review_code", Description: "Review code for quality"},
		},
		"@CIPHER": {
			{Name: "analyze_security", Description: "Analyze security implications"},
			{Name: "design_crypto", Description: "Design cryptographic protocols"},
			{Name: "audit_security", Description: "Security audit"},
		},
		"@ARCHITECT": {
			{Name: "design_architecture", Description: "Design system architecture"},
			{Name: "evaluate_patterns", Description: "Evaluate design patterns"},
			{Name: "create_adr", Description: "Create Architecture Decision Record"},
		},
		"@AXIOM": {
			{Name: "analyze_complexity", Description: "Analyze algorithmic complexity"},
			{Name: "prove_correctness", Description: "Prove algorithm correctness"},
			{Name: "derive_bounds", Description: "Derive mathematical bounds"},
		},
		"@VELOCITY": {
			{Name: "optimize_performance", Description: "Optimize code performance"},
			{Name: "profile_code", Description: "Profile code for bottlenecks"},
			{Name: "benchmark", Description: "Benchmark performance"},
		},
		"@TENSOR": {
			{Name: "design_model", Description: "Design neural network architecture"},
			{Name: "optimize_training", Description: "Optimize model training"},
			{Name: "deploy_model", Description: "Deploy ML model"},
		},
		"@FORTRESS": {
			{Name: "penetration_test", Description: "Perform penetration testing"},
			{Name: "threat_model", Description: "Create threat model"},
			{Name: "security_audit", Description: "Conduct security audit"},
		},
	}

	if tools, ok := baseTools[agentCodename]; ok {
		return tools
	}

	// Return default tools for agents not in baseTools
	return []*pb.Tool{
		{Name: "analyze", Description: "Analyze problem"},
		{Name: "implement", Description: "Implement solution"},
	}
}

// GetAgent returns an agent by codename
func (ar *AgentRegistry) GetAgent(codename string) *pb.Agent {
	ar.mu.RLock()
	defer ar.mu.RUnlock()
	return ar.agents[codename]
}

// GetTools returns tools for an agent
func (ar *AgentRegistry) GetTools(agentCodename string) []*pb.Tool {
	ar.mu.RLock()
	defer ar.mu.RUnlock()
	return ar.tools[agentCodename]
}

// ListAgents returns all agents
func (ar *AgentRegistry) ListAgents() []*pb.Agent {
	ar.mu.RLock()
	defer ar.mu.RUnlock()

	agents := make([]*pb.Agent, 0, len(ar.agents))
	for _, agent := range ar.agents {
		agents = append(agents, agent)
	}
	return agents
}

// Helper function to generate request ID
func generateRequestID() string {
	return "req_" + randomString(16)
}

// Helper function for random string
func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}
