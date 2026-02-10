package agents

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Service handles agent operations
type Service struct {
	agents map[string]*AgentInfo
	mu     sync.RWMutex
}

// AgentInfo represents agent information
type AgentInfo struct {
	Codename       string
	Name           string
	Tier           int
	Philosophy     string
	Capabilities   []string
	MasteryDomains []string
	Tools          map[string]*Tool
}

// Tool represents an agent tool
type Tool struct {
	Name        string
	Description string
	InputSchema map[string]interface{}
}

// NewService creates a new agents service
func NewService() *Service {
	s := &Service{
		agents: make(map[string]*AgentInfo),
	}

	// Register core agents
	s.registerCoreAgents()

	return s
}

// registerCoreAgents registers the Elite Agent Collective from .github/agents
func (s *Service) registerCoreAgents() {
	log.Println("[AgentsService] Loading Elite Agents from .github/agents")

	// Get the project root directory (assuming we're in desktop/cmd/ryzanstein)
	wd, err := os.Getwd()
	if err != nil {
		log.Printf("[AgentsService] Error getting working directory: %v", err)
		return
	}

	// Navigate to project root (.github is at project root)
	projectRoot := filepath.Join(wd, "..", "..")
	agentsDir := filepath.Join(projectRoot, ".github", "agents")

	// Check if agents directory exists
	if _, err := os.Stat(agentsDir); os.IsNotExist(err) {
		log.Printf("[AgentsService] Agents directory not found: %s", agentsDir)
		// Fall back to hardcoded agents
		s.registerFallbackAgents()
		return
	}

	// Load agents from files
	files, err := filepath.Glob(filepath.Join(agentsDir, "*.agent.md"))
	if err != nil {
		log.Printf("[AgentsService] Error reading agents directory: %v", err)
		s.registerFallbackAgents()
		return
	}

	loadedCount := 0
	for _, file := range files {
		if agent, err := s.parseAgentFile(file); err == nil {
			s.agents[agent.Codename] = agent
			loadedCount++
		} else {
			log.Printf("[AgentsService] Error parsing agent file %s: %v", file, err)
		}
	}

	log.Printf("[AgentsService] Loaded %d agents from files", loadedCount)

	// If no agents were loaded, fall back to hardcoded
	if loadedCount == 0 {
		log.Println("[AgentsService] No agents loaded from files, using fallback")
		s.registerFallbackAgents()
	}
}

// parseAgentFile parses an agent file and returns AgentInfo
func (s *Service) parseAgentFile(filePath string) (*AgentInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	inFrontmatter := false
	frontmatter := make(map[string]string)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "---" {
			if !inFrontmatter {
				inFrontmatter = true
			} else {
				break // End of frontmatter
			}
			continue
		}

		if inFrontmatter && strings.Contains(line, ":") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				value := strings.TrimSpace(parts[1])
				// Remove quotes if present
				value = strings.Trim(value, `"'`)
				frontmatter[key] = value
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	// Extract required fields
	codename, hasCodename := frontmatter["codename"]
	name, hasName := frontmatter["name"]
	tierStr, hasTier := frontmatter["tier"]

	if !hasCodename || !hasName || !hasTier {
		return nil, fmt.Errorf("missing required fields in %s", filePath)
	}

	tier := 1
	if tierStr == "2" {
		tier = 2
	} else if tierStr == "3" {
		tier = 3
	} else if tierStr == "4" {
		tier = 4
	}

	// Read the rest of the file for philosophy
	file.Seek(0, 0)
	scanner = bufio.NewScanner(file)
	philosophy := ""
	inPhilosophy := false

	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "**Philosophy:**") {
			inPhilosophy = true
			continue
		}
		if inPhilosophy && strings.TrimSpace(line) == "" {
			break
		}
		if inPhilosophy {
			// Extract philosophy text
			if strings.Contains(line, "_\"") && strings.Contains(line, "\"_") {
				start := strings.Index(line, "_\"") + 2
				end := strings.LastIndex(line, "\"_")
				if start < end {
					philosophy = line[start:end]
				}
			}
		}
	}

	return &AgentInfo{
		Codename:       "@" + codename,
		Name:           name,
		Tier:           tier,
		Philosophy:     philosophy,
		Capabilities:   []string{}, // Will be populated from file content if needed
		MasteryDomains: []string{}, // Will be populated from file content if needed
		Tools:          make(map[string]*Tool),
	}, nil
}

// registerFallbackAgents provides hardcoded agents if file loading fails
func (s *Service) registerFallbackAgents() {
	log.Println("[AgentsService] Using fallback agent registration")

	fallbackAgents := []struct {
		codename   string
		name       string
		tier       int
		philosophy string
	}{
		{"@APEX", "Elite Computer Science Engineering", 1, "Every problem has an elegant solution"},
		{"@CIPHER", "Advanced Cryptography & Security", 1, "Security is not a feature—it is a foundation"},
		{"@ARCHITECT", "Systems Architecture & Design Patterns", 1, "Architecture is making complexity manageable"},
		{"@AXIOM", "Pure Mathematics & Formal Proofs", 1, "From axioms flow theorems"},
		{"@VELOCITY", "Performance Optimization & Sub-Linear Algorithms", 1, "The fastest code is the code that doesn't run"},
		{"@QUANTUM", "Quantum Mechanics & Quantum Computing", 2, "In the quantum realm, superposition is not ambiguity—it is power"},
		{"@TENSOR", "Machine Learning & Deep Neural Networks", 2, "Intelligence emerges from the right architecture trained on the right data"},
		{"@FORTRESS", "Defensive Security & Penetration Testing", 2, "To defend, you must think like the attacker"},
		{"@NEURAL", "Cognitive Computing & AGI Research", 2, "General intelligence emerges from the synthesis of specialized capabilities"},
		{"@CRYPTO", "Blockchain & Distributed Systems", 2, "Trust is not given—it is computed and verified"},
		{"@FLUX", "DevOps & Infrastructure Automation", 2, "Infrastructure is code. Deployment is continuous. Recovery is automatic"},
		{"@PRISM", "Data Science & Statistical Analysis", 2, "Data speaks truth, but only to those who ask the right questions"},
		{"@SYNAPSE", "Integration Engineering & API Design", 2, "Systems are only as powerful as their connections"},
		{"@CORE", "Low-Level Systems & Compiler Design", 2, "At the lowest level, every instruction counts"},
		{"@HELIX", "Bioinformatics & Computational Biology", 2, "Life is information—decode it, model it, understand it"},
		{"@VANGUARD", "Research Analysis & Literature Synthesis", 2, "Knowledge advances by standing on the shoulders of giants"},
		{"@ECLIPSE", "Testing, Verification & Formal Methods", 2, "Untested code is broken code you haven't discovered yet"},
	}

	for _, agent := range fallbackAgents {
		s.agents[agent.codename] = &AgentInfo{
			Codename:       agent.codename,
			Name:           agent.name,
			Tier:           agent.tier,
			Philosophy:     agent.philosophy,
			Capabilities:   []string{},
			MasteryDomains: []string{},
			Tools:          make(map[string]*Tool),
		}
	}

	log.Printf("[AgentsService] Registered %d fallback agents", len(s.agents))
}

// ListAgents returns available agent names
func (s *Service) ListAgents() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	agents := make([]string, 0, len(s.agents))
	for _, agent := range s.agents {
		agents = append(agents, agent.Codename)
	}

	return agents
}

// GetAgent returns agent details
func (s *Service) GetAgent(codename string) *AgentInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.agents[codename]
}

// InvokeTool invokes an agent tool
func (s *Service) InvokeTool(ctx context.Context, agentCodename string, toolName string, params map[string]interface{}) (interface{}, error) {
	log.Printf("[AgentsService] Invoking %s.%s\n", agentCodename, toolName)

	s.mu.RLock()
	agent, ok := s.agents[agentCodename]
	s.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("agent not found: %s", agentCodename)
	}

	_, ok = agent.Tools[toolName]
	if !ok {
		return nil, fmt.Errorf("tool not found: %s", toolName)
	}

	// Simulate tool invocation (will be replaced with MCP calls)
	result := map[string]interface{}{
		"agent":  agentCodename,
		"tool":   toolName,
		"status": "success",
		"output": fmt.Sprintf("Tool %s executed by %s", toolName, agent.Name),
	}

	return result, nil
}
