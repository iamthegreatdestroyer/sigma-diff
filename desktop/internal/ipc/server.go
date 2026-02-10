package ipc

import (
	"fmt"
	"log"
	"net"
	"sync"
)

// Server handles IPC communication between desktop and VS Code
type Server struct {
	listener  net.Listener
	clients   map[string]net.Conn
	mu        sync.RWMutex
	isRunning bool
}

// NewServer creates a new IPC server
func NewServer() *Server {
	return &Server{
		clients: make(map[string]net.Conn),
	}
}

// Start starts the IPC server
func (s *Server) Start() error {
	log.Println("[IPC] Starting server on localhost:9001")

	listener, err := net.Listen("tcp", "localhost:9001")
	if err != nil {
		return fmt.Errorf("failed to start IPC server: %w", err)
	}

	s.listener = listener
	s.isRunning = true

	// Accept connections
	go s.acceptConnections()

	return nil
}

// acceptConnections accepts incoming client connections
func (s *Server) acceptConnections() {
	for s.isRunning {
		conn, err := s.listener.Accept()
		if err != nil {
			if s.isRunning {
				log.Printf("[IPC] Accept error: %v\n", err)
			}
			continue
		}

		clientID := conn.RemoteAddr().String()
		s.mu.Lock()
		s.clients[clientID] = conn
		s.mu.Unlock()

		log.Printf("[IPC] Client connected: %s\n", clientID)

		// Handle client in goroutine
		go s.handleClient(clientID, conn)
	}
}

// handleClient handles a client connection
func (s *Server) handleClient(clientID string, conn net.Conn) {
	defer func() {
		conn.Close()
		s.mu.Lock()
		delete(s.clients, clientID)
		s.mu.Unlock()
		log.Printf("[IPC] Client disconnected: %s\n", clientID)
	}()

	// Read from client
	buf := make([]byte, 1024)
	for {
		n, err := conn.Read(buf)
		if err != nil {
			return
		}

		message := string(buf[:n])
		log.Printf("[IPC] Received from %s: %s\n", clientID, message)

		// Echo response (will be replaced with actual logic)
		response := fmt.Sprintf("ACK: %s", message)
		if _, err := conn.Write([]byte(response)); err != nil {
			return
		}
	}
}

// Broadcast sends message to all connected clients
func (s *Server) Broadcast(message string) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for clientID, conn := range s.clients {
		if _, err := conn.Write([]byte(message)); err != nil {
			log.Printf("[IPC] Error sending to %s: %v\n", clientID, err)
		}
	}
}

// SendToClient sends message to specific client
func (s *Server) SendToClient(clientID string, message string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	conn, ok := s.clients[clientID]
	if !ok {
		return fmt.Errorf("client not found: %s", clientID)
	}

	_, err := conn.Write([]byte(message))
	return err
}

// Close closes the IPC server
func (s *Server) Close() error {
	log.Println("[IPC] Closing server")
	s.isRunning = false

	// Close all client connections
	s.mu.Lock()
	for _, conn := range s.clients {
		conn.Close()
	}
	s.clients = make(map[string]net.Conn)
	s.mu.Unlock()

	// Close listener
	if s.listener != nil {
		return s.listener.Close()
	}

	return nil
}

// GetConnectedClients returns count of connected clients
func (s *Server) GetConnectedClients() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.clients)
}
