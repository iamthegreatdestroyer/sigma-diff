// Package semantic provides embedding-based semantic code similarity detection.
// Connects sigma-diff to Ryzanstein /v1/embeddings for detecting refactored code
// that is structurally different but semantically equivalent.
package semantic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"time"
)

type Client struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

func NewClient() *Client {
	baseURL := os.Getenv("RYZANSTEIN_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}
	return &Client{
		baseURL:    baseURL,
		model:      "ryzanstein-bitnet-7b",
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

type embReq struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embResp struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (c *Client) Embed(text string) ([]float32, error) {
	payload, _ := json.Marshal(embReq{Input: text, Model: c.model})
	resp, err := c.httpClient.Post(c.baseURL+"/v1/embeddings", "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("embeddings API returned %d", resp.StatusCode)
	}
	var result embResp
	json.NewDecoder(resp.Body).Decode(&result)
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("empty embedding")
	}
	return result.Data[0].Embedding, nil
}

func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

type SemanticMatch struct {
	OldName    string
	NewName    string
	Similarity float64
}

func FindSemanticMatches(client *Client, oldFuncs, newFuncs map[string]string, threshold float64) ([]SemanticMatch, error) {
	oldEmbs := make(map[string][]float32)
	for name, body := range oldFuncs {
		emb, err := client.Embed(body)
		if err != nil {
			continue
		}
		oldEmbs[name] = emb
	}

	var matches []SemanticMatch
	for newName, newBody := range newFuncs {
		newEmb, err := client.Embed(newBody)
		if err != nil {
			continue
		}
		for oldName, oldEmb := range oldEmbs {
			if oldName == newName {
				continue
			}
			sim := CosineSimilarity(oldEmb, newEmb)
			if sim >= threshold {
				matches = append(matches, SemanticMatch{
					OldName:    oldName,
					NewName:    newName,
					Similarity: sim,
				})
			}
		}
	}
	return matches, nil
}
