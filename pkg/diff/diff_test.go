package diff

import (
	"testing"
)

const oldCode = `package main

import "fmt"

func Hello() string {
	return "hello"
}

func Goodbye() string {
	return "goodbye"
}

type Config struct {
	Name string
}
`

const newCode = `package main

import "fmt"
import "os"

func Hello() string {
	return "hello world"
}

func NewFunc() int {
	return 42
}

type Config struct {
	Name string
}

type Server struct {
	Port int
}
`

func TestExtractGoSymbols(t *testing.T) {
	symbols, err := ExtractGoSymbols(oldCode, "test.go")
	if err != nil {
		t.Fatal(err)
	}

	names := make(map[string]bool)
	for _, s := range symbols {
		names[s.Kind+":"+s.Name] = true
	}

	if !names["function:Hello"] {
		t.Error("missing function Hello")
	}
	if !names["function:Goodbye"] {
		t.Error("missing function Goodbye")
	}
	if !names["type:Config"] {
		t.Error("missing type Config")
	}
	if !names["import:fmt"] {
		t.Error("missing import fmt")
	}
}

func TestDiffGoFiles(t *testing.T) {
	result, err := DiffGoFiles(oldCode, newCode, "test.go")
	if err != nil {
		t.Fatal(err)
	}

	changeMap := make(map[string]ChangeType)
	for _, c := range result.Changes {
		changeMap[c.Kind+":"+c.Symbol] = c.Type
	}

	if changeMap["function:Hello"] != Modified {
		t.Errorf("Hello should be Modified, got %v", changeMap["function:Hello"])
	}
	if changeMap["function:Goodbye"] != Removed {
		t.Errorf("Goodbye should be Removed, got %v", changeMap["function:Goodbye"])
	}
	if changeMap["function:NewFunc"] != Added {
		t.Errorf("NewFunc should be Added, got %v", changeMap["function:NewFunc"])
	}
	if changeMap["type:Server"] != Added {
		t.Errorf("Server should be Added, got %v", changeMap["type:Server"])
	}
	if changeMap["import:os"] != Added {
		t.Errorf("import os should be Added, got %v", changeMap["import:os"])
	}

	if result.Stats.Added < 3 {
		t.Errorf("expected at least 3 additions, got %d", result.Stats.Added)
	}
	if result.Stats.Removed < 1 {
		t.Errorf("expected at least 1 removal, got %d", result.Stats.Removed)
	}
	if result.Stats.Modified < 1 {
		t.Errorf("expected at least 1 modification, got %d", result.Stats.Modified)
	}

	t.Logf("Changes: +%d -%d ~%d ↕%d", result.Stats.Added, result.Stats.Removed, result.Stats.Modified, result.Stats.Moved)
	for _, c := range result.Changes {
		t.Logf("  [%s] %s %s: %s", c.Type, c.Kind, c.Symbol, c.Details)
	}
}

func TestIdenticalFiles(t *testing.T) {
	result, err := DiffGoFiles(oldCode, oldCode, "same.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Changes) != 0 {
		t.Errorf("identical files should have 0 changes, got %d", len(result.Changes))
	}
}
