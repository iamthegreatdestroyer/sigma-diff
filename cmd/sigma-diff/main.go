// sigma-diff CLI — structural code diffing that understands code semantics.
//
// Usage:
//   sigma-diff <file1> <file2>         Compare two Go files
//   sigma-diff --git HEAD~1            Diff against git history
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/iamthegreatdestroyer/sigma-diff/pkg/diff"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: sigma-diff <file1> <file2>")
		fmt.Println("       sigma-diff --git HEAD~1")
		os.Exit(1)
	}

	if os.Args[1] == "--git" {
		ref := "HEAD~1"
		if len(os.Args) > 2 {
			ref = os.Args[2]
		}
		diffGit(ref)
		return
	}

	if len(os.Args) < 3 {
		fmt.Println("Usage: sigma-diff <file1> <file2>")
		os.Exit(1)
	}

	diffFiles(os.Args[1], os.Args[2])
}

func diffFiles(path1, path2 string) {
	old, err := os.ReadFile(path1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", path1, err)
		os.Exit(1)
	}
	new, err := os.ReadFile(path2)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", path2, err)
		os.Exit(1)
	}

	result, err := diff.DiffGoFiles(string(old), string(new), path2)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Diff error: %v\n", err)
		os.Exit(1)
	}

	printResult(result)
}

func diffGit(ref string) {
	cmd := exec.Command("git", "diff", "--name-only", ref)
	out, err := cmd.Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "git diff failed: %v\n", err)
		os.Exit(1)
	}

	files := strings.Split(strings.TrimSpace(string(out)), "\n")
	for _, f := range files {
		if !strings.HasSuffix(f, ".go") || f == "" {
			continue
		}

		oldCmd := exec.Command("git", "show", ref+":"+f)
		oldContent, err := oldCmd.Output()
		if err != nil {
			continue
		}

		newContent, err := os.ReadFile(f)
		if err != nil {
			continue
		}

		result, err := diff.DiffGoFiles(string(oldContent), string(newContent), f)
		if err != nil || len(result.Changes) == 0 {
			continue
		}

		printResult(result)
	}
}

func printResult(result *diff.DiffResult) {
	if len(result.Changes) == 0 {
		return
	}

	fmt.Printf("\n%s: +%d -%d ~%d move:%d\n",
		result.File, result.Stats.Added, result.Stats.Removed,
		result.Stats.Modified, result.Stats.Moved)

	for _, c := range result.Changes {
		symbol := " "
		switch c.Type {
		case diff.Added:
			symbol = "+"
		case diff.Removed:
			symbol = "-"
		case diff.Modified:
			symbol = "~"
		case diff.Moved:
			symbol = ">"
		}
		fmt.Printf("  %s %s %s: %s\n", symbol, c.Kind, c.Symbol, c.Details)
	}

	if os.Getenv("SIGMA_DIFF_JSON") == "1" {
		data, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(data))
	}
}
