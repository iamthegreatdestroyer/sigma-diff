// Package diff provides structural code diffing that understands code semantics,
// not just text lines. It detects added, removed, modified, and moved functions.
package diff

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
)

type ChangeType string

const (
	Added    ChangeType = "added"
	Removed  ChangeType = "removed"
	Modified ChangeType = "modified"
	Moved    ChangeType = "moved"
)

type Symbol struct {
	Name      string
	Kind      string // "function", "type", "const", "var", "import"
	Signature string
	StartLine int
	EndLine   int
	Body      string
}

type Change struct {
	Type      ChangeType
	Symbol    string
	Kind      string
	OldLine   int
	NewLine   int
	Details   string
}

type DiffResult struct {
	File    string
	Changes []Change
	Stats   DiffStats
}

type DiffStats struct {
	Added    int
	Removed  int
	Modified int
	Moved    int
}

func ExtractGoSymbols(src string, filename string) ([]Symbol, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, src, parser.AllErrors)
	if err != nil {
		return nil, err
	}

	var symbols []Symbol

	for _, imp := range f.Imports {
		path := strings.Trim(imp.Path.Value, "\"")
		symbols = append(symbols, Symbol{
			Name:      path,
			Kind:      "import",
			StartLine: fset.Position(imp.Pos()).Line,
			EndLine:   fset.Position(imp.End()).Line,
		})
	}

	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			sig := d.Name.Name
			if d.Recv != nil && len(d.Recv.List) > 0 {
				sig = "method"
			}
			symbols = append(symbols, Symbol{
				Name:      d.Name.Name,
				Kind:      "function",
				Signature: sig,
				StartLine: fset.Position(d.Pos()).Line,
				EndLine:   fset.Position(d.End()).Line,
				Body:      src[d.Pos()-1 : d.End()-1],
			})
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					symbols = append(symbols, Symbol{
						Name:      s.Name.Name,
						Kind:      "type",
						StartLine: fset.Position(s.Pos()).Line,
						EndLine:   fset.Position(s.End()).Line,
					})
				case *ast.ValueSpec:
					kind := "var"
					if d.Tok == token.CONST {
						kind = "const"
					}
					for _, name := range s.Names {
						symbols = append(symbols, Symbol{
							Name:      name.Name,
							Kind:      kind,
							StartLine: fset.Position(name.Pos()).Line,
							EndLine:   fset.Position(s.End()).Line,
						})
					}
				}
			}
		}
	}

	return symbols, nil
}

func DiffGoFiles(oldSrc, newSrc, filename string) (*DiffResult, error) {
	oldSymbols, err := ExtractGoSymbols(oldSrc, filename)
	if err != nil {
		return nil, err
	}
	newSymbols, err := ExtractGoSymbols(newSrc, filename)
	if err != nil {
		return nil, err
	}

	oldMap := make(map[string]Symbol)
	for _, s := range oldSymbols {
		key := s.Kind + ":" + s.Name
		oldMap[key] = s
	}

	newMap := make(map[string]Symbol)
	for _, s := range newSymbols {
		key := s.Kind + ":" + s.Name
		newMap[key] = s
	}

	var changes []Change
	var stats DiffStats

	for key, newSym := range newMap {
		oldSym, existed := oldMap[key]
		if !existed {
			changes = append(changes, Change{
				Type:    Added,
				Symbol:  newSym.Name,
				Kind:    newSym.Kind,
				NewLine: newSym.StartLine,
				Details: "new " + newSym.Kind,
			})
			stats.Added++
		} else if newSym.Body != oldSym.Body && newSym.Kind == "function" {
			if newSym.StartLine != oldSym.StartLine {
				changes = append(changes, Change{
					Type:    Modified,
					Symbol:  newSym.Name,
					Kind:    newSym.Kind,
					OldLine: oldSym.StartLine,
					NewLine: newSym.StartLine,
					Details: "body changed and moved",
				})
			} else {
				changes = append(changes, Change{
					Type:    Modified,
					Symbol:  newSym.Name,
					Kind:    newSym.Kind,
					OldLine: oldSym.StartLine,
					NewLine: newSym.StartLine,
					Details: "body changed",
				})
			}
			stats.Modified++
		} else if newSym.StartLine != oldSym.StartLine {
			changes = append(changes, Change{
				Type:    Moved,
				Symbol:  newSym.Name,
				Kind:    newSym.Kind,
				OldLine: oldSym.StartLine,
				NewLine: newSym.StartLine,
				Details: "position changed",
			})
			stats.Moved++
		}
	}

	for key, oldSym := range oldMap {
		if _, exists := newMap[key]; !exists {
			changes = append(changes, Change{
				Type:    Removed,
				Symbol:  oldSym.Name,
				Kind:    oldSym.Kind,
				OldLine: oldSym.StartLine,
				Details: "removed",
			})
			stats.Removed++
		}
	}

	return &DiffResult{
		File:    filename,
		Changes: changes,
		Stats:   stats,
	}, nil
}
