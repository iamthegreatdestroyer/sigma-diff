#!/usr/bin/env python3
"""
Ryzanstein LLM - Project Renaming Script
=========================================
Safely renames all occurrences of Ryzanstein LLM/Ryzanstein to Ryzanstein throughout the project.

Usage:
    # Dry run (see what would change)
    python rename_to_ryzanstein.py ./Ryzanstein
    
    # Apply changes
    python rename_to_ryzanstein.py ./Ryzanstein --apply
    
    # Custom output file
    python rename_to_ryzanstein.py ./Ryzanstein --apply --output my_report.json
"""

import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set


class ProjectRenamer:
    """Handles comprehensive project renaming from Ryzanstein LLM/Ryzanstein to Ryzanstein."""
    
    def __init__(self, root_dir: str, dry_run: bool = True):
        self.root_dir = Path(root_dir).resolve()
        self.dry_run = dry_run
        self.changes: List[Dict] = []
        self.errors: List[Dict] = []
        
        # Renaming patterns (order matters - most specific first)
        self.patterns = [
            # Full project names with hyphen
            (r'\bRYZEN-LLM\b', 'Ryzanstein LLM'),
            (r'\bRyzen-LLM\b', 'Ryzanstein LLM'),
            (r'\bryzen-llm\b', 'ryzanstein-llm'),
            
            # Full project names with underscore
            (r'\bRYZEN_LLM\b', 'RYZANSTEIN_LLM'),
            (r'\bryzen_llm\b', 'ryzanstein_llm'),
            (r'\bRyzen_LLM\b', 'Ryzanstein_LLM'),
            
            # Standalone RYZANSTEIN (not followed by hyphen to avoid double-replacing)
            (r'\bRYZEN\b(?![-_])', 'RYZANSTEIN'),
            (r'\bRyzen\b(?![-_])', 'Ryzanstein'),
            (r'\bryzen\b(?![-_])', 'ryzanstein'),
            
            # Ryzanstein variations
            (r'\bRyot\b', 'Ryzanstein'),
            (r'\bRYOT\b', 'RYZANSTEIN'),
            (r'\bryot\b', 'ryzanstein'),
            
            # Module/package names with dots
            (r'ryzanstein\.llm', 'ryzanstein.llm'),
            (r'RYZANSTEIN\.LLM', 'RYZANSTEIN.LLM'),
        ]
        
        # File extensions to process
        self.file_patterns: Set[str] = {
            '.py', '.go', '.md', '.json', '.yaml', '.yml',
            '.toml', '.txt', '.sh', '.ps1', '.cpp', '.c', '.h',
            '.hpp', '.html', '.css', '.js', '.ts', '.tsx', '.svelte',
            '.dockerfile', '.env', '.cfg', '.ini', '.xml', '.rst',
            '.bat', '.cmd', '.make', '.cmake'
        }
        
        # Directories to skip
        self.exclude_dirs: Set[str] = {
            '.git', 'node_modules', '__pycache__', '.venv',
            'venv', 'build', 'dist', '.history', '.cache',
            'target', '.idea', '.vscode', 'vendor', '.tox',
            'eggs', '*.egg-info', '.mypy_cache', '.pytest_cache'
        }
        
        # Special files to always process regardless of extension
        self.special_files: Set[str] = {
            'CMakeLists.txt', 'Makefile', 'Dockerfile',
            'docker-compose.yml', 'docker-compose.yaml',
            '.gitignore', '.dockerignore', 'requirements.txt',
            'go.mod', 'go.sum', 'Cargo.toml', 'Cargo.lock',
            'pyproject.toml', 'setup.py', 'setup.cfg',
            'package.json', 'package-lock.json', 'tsconfig.json',
            'webpack.config.js', 'vite.config.js', 'wails.json'
        }
    
    def should_process_file(self, path: Path) -> bool:
        """Check if file should be processed based on path and extension."""
        # Check if any parent directory should be excluded
        for part in path.parts:
            if part in self.exclude_dirs:
                return False
            # Handle wildcard patterns like *.egg-info
            for exclude in self.exclude_dirs:
                if '*' in exclude and part.endswith(exclude.replace('*', '')):
                    return False
        
        # Always process special files
        if path.name in self.special_files:
            return True
        
        # Check file extension
        return path.suffix.lower() in self.file_patterns
    
    def process_content(self, content: str, file_path: Path) -> Tuple[str, int]:
        """Apply renaming patterns to content, return new content and change count."""
        new_content = content
        total_changes = 0
        
        for pattern, replacement in self.patterns:
            matches = re.findall(pattern, new_content)
            if matches:
                new_content = re.sub(pattern, replacement, new_content)
                total_changes += len(matches)
        
        return new_content, total_changes
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file, return True if changes were made."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Apply changes
            new_content, change_count = self.process_content(content, file_path)
            
            if change_count > 0:
                relative_path = str(file_path.relative_to(self.root_dir))
                self.changes.append({
                    'type': 'content_change',
                    'file': relative_path,
                    'changes': change_count
                })
                
                if not self.dry_run:
                    # Create backup
                    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                    shutil.copy2(file_path, backup_path)
                    
                    # Write new content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                
                return True
            
            return False
            
        except Exception as e:
            self.errors.append({
                'file': str(file_path.relative_to(self.root_dir)),
                'error': str(e)
            })
            return False
    
    def get_new_name(self, name: str) -> str:
        """Apply renaming patterns to a file/directory name."""
        new_name = name
        for pattern, replacement in self.patterns:
            new_name = re.sub(pattern, replacement, new_name)
        return new_name
    
    def rename_files(self) -> None:
        """Rename files that contain old naming patterns."""
        file_renames: List[Tuple[Path, Path]] = []
        
        for file_path in self.root_dir.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Check exclusions
            skip = False
            for part in file_path.parts:
                if part in self.exclude_dirs:
                    skip = True
                    break
            if skip:
                continue
            
            new_name = self.get_new_name(file_path.name)
            
            if new_name != file_path.name:
                new_path = file_path.parent / new_name
                file_renames.append((file_path, new_path))
        
        for old_path, new_path in file_renames:
            self.changes.append({
                'type': 'file_rename',
                'from': str(old_path.relative_to(self.root_dir)),
                'to': str(new_path.relative_to(self.root_dir))
            })
            
            if not self.dry_run:
                old_path.rename(new_path)
    
    def rename_directories(self) -> None:
        """Rename directories that contain old naming patterns (bottom-up)."""
        dir_renames: List[Tuple[Path, Path]] = []
        
        # Walk bottom-up to rename deepest directories first
        for dirpath, dirnames, _ in os.walk(self.root_dir, topdown=False):
            for dirname in dirnames:
                if dirname in self.exclude_dirs:
                    continue
                
                new_name = self.get_new_name(dirname)
                
                if new_name != dirname:
                    old_path = Path(dirpath) / dirname
                    new_path = Path(dirpath) / new_name
                    dir_renames.append((old_path, new_path))
        
        for old_path, new_path in dir_renames:
            # Only add if the path still exists (might have been renamed by parent)
            if old_path.exists():
                self.changes.append({
                    'type': 'directory_rename',
                    'from': str(old_path.relative_to(self.root_dir)),
                    'to': str(new_path.relative_to(self.root_dir))
                })
                
                if not self.dry_run:
                    try:
                        old_path.rename(new_path)
                    except PermissionError as e:
                        # Handle permission errors gracefully
                        self.errors.append({
                            'file': str(old_path.relative_to(self.root_dir)),
                            'error': f'Permission denied during directory rename: {e}'
                        })
                        print(f"  ⚠️  Skipped directory rename due to permission error: {old_path}")
                        continue
    
    def run(self) -> Dict:
        """Execute the renaming process and return a report."""
        mode = "DRY RUN" if self.dry_run else "APPLYING CHANGES"
        print(f"\n{'=' * 60}")
        print(f"  RYZANSTEIN RENAMING TOOL - {mode}")
        print(f"{'=' * 60}")
        print(f"  Root directory: {self.root_dir}")
        print(f"{'=' * 60}\n")
        
        # Phase 1: Process file contents
        print("Phase 1: Processing file contents...")
        processed_count = 0
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file() and self.should_process_file(file_path):
                if self.process_file(file_path):
                    processed_count += 1
        print(f"  → Processed {processed_count} files with content changes")
        
        # Phase 2: Rename files
        print("\nPhase 2: Renaming files...")
        files_before = len(self.changes)
        self.rename_files()
        files_renamed = len([c for c in self.changes[files_before:] if c.get('type') == 'file_rename'])
        print(f"  → Found {files_renamed} files to rename")
        
        # Phase 3: Rename directories
        print("\nPhase 3: Renaming directories...")
        dirs_before = len(self.changes)
        self.rename_directories()
        dirs_renamed = len([c for c in self.changes[dirs_before:] if c.get('type') == 'directory_rename'])
        print(f"  → Found {dirs_renamed} directories to rename")
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'root_directory': str(self.root_dir),
            'statistics': {
                'files_with_content_changes': len([c for c in self.changes if c.get('type') == 'content_change']),
                'total_content_replacements': sum(c.get('changes', 0) for c in self.changes),
                'files_renamed': len([c for c in self.changes if c.get('type') == 'file_rename']),
                'directories_renamed': len([c for c in self.changes if c.get('type') == 'directory_rename']),
                'errors': len(self.errors)
            },
            'changes': self.changes,
            'errors': self.errors
        }
        
        return report


def print_report_summary(report: Dict) -> None:
    """Print a formatted summary of the renaming report."""
    stats = report['statistics']
    
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY {'(DRY RUN)' if report['dry_run'] else '(CHANGES APPLIED)'}")
    print(f"{'=' * 60}")
    print(f"  Files with content changes:  {stats['files_with_content_changes']}")
    print(f"  Total text replacements:     {stats['total_content_replacements']}")
    print(f"  Files renamed:               {stats['files_renamed']}")
    print(f"  Directories renamed:         {stats['directories_renamed']}")
    print(f"  Errors:                      {stats['errors']}")
    print(f"{'=' * 60}")
    
    if report['dry_run']:
        print("\n  ⚠️  This was a DRY RUN. No changes were made.")
        print("  To apply changes, run again with --apply flag.\n")
    else:
        print("\n  ✅ Changes have been applied.")
        print("  Backup files (.bak) created for modified files.\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Rename Ryzanstein LLM/Ryzanstein to Ryzanstein throughout the project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes (dry run)
  python rename_to_ryzanstein.py ./Ryzanstein
  
  # Apply changes
  python rename_to_ryzanstein.py ./Ryzanstein --apply
  
  # Save report to custom file
  python rename_to_ryzanstein.py ./Ryzanstein --output my_report.json

After running with --apply:
  1. Review the report file for all changes made
  2. Run your test suite to verify nothing broke
  3. Delete .bak files when satisfied: find . -name "*.bak" -delete
  4. Rename the root directory: mv Ryzanstein Ryzanstein
  5. Update git remote if needed
        """
    )
    
    parser.add_argument(
        'directory',
        help='Root directory of the project to rename'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply changes (default is dry-run)'
    )
    parser.add_argument(
        '--output',
        default='ryzanstein_rename_report.json',
        help='Output file for the detailed report (default: ryzanstein_rename_report.json)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        return 1
    
    # Run renamer
    renamer = ProjectRenamer(args.directory, dry_run=not args.apply)
    report = renamer.run()
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print_report_summary(report)
    print(f"  Full report saved to: {args.output}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
