---
name: Checkpointing Skill
description: Use this skill when performing risky or multi-step operations that may need to be rolled back. Checkpointing uses git to capture complete workspace state, enabling safe exploration and recovery from mistakes. Essential for refactoring, large-scale changes, experimental implementations, or any operation where rollback capability is valuable.
---

# Checkpointing Skill

## Purpose

This skill enables AI agents to work safely on risky operations by creating git-based restore points. Checkpoints capture complete workspace state, allowing you to confidently attempt complex changes knowing you can roll back if needed.

**Use this skill when:**
- Performing large refactoring or architectural changes
- Experimenting with different implementation approaches
- Making changes that affect multiple interconnected files
- Working on unfamiliar codebases where mistakes are likely
- Before operations that might break the build or tests
- Any time you want a "save point" before risky work

**Don't use this skill for:**
- Simple, isolated file edits
- Adding new files that don't affect existing code
- Documentation-only changes
- When the user explicitly requests no checkpoints

## When to Create Checkpoints

### High-Value Checkpoint Scenarios

| Scenario | Risk Level | Checkpoint Value |
|----------|------------|------------------|
| Refactoring core modules | High | Essential |
| Changing database schemas | High | Essential |
| Updating major dependencies | High | Essential |
| Implementing new architecture patterns | Medium | Recommended |
| Modifying shared utilities | Medium | Recommended |
| Experimenting with alternative approaches | Medium | Recommended |
| Multi-file coordinated changes | Medium | Recommended |
| Simple bug fixes | Low | Optional |
| Adding isolated new features | Low | Optional |

### Decision Framework

Before starting a task, evaluate:

1. **Blast radius**: How many files/systems could be affected?
2. **Reversibility**: Can changes be easily undone manually?
3. **Confidence level**: How certain are you about the approach?
4. **Recovery cost**: How long to redo work if something goes wrong?

**Create a checkpoint when 2+ factors indicate medium-high risk.**

## Git-Based Checkpoint Operations

### Create a Checkpoint

Before risky operations, stage and commit current state:

```bash
# Stage all changes (including untracked files)
git add -A

# Create checkpoint commit with descriptive message
git commit -m "checkpoint: before auth refactor"
```

If you want to preserve the checkpoint without adding to main history:

```bash
# Create checkpoint on a temporary branch
git stash push -u -m "checkpoint: before auth refactor"
```

### Revert to Checkpoint

If something goes wrong:

```bash
# Option 1: Discard all changes since last commit (hard reset)
git reset --hard HEAD

# Option 2: Revert to specific checkpoint commit
git reset --hard <checkpoint-commit-sha>

# Option 3: Restore from stash
git stash pop
```

### View Checkpoint History

```bash
# See recent commits (checkpoints)
git log --oneline -10

# See stashed checkpoints
git stash list
```

## Checkpoint Workflow Patterns

### Basic Pattern

```
1. BEFORE risky operation:
   git add -A && git commit -m "checkpoint: before <operation>"

2. PERFORM the operation:
   - Make your changes
   - Run builds/tests as appropriate

3. EVALUATE results:
   - If successful: Continue working
   - If failed: git reset --hard HEAD~1
```

### Multi-Attempt Pattern

For experimental or uncertain implementations:

```
1. git add -A && git commit -m "checkpoint: before auth refactor"

2. Attempt #1: JWT-based approach
   - Implement changes
   - Test: Build fails
   - git reset --hard HEAD~1

3. Attempt #2: Session-based approach
   - Implement changes
   - Test: Works!
   - git add -A && git commit -m "feat: session-based auth"
```

### Stash-Based Pattern (No History Pollution)

When you don't want checkpoint commits in history:

```bash
# Save current state to stash
git stash push -u -m "checkpoint: working state before experiment"

# Make experimental changes...

# If experiment fails, restore:
git checkout -- .
git clean -fd
git stash pop

# If experiment succeeds, drop stash:
git stash drop
```

## Best Practices

### Checkpoint Naming

Use descriptive commit messages:
- `checkpoint: before payment system refactor`
- `checkpoint: working auth pre-optimization`
- `checkpoint: stable state before migration`

### Checkpoint Hygiene

1. **Don't over-checkpoint**: Not every edit needs a checkpoint
2. **Squash checkpoints**: After success, squash checkpoint commits if needed
3. **Use stash for experiments**: Avoid polluting git history with failed attempts
4. **Verify before reset**: Make sure you're reverting to the right state

### Recovery Procedure

When a checkpoint revert is needed:

1. **Stop current work**: Don't make more changes
2. **Assess what went wrong**: Understand the failure
3. **Check git status**: `git status` and `git log --oneline -5`
4. **Execute revert**: `git reset --hard <target>`
5. **Verify restoration**: Check that state is correct
6. **Plan next attempt**: Decide on different approach

## Quick Reference

### Checkpoint Decision Checklist

Before major operations, ask:
- [ ] Am I changing core business logic?
- [ ] Could this break the build?
- [ ] Are multiple files interconnected?
- [ ] Am I unsure about the best approach?
- [ ] Would I lose significant work if this fails?

**2+ checks = Create a checkpoint**

### Common Commands

| Action | Command |
|--------|---------|
| Create checkpoint | `git add -A && git commit -m "checkpoint: <desc>"` |
| Revert last change | `git reset --hard HEAD~1` |
| Revert all uncommitted | `git reset --hard HEAD` |
| Stash checkpoint | `git stash push -u -m "checkpoint: <desc>"` |
| Restore from stash | `git stash pop` |
| View history | `git log --oneline -10` |
| View stashes | `git stash list` |

### Workflow by Task Type

| Task Type | Checkpoint Strategy |
|-----------|-------------------|
| Bug fix | Optional: checkpoint if touching critical code |
| New feature | Checkpoint at major milestones |
| Refactoring | Essential: checkpoint before starting |
| Dependency update | Essential: checkpoint before `npm install` |
| Database migration | Essential: checkpoint before and after |
| Performance optimization | Checkpoint working version before optimizing |
| Experimental code | Use stash, cleanup on success |
