# Upload to GitHub — Step by Step

## One-time setup

```bash
# 1. Create a new repo on GitHub:
#    Go to github.com → New Repository
#    Name: agi-mvp-arc-agi-1
#    Visibility: Public or Private
#    DO NOT initialise with README (we have one)

# 2. Configure git identity (if not done globally)
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

## Push the code

```bash
cd ~/github/agi-mvp-arc-agi-1

# Initialise git repo
git init

# Stage everything
git add .

# First commit
git commit -m "Initial commit: MDL symbolic search for ARC-AGI-1

- core/: domain-agnostic beam search, expression trees, primitive registry
- domains/arc/: 89 grid primitives, ARCDomain, 76-task benchmark, runner
- domains/symbolic_reg/: symbolic regression domain
- domains/cartpole/: symbolic RL on CartPole
- tests/: 111 unit tests (all passing)
- docs/: theory, adding primitives, adding domains
- README.md: architecture, results, quick start

Benchmark results:
  Baseline  (8  ops): 26% (20/76 tasks solved)
  Expanded  (89 ops): 76% (58/76 tasks solved)"

# Add your GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/agi-mvp-arc-agi-1.git

# Push
git branch -M main
git push -u origin main
```

## Subsequent updates

```bash
cd ~/github/agi-mvp-arc-agi-1

# Stage changes
git add .

# Commit with a descriptive message
git commit -m "Add new ARC primitives: gdilate, gerode, gborder_only"

# Push
git push
```

## Useful git commands

```bash
# Check what's changed
git status
git diff

# See commit history
git log --oneline

# Create a feature branch
git checkout -b feature/arc2-domain

# Switch back to main
git checkout main

# Tag a release
git tag v0.2.0
git push origin v0.2.0
```

## Verifying the push worked

```bash
# Clone to a fresh directory and run tests
git clone https://github.com/YOUR_USERNAME/agi-mvp-arc-agi-1.git /tmp/test-clone
cd /tmp/test-clone
python -m unittest discover tests/ -v
```

All 111 tests should pass with no external dependencies.
