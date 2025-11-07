#!/bin/bash
# Setup script to initialize gh-pages branch for hosting the PyPI index

set -e

echo "Setting up gh-pages branch for fastsafetensors-rocm index..."

# Check if we're in the right repo
if [ ! -d ".github/workflows" ]; then
    echo "Error: Must run from repository root"
    exit 1
fi

# Save current branch
CURRENT_BRANCH=$(git branch --show-current)

# Check if gh-pages branch already exists
if git show-ref --verify --quiet refs/heads/gh-pages; then
    echo "gh-pages branch already exists!"
    read -p "Do you want to recreate it? This will delete existing content. (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    git branch -D gh-pages
fi

# Create orphan gh-pages branch
echo "Creating gh-pages branch..."
git checkout --orphan gh-pages

# Remove all files
git rm -rf .

# Create initial structure
echo "Creating directory structure..."
mkdir -p rocm/simple rocm/packages
mkdir -p cuda/simple cuda/packages

# Create root README
cat > README.md << 'EOF'
# fastsafetensors Custom Package Index

This repository hosts custom Python package indices for fastsafetensors.

## ROCm Index

Install fastsafetensors with ROCm support:

```bash
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/
```

## CUDA Index

Install fastsafetensors with CUDA support:

```bash
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/
```

## What's Included

- fastsafetensors ROCm/CUDA builds for Python 3.9-3.13
- All dependencies mirrored from PyPI

This ensures `pip install` works completely offline from PyPI with just this index.

## For Maintainers

See the main branch's [PUBLISHING.md](https://github.com/EmbeddedLLM/fastsafetensors-rocm/blob/main/PUBLISHING.md) for publishing instructions.
EOF

# Create ROCm README
cat > rocm/README.md << 'EOF'
# fastsafetensors ROCm Index

## Installation

```bash
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/
```

## Packages

This index is automatically updated when new releases are published.

Check the [packages directory](packages/) for available wheels.
EOF

# Create CUDA README
cat > cuda/README.md << 'EOF'
# fastsafetensors CUDA Index

## Installation

```bash
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/
```

## Packages

This index is automatically updated when new releases are published.

Check the [packages directory](packages/) for available wheels.
EOF

# Create initial simple index structure for ROCm
cat > rocm/simple/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>fastsafetensors ROCm Index</title>
    <meta name="api-version" value="2" />
</head>
<body>
<h1>Simple Index - ROCm</h1>
<p>This index will be populated when wheels are published.</p>
<a href="fastsafetensors/">fastsafetensors</a><br/>
</body>
</html>
EOF

# Create fastsafetensors package directory for ROCm
mkdir -p rocm/simple/fastsafetensors

cat > rocm/simple/fastsafetensors/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Links for fastsafetensors</title>
</head>
<body>
<h1>Links for fastsafetensors</h1>
<p>Wheels will be added here when published.</p>
</body>
</html>
EOF

# Create initial simple index structure for CUDA
cat > cuda/simple/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>fastsafetensors CUDA Index</title>
    <meta name="api-version" value="2" />
</head>
<body>
<h1>Simple Index - CUDA</h1>
<p>This index will be populated when wheels are published.</p>
<a href="fastsafetensors/">fastsafetensors</a><br/>
</body>
</html>
EOF

# Create fastsafetensors package directory for CUDA
mkdir -p cuda/simple/fastsafetensors

cat > cuda/simple/fastsafetensors/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Links for fastsafetensors</title>
</head>
<body>
<h1>Links for fastsafetensors</h1>
<p>Wheels will be added here when published.</p>
</body>
</html>
EOF

# Commit
git add .
git commit -m "Initialize gh-pages branch for PyPI index hosting"

echo ""
echo "✓ gh-pages branch created successfully!"
echo ""
echo "Next steps:"
echo "1. Push the branch: git push origin gh-pages"
echo "2. Enable GitHub Pages in repository settings:"
echo "   Settings → Pages → Source: Deploy from branch → Branch: gh-pages / (root)"
echo "3. Wait a few minutes for GitHub Pages to deploy"
echo "4. Your indices will be available at:"
echo "   ROCm: https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/"
echo "   CUDA: https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/"
echo ""
echo "Switching back to ${CURRENT_BRANCH}..."

git checkout "$CURRENT_BRANCH"

echo ""
echo "Done! The gh-pages branch is ready. Don't forget to push it!"
echo "  git push origin gh-pages"
