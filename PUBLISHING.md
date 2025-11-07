# Publishing Wheels to GitHub Pages Index

This document explains how to publish fastsafetensors wheels (ROCm and CUDA) to custom PyPI indices hosted on GitHub Pages.

## Overview

We host custom Python package indices that include:
- **ROCm Index**: `https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/`
- **CUDA Index**: `https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/`

Each index includes:
- Platform-specific fastsafetensors wheels for Python 3.9-3.13
- All dependencies (typer, click, etc.) mirrored from PyPI

This allows users to install with:
```bash
# ROCm
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/

# CUDA
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/
```

## Initial Setup (One-Time)

Follow these steps to set up GitHub Pages hosting for your custom PyPI indices.

### Prerequisites

- You have the repository cloned locally
- You're on the `main` or `upstream` branch
- You have push access to the repository

### Step 1: Run the Setup Script

We provide a script that automates the gh-pages branch creation:

```bash
# Make sure you're in the repository root
cd /path/to/fastsafetensors-rocm

# Make the script executable (if not already)
chmod +x scripts/setup-gh-pages.sh

# Run the setup script
./scripts/setup-gh-pages.sh
```

The script will:
- Save your current branch
- Create a new orphan `gh-pages` branch
- Set up the directory structure for both ROCm and CUDA indices
- Create initial placeholder HTML files
- Commit the initial structure
- Switch you back to your original branch

### Step 2: Push the gh-pages Branch

After the script completes, push the new branch to GitHub:

```bash
git push origin gh-pages
```

### Step 3: Enable GitHub Pages in Repository Settings

1. Go to your repository on GitHub: `https://github.com/EmbeddedLLM/fastsafetensors-rocm`
2. Click **Settings** (top navigation bar)
3. In the left sidebar, click **Pages**
4. Under **Build and deployment**:
   - **Source**: Select "Deploy from a branch"
   - **Branch**: Select `gh-pages`
   - **Folder**: Select `/ (root)`
5. Click **Save**

### Step 4: Wait for Initial Deployment

GitHub will automatically deploy your gh-pages branch. This takes 1-2 minutes.

You can check the deployment status:
- Go to **Actions** tab in your repository
- Look for a "pages build and deployment" workflow

Once complete, your indices will be available at:
- **ROCm**: `https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/`
- **CUDA**: `https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/`

### Verification

Test that the pages are live:

```bash
# Check ROCm index
curl -I https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/

# Check CUDA index
curl -I https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/

# Should return: HTTP/2 200
```

**That's it!** Your GitHub Pages indices are now set up and ready to receive wheel packages.

## Publishing Wheels

Once you've completed the initial setup above, you can publish wheels to the indices.

### Automated Publishing (Recommended)

The workflow `.github/workflows/publish-to-index.yaml` automatically publishes when:

**Option 1: On Release**
1. Create a GitHub Release with tag like `v0.1.15-rocm` or `v0.1.15-cuda`
2. Upload your wheel files to the release
3. Workflow automatically detects platform from tag and publishes to appropriate index

**Option 2: Manual Trigger**
1. Go to **Actions** → **Publish wheels to custom index**
2. Click **Run workflow**
3. Choose platform (rocm, cuda, or both)
4. Enter version/tag (e.g., `v0.1.15-rocm`)
5. Click **Run workflow**

### What the Workflow Does

1. Downloads wheel files from the GitHub release
2. Downloads all dependencies from PyPI for multiple Python versions
3. Generates PEP 503 compliant index structure
4. Commits and pushes to `gh-pages` branch
5. GitHub Pages automatically deploys the update

## Directory Structure

After publishing, the `gh-pages` branch will contain:

```
gh-pages/
├── rocm/
│   ├── README.md                                    # ROCm user documentation
│   ├── packages/                                    # ROCm wheel files
│   │   ├── fastsafetensors-0.1.15-*.whl
│   │   ├── typer-0.9.0-*.whl
│   │   ├── click-8.1.7-*.whl
│   │   └── [other dependencies]
│   ├── simple/                                      # PEP 503 index
│   │   ├── index.html                              # Root package list
│   │   ├── fastsafetensors/
│   │   │   └── index.html                          # fastsafetensors wheels
│   │   ├── typer/
│   │   │   └── index.html
│   │   └── [other packages]/
│   └── package-list.txt                            # List for dumb-pypi
├── cuda/
│   ├── README.md                                    # CUDA user documentation
│   ├── packages/                                    # CUDA wheel files
│   ├── simple/                                      # PEP 503 index
│   └── package-list.txt
└── README.md
```

## User Installation

Once published, users can install with:

```bash
# ROCm - Install latest version
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/

# CUDA - Install latest version
pip install fastsafetensors --index-url https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/

# Install specific version
pip install fastsafetensors==0.1.15 --index-url https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/

# In requirements.txt (ROCm)
--index-url https://embeddedllm.github.io/fastsafetensors-rocm/rocm/simple/
fastsafetensors>=0.1.15

# In requirements.txt (CUDA)
--index-url https://embeddedllm.github.io/fastsafetensors-rocm/cuda/simple/
fastsafetensors>=0.1.15
```

## Troubleshooting

### Workflow fails to download release assets

- Ensure the release exists and has wheel files attached
- Check that `GITHUB_TOKEN` has appropriate permissions

### Index not updating

- Check GitHub Pages deployment status in **Settings** → **Pages**
- Verify the `gh-pages` branch was updated
- Wait 1-2 minutes for GitHub Pages to deploy

### Missing dependencies

- The workflow downloads common platforms (manylinux2014, manylinux_2_27, etc.)
- For specific platforms, modify the `pip download` command in the workflow

### Users can't install

- Verify the URL is correct and publicly accessible
- Check that `simple/` directory structure follows PEP 503
- Ensure wheel filenames are valid
