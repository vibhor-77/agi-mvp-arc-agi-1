#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Upload this folder to GitHub as vibhor-77/agi-mvp-arc-agi-1
# Run this script from inside the folder you want to upload.
# ─────────────────────────────────────────────────────────────

read -rsp "Enter your GitHub Personal Access Token: " GITHUB_TOKEN
echo
GITHUB_USER="vibhor-77"
REPO_NAME="agi-mvp-arc-agi-1"

echo "🚀 Creating GitHub repository '$REPO_NAME' under '$GITHUB_USER'..."

# Step 1: Create the repository via GitHub API
RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO_NAME\",\"private\":false,\"description\":\"Uploaded via Cowork\"}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)

if [ "$HTTP_CODE" == "201" ]; then
  echo "✅ Repository created successfully!"
elif [ "$HTTP_CODE" == "422" ]; then
  echo "ℹ️  Repository already exists — continuing..."
else
  echo "❌ Failed to create repository (HTTP $HTTP_CODE). Check your token and try again."
  echo "$BODY"
  exit 1
fi

# Step 2: Initialize git (if not already)
if [ ! -d ".git" ]; then
  echo "📁 Initializing git repository..."
  git init
  git branch -M main
fi

# Step 3: Configure remote
REMOTE_URL="https://$GITHUB_TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"

if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

# Step 4: Stage, commit, and push
echo "📦 Staging all files..."
git add .

echo "💾 Committing..."
git commit -m "Initial upload" 2>/dev/null || echo "ℹ️  Nothing new to commit."

echo "⬆️  Pushing to GitHub..."
git push -u origin main --force

echo ""
echo "✅ Done! View your repo at: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "🔐 Security tip: Delete this script and revoke your token at"
echo "   https://github.com/settings/tokens once the upload is confirmed."
