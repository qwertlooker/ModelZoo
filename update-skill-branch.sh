#!/bin/bash
set -e

SKILL_PATH=".agents/skills/ascend-model-adapter"
SPLIT_BRANCH="skill/ascend-model-adapter"
MAIN_BRANCH="main"

echo "==> Updating $SPLIT_BRANCH from $MAIN_BRANCH..."

git checkout $MAIN_BRANCH
git pull origin $MAIN_BRANCH

if git show-ref --verify --quiet refs/heads/$SPLIT_BRANCH; then
    echo "==> Deleting existing local $SPLIT_BRANCH..."
    git branch -D $SPLIT_BRANCH
fi

echo "==> Running subtree split..."
SPLIT_HASH=$(git subtree split --prefix=$SKILL_PATH)
echo "==> Split hash: $SPLIT_HASH"

git branch $SPLIT_BRANCH $SPLIT_HASH

echo "==> Force pushing $SPLIT_BRANCH to origin..."
git push origin $SPLIT_BRANCH --force

git checkout $MAIN_BRANCH

echo "==> Done! $SPLIT_BRANCH is now up to date on remote."
echo "==> Consumers can pull directly from $SPLIT_BRANCH now."
