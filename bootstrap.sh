#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Swarm Prime Directive — Repo Bootstrap
#
# Run this ONCE from the swarm_prime/ directory after downloading the files.
#
# Prerequisites:
#   - gh CLI installed (brew install gh / apt install gh)
#   - gh auth login completed
#   - ANTHROPIC_API_KEY env var set
#   - AWS CLI configured (aws configure) with an IAM role that has:
#       ecs:*, ecr:*, cloudformation:*, secretsmanager:*, s3:*, iam:PassRole,
#       iam:CreateRole, logs:*, events:*
#
# Usage:
#   cd /path/to/swarm_prime
#   chmod +x bootstrap.sh
#   ./bootstrap.sh
# ═══════════════════════════════════════════════════════════════════════════════

REPO_NAME="swarm-prime"
REPO_VISIBILITY="private"  # Change to "public" if you want it public

echo "═══ SWARM PRIME — REPO BOOTSTRAP ═══"
echo ""

# ── Preflight ────────────────────────────────────────────────────────────────
if ! command -v gh &>/dev/null; then
    echo "ERROR: GitHub CLI (gh) not found."
    echo "  macOS:  brew install gh"
    echo "  Linux:  https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "ERROR: Not authenticated with GitHub. Run: gh auth login"
    exit 1
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

GH_USER=$(gh api user --jq .login)
echo "  GitHub user: ${GH_USER}"
echo "  Repo:        ${GH_USER}/${REPO_NAME}"
echo ""

# ── Step 1: Create GitHub repo ───────────────────────────────────────────────
echo "═══ Step 1/4: Creating GitHub repository ═══"

if gh repo view "${GH_USER}/${REPO_NAME}" &>/dev/null; then
    echo "  Repo already exists: ${GH_USER}/${REPO_NAME}"
else
    gh repo create "${REPO_NAME}" \
        --"${REPO_VISIBILITY}" \
        --description "Swarm Prime Directive — Recursive General Intelligence Construction Framework" \
        --disable-wiki \
        --clone=false
    echo "  Created: ${GH_USER}/${REPO_NAME}"
fi

# ── Step 2: Set GitHub secrets ───────────────────────────────────────────────
echo ""
echo "═══ Step 2/4: Setting repository secrets ═══"

# Anthropic API key
echo "${ANTHROPIC_API_KEY}" | gh secret set ANTHROPIC_API_KEY --repo "${GH_USER}/${REPO_NAME}"
echo "  ✓ ANTHROPIC_API_KEY"

# AWS Role ARN — check if user has one configured
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
if [[ -n "$AWS_ACCOUNT" ]]; then
    # Try to find or create an OIDC role for GitHub Actions
    ROLE_ARN=$(aws iam get-role --role-name github-actions-swarm-prime --query Role.Arn --output text 2>/dev/null || echo "")

    if [[ -z "$ROLE_ARN" || "$ROLE_ARN" == "None" ]]; then
        echo ""
        echo "  No GitHub Actions OIDC role found. Creating one..."

        # Create OIDC provider if it doesn't exist
        OIDC_ARN=$(aws iam list-open-id-connect-providers --query "OpenIDConnectProviderList[?ends_with(Arn, 'token.actions.githubusercontent.com')].Arn | [0]" --output text 2>/dev/null || echo "")
        if [[ -z "$OIDC_ARN" || "$OIDC_ARN" == "None" ]]; then
            OIDC_ARN=$(aws iam create-open-id-connect-provider \
                --url "https://token.actions.githubusercontent.com" \
                --client-id-list "sts.amazonaws.com" \
                --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" \
                --query OpenIDConnectProviderArn --output text)
            echo "  ✓ Created OIDC provider"
        fi

        # Create role with trust policy
        TRUST_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "${OIDC_ARN}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                },
                "StringLike": {
                    "token.actions.githubusercontent.com:sub": "repo:${GH_USER}/${REPO_NAME}:*"
                }
            }
        }
    ]
}
EOF
)
        ROLE_ARN=$(aws iam create-role \
            --role-name github-actions-swarm-prime \
            --assume-role-policy-document "${TRUST_POLICY}" \
            --query Role.Arn --output text)

        # Attach required policies
        for POLICY in \
            arn:aws:iam::aws:policy/AmazonECS_FullAccess \
            arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser \
            arn:aws:iam::aws:policy/SecretsManagerReadWrite \
            arn:aws:iam::aws:policy/CloudWatchLogsFullAccess; do
            aws iam attach-role-policy --role-name github-actions-swarm-prime --policy-arn "$POLICY"
        done

        # Inline policy for CloudFormation + IAM PassRole
        aws iam put-role-policy \
            --role-name github-actions-swarm-prime \
            --policy-name cfn-deploy \
            --policy-document '{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["cloudformation:*", "iam:PassRole", "iam:GetRole", "s3:*", "events:*", "ec2:*"],
                        "Resource": "*"
                    }
                ]
            }'

        echo "  ✓ Created IAM role: ${ROLE_ARN}"
    fi

    gh secret set AWS_ROLE_ARN --repo "${GH_USER}/${REPO_NAME}" --body "${ROLE_ARN}"
    echo "  ✓ AWS_ROLE_ARN = ${ROLE_ARN}"
else
    echo "  ⚠ AWS not configured. Skipping AWS_ROLE_ARN."
    echo "    Set it manually: gh secret set AWS_ROLE_ARN --repo ${GH_USER}/${REPO_NAME} --body <arn>"
fi

# ── Step 3: Git init + commit ────────────────────────────────────────────────
echo ""
echo "═══ Step 3/4: Initializing git repository ═══"

if [[ ! -d .git ]]; then
    git init
    echo "  ✓ git init"
fi

# Set remote
REMOTE_URL="https://github.com/${GH_USER}/${REPO_NAME}.git"
if git remote get-url origin &>/dev/null; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi
echo "  ✓ Remote: ${REMOTE_URL}"

# Ensure main branch
git checkout -B main 2>/dev/null || true

# Stage and commit
git add -A
git commit -m "feat: Swarm Prime Directive v1.0.0

6-agent recursive improvement loop for general intelligence construction.

Agents: Architect, Skeptic, Experiment Designer, Evaluator, Memory Curator, Alignment Guardian
Infrastructure: Docker, ECS Fargate, CloudFormation, GitHub Actions CI/CD
Core: Pydantic v2 domain models, peer review protocol, constraint layer, meta-cognition engine" \
    2>/dev/null || echo "  (nothing to commit)"

echo "  ✓ Committed"

# ── Step 4: Push ─────────────────────────────────────────────────────────────
echo ""
echo "═══ Step 4/4: Pushing to GitHub ═══"

git push -u origin main
echo "  ✓ Pushed to ${REMOTE_URL}"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " SWARM PRIME — BOOTSTRAP COMPLETE"
echo ""
echo " Repo:     https://github.com/${GH_USER}/${REPO_NAME}"
echo " CI:       https://github.com/${GH_USER}/${REPO_NAME}/actions"
echo ""
echo " The CI pipeline is now running. It will:"
echo "   1. Lint (ruff + mypy)"
echo "   2. Run tests (pytest)"
echo "   3. Build + push Docker image to ECR"
echo "   4. Deploy to ECS Fargate"
echo ""
echo " Next steps:"
echo "   # Deploy AWS infra (if not using CI auto-deploy):"
echo "   ./deploy.sh --cycles 3 --focus 'cross-domain transfer'"
echo ""
echo "   # Run on-demand from GitHub:"
echo "   gh workflow run ci.yml -f cycles=5 -f focus='abstraction formation'"
echo ""
echo "   # Monitor:"
echo "   gh run list --repo ${GH_USER}/${REPO_NAME}"
echo "═══════════════════════════════════════════════════════════════════════"
