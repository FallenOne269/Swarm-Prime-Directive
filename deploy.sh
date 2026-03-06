#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Swarm Prime Directive — One-Shot Deploy
# From zero to running on AWS ECS Fargate in one command.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed and running
#   - ANTHROPIC_API_KEY env var set
#
# Usage:
#   ./deploy.sh                      # Full deploy with defaults
#   ./deploy.sh --cycles 5           # Deploy and run 5 cycles
#   ./deploy.sh --region us-west-2   # Deploy to specific region
#   ./deploy.sh --dry-run            # Show what would happen
# ═══════════════════════════════════════════════════════════════════════════════

# ── Defaults ─────────────────────────────────────────────────────────────────
REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="prod"
STACK_NAME="swarm-prime-${ENVIRONMENT}"
CYCLES=3
FOCUS="general capability improvement"
SCHEDULE="rate(6 hours)"
DRY_RUN=false

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)    REGION="$2"; shift 2 ;;
        --cycles)    CYCLES="$2"; shift 2 ;;
        --focus)     FOCUS="$2"; shift 2 ;;
        --schedule)  SCHEDULE="$2"; shift 2 ;;
        --env)       ENVIRONMENT="$2"; STACK_NAME="swarm-prime-$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        --no-schedule) SCHEDULE="none"; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Preflight checks ────────────────────────────────────────────────────────
echo "═══ SWARM PRIME — DEPLOY ═══"
echo ""

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY environment variable not set."
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

if ! command -v aws &>/dev/null; then
    echo "ERROR: AWS CLI not found. Install: https://aws.amazon.com/cli/"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not found. Install: https://docs.docker.com/get-docker/"
    exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
    echo "ERROR: AWS credentials not configured. Run: aws configure"
    exit 1
}

ECR_URI="${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/swarm-prime-${ENVIRONMENT}"
IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "v1")

echo "  Region:       ${REGION}"
echo "  Account:      ${AWS_ACCOUNT}"
echo "  Stack:        ${STACK_NAME}"
echo "  ECR:          ${ECR_URI}"
echo "  Image Tag:    ${IMAGE_TAG}"
echo "  Cycles:       ${CYCLES}"
echo "  Focus:        ${FOCUS}"
echo "  Schedule:     ${SCHEDULE}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would deploy the above configuration."
    exit 0
fi

read -p "Deploy? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ── Step 1: Store API key in Secrets Manager ─────────────────────────────────
echo ""
echo "═══ Step 1/5: Storing API key in Secrets Manager ═══"
SECRET_NAME="swarm-prime/anthropic-api-key-${ENVIRONMENT}"
SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$REGION" --query ARN --output text 2>/dev/null || echo "")

if [[ -z "$SECRET_ARN" || "$SECRET_ARN" == "None" ]]; then
    SECRET_ARN=$(aws secretsmanager create-secret \
        --name "$SECRET_NAME" \
        --secret-string "$ANTHROPIC_API_KEY" \
        --region "$REGION" \
        --query ARN --output text)
    echo "  Created secret: ${SECRET_ARN}"
else
    aws secretsmanager put-secret-value \
        --secret-id "$SECRET_NAME" \
        --secret-string "$ANTHROPIC_API_KEY" \
        --region "$REGION" >/dev/null
    echo "  Updated secret: ${SECRET_ARN}"
fi

# ── Step 2: Deploy CloudFormation stack ──────────────────────────────────────
echo ""
echo "═══ Step 2/5: Deploying CloudFormation stack ═══"
aws cloudformation deploy \
    --template-file infra/cloudformation.yml \
    --stack-name "$STACK_NAME" \
    --parameter-overrides \
        Environment="$ENVIRONMENT" \
        AnthropicApiKeyArn="$SECRET_ARN" \
        CyclesPerRun="$CYCLES" \
        FocusArea="$FOCUS" \
        ScheduleExpression="$SCHEDULE" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$REGION" \
    --no-fail-on-empty-changeset

echo "  Stack deployed: ${STACK_NAME}"

# ── Step 3: Build Docker image ───────────────────────────────────────────────
echo ""
echo "═══ Step 3/5: Building Docker image ═══"
docker build -t "swarm-prime:${IMAGE_TAG}" -t swarm-prime:latest --target production .
echo "  Built: swarm-prime:${IMAGE_TAG}"

# ── Step 4: Push to ECR ─────────────────────────────────────────────────────
echo ""
echo "═══ Step 4/5: Pushing to ECR ═══"
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

docker tag "swarm-prime:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker tag swarm-prime:latest "${ECR_URI}:latest"
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"
echo "  Pushed: ${ECR_URI}:${IMAGE_TAG}"

# ── Step 5: Run first cycle ─────────────────────────────────────────────────
echo ""
echo "═══ Step 5/5: Running first improvement cycle ═══"

# Get networking config from stack outputs
SUBNETS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].Outputs' --output json | python3 -c "
import sys, json
outputs = json.load(sys.stdin)
# Parse subnets from RunTaskCommand output
for o in outputs:
    if o['OutputKey'] == 'RunTaskCommand':
        import re
        m = re.search(r'subnets=\[([^\]]+)\]', o['OutputValue'])
        if m: print(m.group(1).replace(',', ' '))
        break
" 2>/dev/null || echo "")

SG=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].Outputs' --output json | python3 -c "
import sys, json
outputs = json.load(sys.stdin)
for o in outputs:
    if o['OutputKey'] == 'RunTaskCommand':
        import re
        m = re.search(r'securityGroups=\[([^\]]+)\]', o['OutputValue'])
        if m: print(m.group(1))
        break
" 2>/dev/null || echo "")

CLUSTER=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='ClusterName'].OutputValue" --output text)

TASK_DEF=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='TaskDefinitionArn'].OutputValue" --output text)

if [[ -n "$SUBNETS" && -n "$SG" ]]; then
    SUBNET_ARR=$(echo "$SUBNETS" | tr ' ' ',')
    TASK_ARN=$(aws ecs run-task \
        --cluster "$CLUSTER" \
        --task-definition "$TASK_DEF" \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ARR],securityGroups=[$SG],assignPublicIp=ENABLED}" \
        --region "$REGION" \
        --query 'tasks[0].taskArn' --output text)
    echo "  Task started: ${TASK_ARN}"
    echo "  Logs: https://${REGION}.console.aws.amazon.com/ecs/v2/clusters/${CLUSTER}/tasks"
else
    echo "  WARNING: Could not parse network config. Run manually:"
    echo "  make ecs-run CYCLES=${CYCLES}"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " SWARM PRIME DIRECTIVE — DEPLOYED"
echo ""
echo " Stack:     ${STACK_NAME}"
echo " ECR:       ${ECR_URI}:${IMAGE_TAG}"
echo " Cluster:   ${CLUSTER}"
echo " Schedule:  ${SCHEDULE}"
echo ""
echo " Commands:"
echo "   make ecs-run CYCLES=5           # Run on-demand"
echo "   make status                     # Check state"
echo "   aws logs tail /ecs/${STACK_NAME} --follow  # Stream logs"
echo "═══════════════════════════════════════════════════════════════════════"
