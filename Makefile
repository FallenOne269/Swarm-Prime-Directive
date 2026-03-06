.PHONY: install dev lint test build push deploy run status clean infra-deploy infra-destroy

# ── Config ───────────────────────────────────────────────────────────────────
AWS_REGION    ?= us-east-1
AWS_ACCOUNT   ?= $(shell aws sts get-caller-identity --query Account --output text)
ECR_REPO      ?= swarm-prime-prod
IMAGE_TAG     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "latest")
ECR_URI       ?= $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)
STACK_NAME    ?= swarm-prime-prod
CYCLES        ?= 1
FOCUS         ?= general capability improvement

# ── Local Development ────────────────────────────────────────────────────────
install:
	pip install .

dev:
	pip install ".[dev]"

lint:
	ruff check swarm_prime/
	ruff format --check swarm_prime/
	mypy swarm_prime/ --ignore-missing-imports

test:
	pytest tests/ -v --tb=short

# ── Docker ───────────────────────────────────────────────────────────────────
build:
	docker build -t swarm-prime:$(IMAGE_TAG) -t swarm-prime:latest --target production .

run: build
	docker run --rm \
		-e ANTHROPIC_API_KEY \
		-v swarm-output:/app/swarm_output \
		swarm-prime:latest run --cycles $(CYCLES) --focus "$(FOCUS)"

status:
	docker run --rm -v swarm-output:/app/swarm_output swarm-prime:latest status

# ── AWS ECR Push ─────────────────────────────────────────────────────────────
ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_URI)

push: build ecr-login
	docker tag swarm-prime:$(IMAGE_TAG) $(ECR_URI):$(IMAGE_TAG)
	docker tag swarm-prime:latest $(ECR_URI):latest
	docker push $(ECR_URI):$(IMAGE_TAG)
	docker push $(ECR_URI):latest
	@echo "Pushed: $(ECR_URI):$(IMAGE_TAG)"

# ── AWS Infrastructure ──────────────────────────────────────────────────────
infra-deploy:
	@echo "Deploying CloudFormation stack: $(STACK_NAME)"
	aws cloudformation deploy \
		--template-file infra/cloudformation.yml \
		--stack-name $(STACK_NAME) \
		--parameter-overrides \
			Environment=prod \
			AnthropicApiKeyArn=$(ANTHROPIC_SECRET_ARN) \
			CyclesPerRun=$(CYCLES) \
			FocusArea="$(FOCUS)" \
			ScheduleExpression="rate(6 hours)" \
		--capabilities CAPABILITY_NAMED_IAM \
		--region $(AWS_REGION)
	@echo "Stack deployed. ECR URI:"
	@aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryUri`].OutputValue' --output text

infra-destroy:
	aws cloudformation delete-stack --stack-name $(STACK_NAME) --region $(AWS_REGION)
	@echo "Stack deletion initiated: $(STACK_NAME)"

# ── Full Deploy Pipeline ────────────────────────────────────────────────────
deploy: infra-deploy push
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo " Swarm Prime deployed to ECS Fargate"
	@echo " ECR: $(ECR_URI):$(IMAGE_TAG)"
	@echo " Cluster: swarm-prime-prod"
	@echo " Schedule: every 6 hours"
	@echo "═══════════════════════════════════════════════════"

# ── On-Demand ECS Run ────────────────────────────────────────────────────────
ecs-run:
	@echo "Triggering on-demand cycle run on ECS..."
	aws ecs run-task \
		--cluster swarm-prime-prod \
		--task-definition swarm-prime-prod \
		--launch-type FARGATE \
		--network-configuration "$$(aws cloudformation describe-stacks \
			--stack-name $(STACK_NAME) \
			--query 'Stacks[0].Outputs[?OutputKey==`RunTaskCommand`].OutputValue' \
			--output text | grep -oP 'awsvpcConfiguration=\{[^}]+\}')" \
		--overrides '{"containerOverrides":[{"name":"swarm-prime","command":["run","--cycles","$(CYCLES)","--focus","$(FOCUS)"]}]}'

# ── Docker Compose ───────────────────────────────────────────────────────────
compose-run:
	SWARM_CYCLES=$(CYCLES) SWARM_FOCUS="$(FOCUS)" docker compose up --build

compose-status:
	docker compose --profile status up swarm-status

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	rm -rf swarm_output/ dist/ build/ *.egg-info __pycache__
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	docker volume rm swarm-output 2>/dev/null || true

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo "Swarm Prime Directive — Deployment Commands"
	@echo ""
	@echo "Local:"
	@echo "  make install          Install package"
	@echo "  make dev              Install with dev dependencies"
	@echo "  make lint             Ruff + mypy"
	@echo "  make test             Run pytest"
	@echo ""
	@echo "Docker:"
	@echo "  make build            Build Docker image"
	@echo "  make run              Run cycle locally in Docker"
	@echo "  make status           Show swarm state"
	@echo "  make compose-run      Run via docker-compose"
	@echo ""
	@echo "AWS:"
	@echo "  make infra-deploy     Deploy CloudFormation stack"
	@echo "  make push             Build + push to ECR"
	@echo "  make deploy           Full pipeline (infra + push)"
	@echo "  make ecs-run          Trigger on-demand ECS task"
	@echo "  make infra-destroy    Tear down stack"
	@echo ""
	@echo "Variables:"
	@echo "  CYCLES=3              Number of cycles (default: 1)"
	@echo "  FOCUS='...'           Focus area"
	@echo "  ANTHROPIC_SECRET_ARN  Secrets Manager ARN (for infra-deploy)"
