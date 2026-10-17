# Compose would otherwise read .env for interpolation; the local .env holds
# app secrets (not compose variables) and isn't valid compose syntax.
COMPOSE = docker compose --env-file /dev/null

.PHONY: up down logs test lint

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

test:
	docker build -q --target test -t travely-api-test . >/dev/null
	docker run --rm --network travely-backend_default \
		-e TEST_DATABASE_URL=postgresql+asyncpg://travely:travely@db:5432/travely_test \
		travely-api-test pytest -q

lint:
	docker build -q --target test -t travely-api-test . >/dev/null
	docker run --rm travely-api-test ruff check .
