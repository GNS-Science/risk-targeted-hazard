.PHONY: developer
developer:
	docker compose build risk-targeted-hazard
	docker compose run -T --rm --entrypoint="poetry install" risk-targeted-hazard

.PHONY: start
start:
	docker compose up -d risk-targeted-hazard && sleep 1
	docker compose exec risk-targeted-hazard poetry run jupyter notebook list

.PHONY: stop
stop:
	docker compose down

.PHONY: shell
shell:
	docker compose exec risk-targeted-hazard poetry shell

.PHONY: venv-update
venv-update:
	docker compose run -T --rm --entrypoint="poetry update ${POETRY_PACKAGES}" risk-targeted-hazard

.PHONY: test
test:
	docker compose run --rm --entrypoint="poetry run pytest test" risk-targeted-hazard

.PHONY: lint
lint:
	docker compose run -T --rm --entrypoint="poetry run mypy risk_targeted_hazard" risk-targeted-hazard