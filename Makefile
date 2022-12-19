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
	docker compose stop risk-targeted-hazard

.PHONY: dev-shell
dev-shell:
	docker compose exec risk-targeted-hazard poetry run bash

.PHONY: lint
lint:
	docker compose run -T --rm --entrypoint="poetry run mypy risk_targeted_hazard" risk-targeted-hazard