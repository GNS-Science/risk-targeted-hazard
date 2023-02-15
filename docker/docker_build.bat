:: build docker image
docker build -t risk_targeted_hazard . || exit /b 1

:: install dependencies (runs poetry install)
docker run --rm -v "%cd%\..:/src" --entrypoint="poetry" risk_targeted_hazard install
