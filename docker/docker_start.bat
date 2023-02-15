:: read .env file variables
FOR /F "eol=# tokens=*" %%i IN (%~dp0.env) DO SET %%i

echo Mounting %cd% inside container with AWS credentials from %DOCKER_AWS_CREDENTIALS_DIR%

docker run --rm -i -t --env-file .env -p "8888:8888" -v "%cd%\..:/src" -v "%DOCKER_AWS_CREDENTIALS_DIR%:/home/user/.aws" risk_targeted_hazard
