set AWS_CREDENTIALS_DIR=%userprofile%\.aws

echo Mounting %cd% inside container with AWS credentials from %AWS_CREDENTIALS_DIR%

docker run --rm -i -t --env-file .env -p "8888:8888" -v "%cd%:/src" -v "%AWS_CREDENTIALS_DIR%:/home/user/.aws" risk_targeted_hazard