# AWS deployment notes (not executed - no cloud account was used)

Target: ECR (image) + ECS Fargate (API) behind an Application Load Balancer. Everything is parameterised by environment variables, so no values are hard-coded.

```bash
export AWS_REGION=eu-west-1 AWS_ACCOUNT_ID=<your-account-id> REPO=purchase-intent
aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker build -f docker/Dockerfile -t $REPO:latest .
docker tag $REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

ECS task definition essentials: 0.5 vCPU / 1 GB, container port 8000, health check `GET /health`, command `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`, run as the non-root user baked into the image.

Rules: never put AWS keys in the repo or the image (use an IAM task role); keep `.env` out of git; the model bundle is a pickle, so only load bundles you built yourself.
Cost note: nothing here is required to pass the project; deploying has an ongoing cost.
