#!/usr/bin/env bash
# Nexus — AWS infrastructure setup
# Run once before deploying. Requires AWS CLI configured with valid credentials.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
TABLE="nexus-incidents"
TOPIC="nexus-drift-check"

echo "==> Setting up Nexus infrastructure in $REGION"

# ── 1. DynamoDB table ───────────────────────────────────────────────────────
echo "--> Creating DynamoDB table: $TABLE"
aws dynamodb create-table \
  --table-name "$TABLE" \
  --attribute-definitions AttributeName=incident_id,AttributeType=S \
  --key-schema AttributeName=incident_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region "$REGION" \
  --output text --query 'TableDescription.TableStatus' 2>/dev/null \
  && echo "    Created" || echo "    Already exists — skipping"

# ── 2. SNS topic for background drift-check triggers ───────────────────────
echo "--> Creating SNS topic: $TOPIC"
TOPIC_ARN=$(aws sns create-topic \
  --name "$TOPIC" \
  --region "$REGION" \
  --output text --query 'TopicArn')
echo "    Topic ARN: $TOPIC_ARN"

# ── 3. EventBridge rule — trigger every 30 minutes ─────────────────────────
echo "--> Creating EventBridge scheduled rule (rate: 30 minutes)"
aws events put-rule \
  --name "nexus-scheduled-drift-check" \
  --schedule-expression "rate(30 minutes)" \
  --state ENABLED \
  --region "$REGION" \
  --output text --query 'RuleArn'

# ── 4. Wire EventBridge → SNS ───────────────────────────────────────────────
echo "--> Adding SNS as EventBridge target"
aws events put-targets \
  --rule "nexus-scheduled-drift-check" \
  --targets "Id=NexusSNS,Arn=$TOPIC_ARN" \
  --region "$REGION" \
  --output text --query 'FailedEntryCount'

echo ""
echo "==> Infrastructure ready."
echo "    Next: deploy the backend to App Runner, then subscribe the"
echo "    /api/sns/drift-check endpoint to the SNS topic:"
echo ""
echo "    aws sns subscribe \\"
echo "      --topic-arn $TOPIC_ARN \\"
echo "      --protocol https \\"
echo "      --notification-endpoint https://<your-app-runner-url>/api/sns/drift-check \\"
echo "      --region $REGION"
