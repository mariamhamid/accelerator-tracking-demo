export DAGSHUB_USER= 'MariamHamid'
export DAGSHUB_TOKEN='xxx'

#!/bin/bash

export DAGSHUB_USER="MariamHamid"
export DAGSHUB_TOKEN="YOUR_DAGSHUB_TOKEN"

docker run -it --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="https://dagshub.com/MariamHamid/accelerator-tracking-demo.mlflow" \
  -e MLFLOW_TRACKING_USERNAME="$DAGSHUB_USER" \
  -e MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN" \
  -e DAGSHUB_OWNER="MariamHamid" \
  -e DAGSHUB_REPO="accelerator-tracking-demo" \
  tracking-demos:latest
