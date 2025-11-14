Here's how simple it is to train a specialized model with distil labs (get [`YOUR_API_KEY`](https://docs.distillabs.ai/getting-started/authentication)):

```python maxLines=100
import json
import requests

# Register a new model
auth_header = {"Authorization": f"Bearer {distil_bearer_token()}"}
response = requests.post(
    "https://api.distillabs.ai/models",
    data=json.dumps({"name": "testmodel-123"}),
    headers={"Content-Type": "application/json", **auth_header},
)
model_id = response.json()["id"]

# Upload your task description and examples
data = {
    "job_description": {"type": "json", "content": open("data/job_description.json").read()},
    "train_data": {"type": "jsonl", "content": open("data/train.jsonl").read()},
    "test_data": {"type": "jsonl", "content": open("data/test.jsonl").read()},
    "config": {"type": "yaml", "content": open("data/config.yaml").read()},
}
response = requests.post(
    f"https://api.distillabs.ai/models/{model_id}/uploads",
    data=json.dumps(data),
    headers={"Content-Type": "application/json", **auth_header},
)
upload_id = response.json()["id"]

# Start SLM training
response = requests.post(
    f"https://api.distillabs.ai/models/{model_id}/training",
    data=json.dumps({"upload_id": upload_id}),
    headers={"Content-Type": "application/json", **auth_header},
)
slm_training_job_id = response.json()["id"]

# When training completes, get your model download link
response = requests.get(
    f"https://api.distillabs.ai/trainings/{slm_training_job_id}/model",
    headers=auth_header,
)
print(response.json())
```