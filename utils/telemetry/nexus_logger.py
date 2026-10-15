import requests
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


class ExperimentLogger:
    def __init__(self, token: str, database_id: str):
        # Bulletproof sanitization: removes whitespace, \r, and accidental quotes
        self.token = token.strip().replace('"', "").replace("'", "")
        self.database_id = database_id.strip().replace('"', "").replace("'", "")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def log_training_run(
        self, run_id, run_type, hyperparams, cumulative_reward, commit_hash
    ):
        url = "https://api.notion.com/v1/pages"
        payload = {
            "parent": {"database_id": self.database_id},
            "properties": {
                "Run ID": {"title": [{"text": {"content": run_id}}]},
                "Run Type": {"select": {"name": run_type}},
                "Hyperparameters": {"rich_text": [{"text": {"content": hyperparams}}]},
                "Cumulative Reward": {"number": cumulative_reward},
                "Commit Hash": {"rich_text": [{"text": {"content": commit_hash}}]},
                "Date Executed": {
                    "date": {"start": datetime.now(timezone.utc).isoformat()}
                },
            },
        }

        response = requests.post(url, headers=self.headers, data=json.dumps(payload))

        if response.status_code == 200:
            print(f"✅ Successfully logged {run_id} to Notion.")
            return response.json()["url"]
        else:
            print(f"❌ Failed to log: {response.status_code}")
            print(response.text)
            return None


raw_token = os.getenv("NOTION_TOKEN")
raw_db_id = os.getenv("NOTION_DB_ID")

if not raw_token or not raw_db_id:
    print("❌ Error: Missing credentials in .env file.")
else:
    # Diagnostic Check
    clean_token = raw_token.strip().replace('"', "").replace("'", "")
    print(f"DEBUG: Token length is {len(clean_token)} characters. (It should be 50)")

    logger = ExperimentLogger(raw_token, raw_db_id)
    logger.log_training_run(
        run_id="DDQN-TechVol-v1.4",
        run_type="RL Training (Colab)",
        hyperparams="alpha=0.001, gamma=0.99, batch=64, buffer=100k",
        cumulative_reward=428.15,
        commit_hash="7baac39",
    )
