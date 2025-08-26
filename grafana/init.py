import os
import json
import requests
import time

from dotenv import load_dotenv

load_dotenv()

# Updated URLs for Docker environment
GRAFANA_URL = "http://grafana:3000"  # Use service name, not localhost

GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")

# Use correct environment variable names from your docker-compose
PG_HOST = os.getenv("POSTGRES_HOST", "postgres")  # Docker service name
PG_DB = os.getenv("POSTGRES_DB", "knowledge_base_assistant")
PG_USER = os.getenv("POSTGRES_USER", "user")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")


def wait_for_grafana(max_retries=30, delay=2):
    """Wait for Grafana to be ready before proceeding."""
    print(f"Waiting for Grafana to be ready at {GRAFANA_URL}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
            if response.status_code == 200:
                print("Grafana is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Attempt {attempt + 1}/{max_retries}: Grafana not ready, waiting {delay}s...")
        time.sleep(delay)
    
    print("Grafana failed to become ready within the timeout period")
    return False


def get_existing_service_account(auth):
    """Get existing service account by name."""
    try:
        sa_list_response = requests.get(f"{GRAFANA_URL}/api/serviceaccounts/search", auth=auth, timeout=10)
        if sa_list_response.status_code == 200:
            accounts = sa_list_response.json()["serviceAccounts"]
            automation_account = next((acc for acc in accounts if acc["name"] == "automation"), None)
            if automation_account:
                return automation_account["id"]
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting existing service account: {e}")
        return None


def create_service_account_and_token():
    """Create service account and token using the new Grafana API."""
    auth = (GRAFANA_USER, GRAFANA_PASSWORD)
    headers = {"Content-Type": "application/json"}
    
    try:
        # First, try to find existing service account
        print("Checking for existing service account...")
        sa_id = get_existing_service_account(auth)
        
        if sa_id:
            print(f"Found existing service account (ID: {sa_id})")
        else:
            # Step 1: Create service account
            service_account_payload = {
                "name": "automation",
                "displayName": "Automation Service Account",
                "role": "Admin"
            }
            
            print("Creating service account...")
            sa_response = requests.post(
                f"{GRAFANA_URL}/api/serviceaccounts", 
                auth=auth, 
                headers=headers, 
                json=service_account_payload,
                timeout=10
            )
            
            if sa_response.status_code == 201:
                service_account = sa_response.json()
                sa_id = service_account["id"]
                print(f"Service account created successfully (ID: {sa_id})")
            elif sa_response.status_code in [400, 409]:
                # Service account exists, try to get it
                print("Service account already exists, trying to find it...")
                sa_id = get_existing_service_account(auth)
                if not sa_id:
                    print("Could not find existing automation service account")
                    return None
            else:
                print(f"Failed to create service account. Status: {sa_response.status_code}")
                print(f"Response: {sa_response.text}")
                return None

        # Step 2: Check for existing tokens first
        print("Checking for existing tokens...")
        existing_tokens_response = requests.get(
            f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens",
            auth=auth,
            timeout=10
        )
        
        if existing_tokens_response.status_code == 200:
            existing_tokens = existing_tokens_response.json()
            # Delete existing tokens to avoid conflicts
            for token in existing_tokens:
                if token["name"] == "automation-token":
                    print(f"Deleting existing token: {token['id']}")
                    delete_response = requests.delete(
                        f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens/{token['id']}",
                        auth=auth,
                        timeout=10
                    )
                    if delete_response.status_code != 200:
                        print(f"Warning: Could not delete existing token {token['id']}")

        # Step 3: Create new token for the service account
        token_payload = {
            "name": "automation-token"
        }
        
        print("Creating service account token...")
        token_response = requests.post(
            f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens",
            auth=auth,
            headers=headers,
            json=token_payload,
            timeout=10
        )
        
        if token_response.status_code == 200:
            token_data = token_response.json()
            print("Service account token created successfully")
            return token_data["key"]
        else:
            print(f"Failed to create token. Status: {token_response.status_code}")
            print(f"Response: {token_response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def create_or_update_datasource(api_key):
    """Create or update PostgreSQL datasource in Grafana."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    datasource_payload = {
        "name": "PostgreSQL",
        "type": "postgres",
        "url": f"{PG_HOST}:{PG_PORT}",
        "access": "proxy",
        "user": PG_USER,
        "database": PG_DB,
        "basicAuth": False,
        "isDefault": True,
        "jsonData": {
            "sslmode": "disable", 
            "postgresVersion": 1300,
            "timescaledb": False
        },
        "secureJsonData": {"password": PG_PASSWORD},
    }

    print("Creating/updating PostgreSQL datasource...")
    print(f"Database URL: {PG_HOST}:{PG_PORT}")
    print(f"Database: {PG_DB}")
    print(f"User: {PG_USER}")

    try:
        # Check if datasource exists
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources/name/{datasource_payload['name']}",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            # Update existing datasource
            existing_datasource = response.json()
            datasource_id = existing_datasource["id"]
            print(f"Updating existing datasource (ID: {datasource_id})")
            
            response = requests.put(
                f"{GRAFANA_URL}/api/datasources/{datasource_id}",
                headers=headers,
                json=datasource_payload,
                timeout=10
            )
        else:
            # Create new datasource
            print("Creating new datasource")
            response = requests.post(
                f"{GRAFANA_URL}/api/datasources", 
                headers=headers, 
                json=datasource_payload,
                timeout=10
            )

        if response.status_code in [200, 201]:
            print("Datasource configured successfully")
            result = response.json()
            return result.get("datasource", {}).get("uid") or result.get("uid")
        else:
            print(f"Failed to configure datasource. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def create_dashboard(api_key, datasource_uid):
    """Create or update dashboard from JSON file."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    dashboard_file = os.path.join(os.path.dirname(__file__), "dashboard.json")
    
    try:
        with open(dashboard_file, "r") as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: {dashboard_file} not found.")
        print("Please ensure dashboard.json exists in the grafana/ directory")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing {dashboard_file}: {e}")
        return None

    print("Dashboard JSON loaded successfully")

    # Update datasource UID in all panels and targets
    panels_updated = 0
    
    def update_datasource_in_object(obj):
        nonlocal panels_updated
        if isinstance(obj, dict):
            if "datasource" in obj and isinstance(obj["datasource"], dict):
                obj["datasource"]["uid"] = datasource_uid
                panels_updated += 1
            if "targets" in obj and isinstance(obj["targets"], list):
                for target in obj["targets"]:
                    update_datasource_in_object(target)
            # Recursively check nested objects
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    update_datasource_in_object(value)
        elif isinstance(obj, list):
            for item in obj:
                update_datasource_in_object(item)

    update_datasource_in_object(dashboard_json)
    print(f"Updated datasource UID in {panels_updated} locations")

    # Get the target UID from environment variable
    target_dashboard_uid = os.getenv("GRAFANA_DASHBOARD_UID")
    dashboard_title = dashboard_json.get("title", "Knowledge Base Assistant Dashboard")
    
    # Check if dashboard exists by title or UID
    existing_dashboard = None
    try:
        if target_dashboard_uid:
            # Try to get dashboard by UID first
            uid_response = requests.get(
                f"{GRAFANA_URL}/api/dashboards/uid/{target_dashboard_uid}",
                headers=headers,
                timeout=10
            )
            if uid_response.status_code == 200:
                existing_dashboard = uid_response.json()["dashboard"]
                print(f"Found existing dashboard by UID: {target_dashboard_uid}")
    except requests.exceptions.RequestException:
        pass
    
    if not existing_dashboard:
        # Try to find by title
        try:
            search_response = requests.get(
                f"{GRAFANA_URL}/api/search?query={dashboard_title}",
                headers=headers,
                timeout=10
            )
            if search_response.status_code == 200:
                dashboards = search_response.json()
                for dash in dashboards:
                    if dash["title"] == dashboard_title:
                        existing_dashboard = dash
                        target_dashboard_uid = dash["uid"]
                        print(f"Found existing dashboard by title: {dashboard_title}")
                        break
        except requests.exceptions.RequestException:
            pass

    # Set up the dashboard with proper UID handling
    if existing_dashboard:
        # Update existing dashboard - preserve UID
        dashboard_json["uid"] = target_dashboard_uid
        dashboard_json["id"] = existing_dashboard.get("id")
        dashboard_json["version"] = existing_dashboard.get("version", 0) + 1
        print(f"Updating existing dashboard (UID: {target_dashboard_uid})")
    else:
        # Create new dashboard with specific UID if provided
        if target_dashboard_uid:
            dashboard_json["uid"] = target_dashboard_uid
        dashboard_json.pop("id", None)
        dashboard_json.pop("version", None)
        print("Creating new dashboard")

    dashboard_payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Updated by init script"
    }

    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db", 
            headers=headers, 
            json=dashboard_payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            dashboard_uid = result.get("uid")
            print(f"Dashboard {'updated' if existing_dashboard else 'created'} successfully!")
            print(f"Dashboard URL: {GRAFANA_URL}/d/{dashboard_uid}")
            
            # Update environment variable if needed
            if target_dashboard_uid != dashboard_uid:
                print(f"Note: Dashboard UID is {dashboard_uid}")
                print(f"You may want to update GRAFANA_DASHBOARD_UID in your .env file")
            
            return dashboard_uid
        else:
            print(f"Failed to create/update dashboard. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def main():
    """Main setup function."""
    print("Starting Grafana setup...")
    
    # Wait for Grafana to be ready
    if not wait_for_grafana():
        print("Grafana setup failed - service not ready")
        return False
    
    # Create service account and token (new API)
    api_key = create_service_account_and_token()
    if not api_key:
        print("Grafana setup failed - service account/token creation failed")
        return False
    
    # Setup datasource
    datasource_uid = create_or_update_datasource(api_key)
    if not datasource_uid:
        print("Grafana setup failed - datasource configuration failed")
        return False
    
    # Create dashboard
    dashboard_uid = create_dashboard(api_key, datasource_uid)
    if dashboard_uid:
        print("Grafana setup completed successfully!")
        print(f"Access your dashboard at: {GRAFANA_URL.replace('grafana:3000', 'localhost:3000')}/d/{dashboard_uid}")
        return True
    else:
        print("Grafana setup partially completed - dashboard creation failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)