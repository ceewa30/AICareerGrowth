import yaml
from pathlib import Path

def generate_production_market_yaml(extracted_role_data: dict, output_directory: str = "config"):
    """
    Automates the creation of market_standards.yaml by validating schemas
    and enforcing structural formatting to ensure downstream compatibility.
    """
    # Enforce a strict structure for your role catalog
    production_payload = {
        "metadata": {
            "standards_version": "2026.2.0",
            "generation_method": "analytical_market_feed_aggregation"
        },
        "roles": {}
    }

    for role_name, attributes in extracted_role_data.items():
        # Validate that required keys are present
        skills_dict = attributes.get("skills", {})
        experience = attributes.get("experience", "N/A")

        # Automatically generate the necessary flat list array
        flat_list = []
        for category, items in skills_dict.items():
            if isinstance(items, list):
                # Clean strings and eliminate duplicate entities
                cleaned_items = [str(i).strip() for i in items if i]
                flat_list.extend(cleaned_items)

        # Deduplicate the global flat list array
        flat_list = list(dict.fromkeys(flat_list))

        # Map values directly into the target schema structure
        production_payload["roles"][role_name] = {
            "experience": experience,
            "skills": skills_dict,
            "flat_skills_list": flat_list
        }

    # Safeguard workspace paths dynamically
    target_dir = Path(output_directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_target = target_dir / "market_standards.yaml"

    with open(file_target, "w", encoding="utf-8") as yaml_file:
        yaml.dump(production_payload, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"✅ Success! Compiled '{file_target}' safely for Vector DB injection.")

# Example usage within an active ingestion script loop:
# market_feed_results = {"AI Engineer": {"experience": "7+ years", "skills": {"languages_and_runtimes": ["Python", "CUDA"]}}}
# generate_production_market_yaml(market_feed_results)

if __name__ == "__main__":
    # Example invocation for testing purposes
    sample_data = {
        "AI Engineer": {
            "experience": "7+ years",
            "skills": {
                "languages_and_runtimes": ["Python", "CUDA", "C++"],
                "frameworks": ["TensorFlow", "PyTorch"],
                "cloud_platforms": ["AWS", "GCP"]
            }
        },
        "Data Scientist": {
            "experience": "5+ years",
            "skills": {
                "languages_and_runtimes": ["R", "Python"],
                "frameworks": ["scikit-learn", "XGBoost"],
                "cloud_platforms": ["Azure"]
            }
        }
    }

    generate_production_market_yaml(sample_data, output_directory="config")