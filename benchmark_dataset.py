import json
from statistics import mean
import matplotlib.pyplot as plt
import uuid

def process_data(data, dataset_name, output_filename):
    question_lengths = {"compliance": [], "external_surface": [], "other": []}
    weights = {
        "compliance": [],
        "external_surface": [],
        "other": []
    }

    # Process each question
    for item in data:
        question = item["question"]
        q_type = item["type"]
        question_length = len(question)
        
        # Store question length by type
        question_lengths[q_type].append(question_length)
        
        # Store weights by type
        if q_type == "compliance" or q_type == "other":
            weight = item.get("weight", 0)
            weights[q_type].append(weight)
        elif q_type == "external_surface":
            weight_set = [
                item.get("weights_vulnerability_weight", 0),
                item.get("weights_likelihood_probability", 0),
                item.get("weights_asset_impact", 0)
            ]
            weights[q_type].append(weight_set)

    # Calculate averages
    results = {
        "average_question_length_by_type": {},
        "average_weights_by_type": {},
        "overall_average_question_length": 0
    }

    # Average question length by type
    for q_type, lengths in question_lengths.items():
        results["average_question_length_by_type"][q_type] = round(mean(lengths), 2) if lengths else 0

    # Average weights by type
    for q_type in weights:
        if q_type == "external_surface" and weights[q_type]:
            vuln_weights = [w[0] for w in weights[q_type]]
            likelihood_probs = [w[1] for w in weights[q_type]]
            asset_impacts = [w[2] for w in weights[q_type]]
            results["average_weights_by_type"][q_type] = {
                "vulnerability_weight": round(mean(vuln_weights), 2),
                "likelihood_probability": round(mean(likelihood_probs), 2),
                "asset_impact": round(mean(asset_impacts), 2)
            }
        elif weights[q_type]:
            results["average_weights_by_type"][q_type] = {
                "average_weight": round(mean(weights[q_type]), 2)
            }
        else:
            results["average_weights_by_type"][q_type] = {
                "average_weight": 0
            }

    # Overall average question length
    all_lengths = [length for lengths in question_lengths.values() for length in lengths]
    results["overall_average_question_length"] = round(mean(all_lengths), 2) if all_lengths else 0

    # Print JSON results
    print(f"Results for {dataset_name}:")
    print(json.dumps(results, indent=2))

    # Prepare data for the table
    table_data = [
        ["Question Type", "Avg. Question Length", "Weight Metrics"]
    ]
    
    for q_type in ["compliance", "external_surface", "other"]:
        avg_length = results["average_question_length_by_type"][q_type]
        weight_info = results["average_weights_by_type"][q_type]
        
        if q_type == "external_surface":
            weight_str = (
                f"Vulnerability: {weight_info['vulnerability_weight']}\n"
                f"Likelihood: {weight_info['likelihood_probability']}\n"
                f"Asset Impact: {weight_info['asset_impact']}"
            )
        else:
            weight_str = f"Average: {weight_info['average_weight']}"
        
        table_data.append([q_type.capitalize(), str(avg_length), weight_str])
    
    # Add overall average question length
    table_data.append([
        "Overall",
        str(results["overall_average_question_length"]),
        "-"
    ])

    # Create the table using matplotlib
    plt.figure(figsize=(12, 7))  # Increased figure size for better spacing
    ax = plt.gca()
    ax.axis('off')  # Hide axes
    
    # Create table
    the_table = ax.table(
        cellText=table_data,
        colLabels=None,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.2, 0.55],  # Adjusted column widths
        bbox=[0.05, 0.05, 0.9, 0.9]  # Slightly smaller bbox for margins
    )
    
    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)  # Slightly larger font for readability
    the_table.scale(1.3, 1.5)  # Increased scaling for better cell padding
    
    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2F4F4F')  # Dark slate gray header
            cell.set_height(0.1)  # Taller header row
        else:
            cell.set_facecolor('#F5F6F5' if row % 2 == 0 else '#E8ECEF')  # Subtle alternating colors
            cell.set_height(0.15)  # Taller rows for content
        cell.set_edgecolor('#A9A9A9')  # Darker gray borders for clarity
        if col == 2 and row > 0:  # Wrap text in Weight Metrics column
            cell.set_text_props(ha='left', va='center', wrap=True)
    
    # Add title
    plt.title(f"{dataset_name} Analysis", fontsize=16, weight='bold', pad=30, color='#333333')
    
    # Save as JPG
    plt.savefig(output_filename, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

def original_questions():
    with open("./datasets/transformed_questions.json", "r") as f:
        dataset = json.load(f)
    process_data(dataset, "Original Dataset", "original_dataset_table.jpg")

def check_valid_question(item):
    """Validate a question based on category, type, and weight fields."""
    # Check common fields: category, type, question
    if not isinstance(item.get('category'), str) or not item['category'].strip():
        return False, "Invalid or missing category"
    if not isinstance(item.get('type'), str) or not item['type'].strip():
        return False, "Invalid or missing type"
    if not isinstance(item.get('question'), str) or not item['question'].strip():
        return False, "Invalid or missing question"

    # Validate type-specific fields
    question_type = item['type'].lower()
    if question_type in ['compliance', 'other']:
        # Check weight field
        if 'weight' not in item or not isinstance(item['weight'], (int, float)):
            return False, "Missing or invalid weight for compliance/other type"
        if not (0 <= item['weight'] <= 10):
            return False, "Weight out of range (0-10) for compliance/other type"
    elif question_type == 'external_surface':
        # Check weights_vulnerability_weight
        if 'weights_vulnerability_weight' not in item or not isinstance(item['weights_vulnerability_weight'], (int, float)):
            return False, "Missing or invalid weights_vulnerability_weight for external_surface type"
        if not (0 <= item['weights_vulnerability_weight'] <= 10):
            return False, "weights_vulnerability_weight out of range (0-10) for external_surface type"
        # Check weights_likelihood_probability
        if 'weights_likelihood_probability' not in item or not isinstance(item['weights_likelihood_probability'], (int, float)):
            return False, "Missing or invalid weights_likelihood_probability for external_surface type"
        if not (0 <= item['weights_likelihood_probability'] <= 1):
            return False, "weights_likelihood_probability out of range (0-1) for external_surface type"
        # Check weights_asset_impact
        if 'weights_asset_impact' not in item or not isinstance(item['weights_asset_impact'], (int, float)):
            return False, "Missing or invalid weights_asset_impact for external_surface type"
        if not (0 <= item['weights_asset_impact'] <= 1):
            return False, "weights_asset_impact out of range (0-1) for external_surface type"
    else:
        return False, f"Unknown question type: {question_type}"

    return True, "Valid question"

def predicted_questions():
    with open("./predicted_weights.json", "r") as f:
        dataset = json.load(f)
    
    valid_dataset = []
    
    for vendor in dataset:
        all_questions = vendor.get('questions')
        for item in all_questions:
            valid, temp = check_valid_question(item)
            if valid:
                new_item = item.copy()
                del new_item['similar']
                valid_dataset.append(new_item)
    process_data(valid_dataset, "Predicted Dataset", "predicted_dataset_table.jpg")

# Run the functions
original_questions()
print("---------------------------")
predicted_questions()