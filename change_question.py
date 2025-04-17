import json 

with open("./datasets/questions_with_weights.json", "r") as f:
    input_data = json.load(f)

output_list = []
for item in input_data:
    Qtype = item.get("type")
    if Qtype == "external_surface":
        transformed = {
            "category" : item["category"],
            "question" : item["question"],
            "type" : item["type"]
        }
        transformed["weights_vulnerability_weight"] = item["weights"]["vulnerability_weight"]
        transformed["weights_likelihood_probability"] = item["weights"]["likelihood_probability"]
        transformed["weights_asset_impact"] = item["weights"]["asset_impact"]
        transformed["weights_likelihood_probability"] = transformed["weights_likelihood_probability"] /10.0
        output_list.append(transformed)
    else :
        output_list.append(item)
with open("./datasets/transformed_questions.json", "w") as f:
    json.dump(output_list, f, indent=4)