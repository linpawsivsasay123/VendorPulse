from LLM.inference_generate import InferenceModel_Generate
from LLM.inference_similarity import InferenceModel_Similarity
import json
import re
import matplotlib.pyplot as plt

# System TAGS
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

# SYSTEM_MESSAGE_ZERO_SHOT = ""
SYSTEM_MESSAGE_ZERO_SHOT= f"""{B_SYS}
You are tasked with generating a maximum of 20 highly relevant questions to assess the cybersecurity risk of a vendor being considered for onboarding by our firm. These questions must effectively evaluate the vendor's cyber risk profile to minimize potential risks to our organization. 

Questions should span categories such as RBI Cybersecurity Framework, Cloud Security, NIST, Finance, ISO Standards, API Security, Web Redirections, Deep Tech, and general cybersecurity practices. 

Each question should be classified into one of the following categories:
1. Compliance-Based: Related to regulatory or standards compliance.
2. External Surface-Based Risk: Focused on software vulnerabilities (e.g., unpatched software, IP leaks).
3. Other: Questions outside the above categories, such as operational or procedural risks.

**Use the vendor metadata to generate even more relevant questions.**

### Important:
- **Return output strictly in JSON format.**
- **DO NOT** include any explanations, greetings, or introductory text.
- The JSON format must be:
```json
{{
  "questions": [
    {{"question": "your question here", "classification": "Compliance-Based"}},
    {{"question": "your question here", "classification": "External Surface-Based Risk"}}
  ]
}}
{E_SYS}"""

SYSTEM_MESSAGE_ONE_SHOT = f"""{B_SYS}
You are an intelligent assistant tasked with predicting the cybersecurity risk weights of a new vendor-related question using in-context learning from a provided dataset. The dataset and the new question share the same type, defined as follows:

- **Compliance-Based**: A single weight (0-10) reflecting regulatory or standards adherence.
- **External Surface-Based Risk**: Related to software vulnerabilities, with three weights:
  - **vulnerability_weight** (0-10): Severity of the software vulnerability type (e.g., unpatched software, injection flaws, ip leak).
  - **likelihood_probability** (0-1): Likelihood of exploitation within the next 30 days.
  - **asset_impact** (0-1): Proportion of assets impacted by the vulnerability.
- **Other**: A single weight (0-10) for operational or procedural risks.

Given a new question and its type (matching the dataset), analyze its content and intent, then predict its weight(s) by inferring from similar questions in the dataset using contextual reasoning. Ensure the output matches the dataset format and includes all fields present in the dataset entries. Output the result **strictly as a JSON object**—no plain text or other formats are allowed—with the following structure based on the type:

- For **type: 'compliance'** or **type: 'other'**:
  ```json
  {{
    "category": "<inferred from context>",
    "question": "<new question>",
    "type": "compliance" or "other",
    "weight": <predicted value (0-10)>
  }}
  - For **type: 'external_surface'** :
  ```json
  {{
  "category": "<inferred from context>",
  "question": "<new question>",
  "type": "external_surface",
  "weights_vulnerability_weight": <predicted value (0-10)>,
  "weights_likelihood_probability": <predicted value (0-1)>,
  "weights_asset_impact": <predicted value (0-1)>
  
  }}
  
{E_SYS}"""

class Pipeline:
    
    def __init__(self):
        self.model1 = InferenceModel_Generate()
        self.model2 = InferenceModel_Similarity()
        
    
    def get_vendor_metadata(self,vendor_metadata):
      content = (
          f"I am a vendor in the sector {vendor_metadata['vendor_metadata_sector']} and deal with "
          f"{vendor_metadata['vendor_metadata_domain']}. We are located in "
          f"{vendor_metadata['vendor_metadata_location']} and our employee strength is "
          f"{vendor_metadata['vendor_metadata_employee_strength']}.\n" 
      )
      return content

    def extract_json_from_response(self,response):
        # This matches the LAST JSON object that begins with {"questions":
        matches = re.findall(r'\{\s*"questions"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
        if matches:
            json_str = matches[-1]  # Take the last JSON match (actual output)
            try:
                parsed_json = json.loads(json_str)
                return parsed_json
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                return None
        else:
            print("No JSON found in the response.")
            return None

    def get_questions(self,metadata):

      content = self.get_vendor_metadata(metadata)
      payload = {
          "messages": [
              {
                  "role": "system",
                  "content": SYSTEM_MESSAGE_ZERO_SHOT
              },
              {
                  "role": "user",
                  "content": f"{B_INST}{content}{E_INST}"
              }
          ]
      }
      
    #   print(json.dumps(payload,indent=2))
      
      response = self.model1.inference(payload["messages"])
      json_response = self.extract_json_from_response(response)
      return json_response
      
    def match_questions(self,question):
      
      Q = question.get("question")
      Q_Tag = question.get("classification")
      if Q_Tag == "External Surface-Based Risk":
        Q_Tag = "external_surface"
      elif Q_Tag == "Compliance-Based":
        Q_Tag = "compliance"
      else :
        Q_Tag = "other"
        
      similar_Q = self.model2.most_similar(Q,Q_Tag)
      return similar_Q
      
    def extract_last_json_object(self,text: str) -> dict:
        """
        Extract the last JSON object from the input text string.
        Returns the JSON object as a Python dict, or {} if extraction fails.
        """
        matches = re.findall(r'\{(?:[^{}]|(?:(?<=\{)[^{}]*?(?=\})))+\}', text, re.DOTALL)

        for match in reversed(matches):
            try:
                obj = json.loads(match)
                return obj
            except json.JSONDecodeError:
                continue
        print("No valid JSON object found.")
        return {}
      
    def format_dict_to_string(self,data, indent_level=0, indent_size=2):
        """Recursively format a dictionary or list into a readable string."""
        indent = " " * (indent_level * indent_size)
        
        if isinstance(data, dict):
            lines = [f"{indent}{{"]
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{indent}  \"{key}\": {self.format_dict_to_string(value, indent_level + 1, indent_size)},")
                else:
                    value_str = json.dumps(value)  # Handles strings, numbers, etc.
                    lines.append(f"{indent}  \"{key}\": {value_str},")
            lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma
            lines.append(f"{indent}}}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = [f"{indent}["]
            for item in data:
                lines.append(f"{self.format_dict_to_string(item, indent_level + 1, indent_size)},")
            lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma
            lines.append(f"{indent}]")
            return "\n".join(lines)
        else:
            return json.dumps(data)
    
    def weights_prediction(self, data):
        all_questions = data.get("questions", [])
        final_question_with_their_weights = []

        for question in all_questions:
            similar_Q = self.match_questions(question)

            Q = question.get("question")
            Q_Tag = question.get("classification")
            if Q_Tag == "External Surface-Based Risk":
                Q_Tag = "external_surface"
            elif Q_Tag == "Compliance-Based":
                Q_Tag = "compliance"
            else:
                Q_Tag = "other"

            # Format dataset_content based on Q_Tag
            formatted_similar_Q = []  # Initialize the list
            for q in similar_Q:
                if Q_Tag in ["compliance", "other"]:
                    formatted_q = {
                        "category": q.get("category"),
                        "question": q.get("question"),
                        "type": q.get("type", Q_Tag),
                        "weight": q.get("weight", q.get("weights", 0))  # Default to 0 if missing
                    }
                elif Q_Tag == "external_surface":
                    formatted_q = {
                        "category": q.get("category"),
                        "question": q.get("question"),
                        "type": q.get("type", Q_Tag),
                        "weights_vulnerability_weight" : q["weights_vulnerability_weight"],
                        "weights_likelihood_probability" : q["weights_likelihood_probability"],
                        "weights_asset_impact" : q["weights_asset_impact"]
                    }
                formatted_similar_Q.append(formatted_q)

            dataset_content = self.format_dict_to_string(formatted_similar_Q)

            new_question_content = self.format_dict_to_string({
                "question": Q,
                "type": Q_Tag
            })

            content = f"""
            Given the following dataset of similar questions and a new question, predict the weights for the new question:
            
            **Dataset:**
            {dataset_content}
            
            **New Question:**
            {new_question_content}
            """

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE_ONE_SHOT
                    },
                    {
                        "role": "user",
                        "content": f"{B_INST}{content}{E_INST}"
                    }
                ]
            }
            # print(json.dumps(payload,indent=2))
            response = self.model1.inference(payload["messages"])
            item = self.extract_last_json_object(response)
            item["similar"] = formatted_similar_Q
            final_question_with_their_weights.append(item)

        return final_question_with_their_weights


          
# class Error_calculation:
#     def __init__(self):
        

def check_keys(item):
    keys_to_check = ["weights_vulnerability_weight", "weights_likelihood_probability", "weights_asset_impact"]
    external_check = 0
    compliance_others_check = 0

    if all(k in item for k in keys_to_check):
        external_check += 1

    if external_check == 1:
        try:
            v_wt = float(item.get("weights_vulnerability_weight"))
            li_pr = float(item.get("weights_likelihood_probability"))
            as_im = float(item.get("weights_asset_impact"))
        except (TypeError, ValueError):
            # Any of the values are None or not convertible to float
            v_wt = li_pr = as_im = None

        if v_wt is not None and 0 <= v_wt <= 10:
            external_check += 1
        if li_pr is not None and 0 <= li_pr <= 10:
            external_check += 1
        if as_im is not None and 0.0 <= as_im <= 1.0:
            external_check += 1

    if item.get("question") != "<new question>":
        compliance_others_check += 1

    try:
        wt = float(item.get("weight"))
    except (TypeError, ValueError):
        wt = None

    if wt is not None and 0 <= wt <= 10:
        compliance_others_check += 1

    return compliance_others_check == 2 and external_check == 4

    
if __name__ == "__main__":
    obj = Pipeline()

    with open("./datasets/data.json" , "r") as f:
        vendor_metadata_list = json.load(f)
    
    output_list = []
    for vendor_metadata in vendor_metadata_list:
        
        all_questions = obj.get_questions(vendor_metadata)
        if all_questions == None:
            continue
        final_question_with_their_weights = obj.weights_prediction(all_questions)
        item = vendor_metadata
        item["questions"] = final_question_with_their_weights
        # print(json.dumps(item,indent=4))
        output_list.append(item)
        # break
    with open("./predicted_weights.json", "w") as f:
        json.dump(output_list, f, indent=4)
        
    
