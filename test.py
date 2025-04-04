from LLM.inference_generate import InferenceModel_Generate
import json
import re
obj = InferenceModel_Generate()



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


SYSTEM_MESSAGE_ZERO_SHOT = f"""{B_SYS}
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
You are an intelligent assistant tasked with predicting the cybersecurity risk weights of a new vendor-related question using in-context learning from a provided dataset. The dataset and the new question share the same classification, defined as: 
(1) **Compliance-Based**: Single weight (0-10) reflecting regulatory or standards adherence.
(2) **External Surface-Based Risk**: Related to software vulnerabilities, with three weights: 
   - **vulnerability_weight** (0-10): Severity of the software vulnerability type (e.g., unpatched software, injection flaws).
   - **likelihood_probability** (0-10): Likelihood of exploitation within the next 30 days.
   - **asset_impact** (0-1): Proportion of assets impacted by the vulnerability.
(3) **Other**: Single weight (0-10) for operational or procedural risks.
Given a new question and its classification (matching the dataset), analyze its content and intent, then predict its weight(s) by inferring from similar questions in the dataset. Use averaging or contextual reasoning to determine the weights, ensuring the output matches the dataset format (single weight for Compliance-Based/Other, three weights for External Surface-Based Risk). Output only the predicted result as a JSON object with the category (inferred from context), classification (as provided), and weight(s), in the same structure as the dataset entries.
{E_SYS}"""


vendor_metadata = {
    "vendor_metadata_sector": "Logistics",
    "vendor_metadata_domain": "Infrastructure",
    "vendor_metadata_location": "Hyderabad",
    "vendor_metadata_employee_strength": 937
}

content = (
    f"I am a vendor in the sector {vendor_metadata['vendor_metadata_sector']} and deal with "
    f"{vendor_metadata['vendor_metadata_domain']}. We are located in "
    f"{vendor_metadata['vendor_metadata_location']} and our employee strength is "
    f"{vendor_metadata['vendor_metadata_employee_strength']}.\n" 
)
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
def extract_json_from_response(response):
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


response = obj.inference(payload["messages"])





# print(response)
json_response = extract_json_from_response(response)



print(json.dumps(json_response,indent=2))