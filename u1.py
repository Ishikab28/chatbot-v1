# prompt_utils.py

class InsurancePromptTemplates:
    """Class containing all prompt templates for insurance policy review"""
    
    # Classification keywords
    COVERED_KEYWORDS = [
        "reasonable and necessary",
        "considered medically reasonable and necessary",
        "considered reasonable and necessary for coverage",
        "covered indications",
        "medically necessary",
        "Criteria for coverage",
        "limited to the following criteria",
        "limited coverage",
        "timely, objective, and actionable information"
    ]
    
    
    NOT_COVERED_KEYWORDS = [
        "non-coverage",
        "non-covered",
        "non-covered indications",
        "LIMITATIONS",
        "investigational and not covered"
    ]


    
    @classmethod
    def get_base_prompt(cls):
        """Base system prompt for the insurance policy reviewer"""
        return f"""
        You are an expert insurance policy reviewer with 20 years of experience analyzing 
        medical coverage policies. Your task is to classify whether a given CPT code is 
        covered or not covered based on the policy document text provided.

        You MUST follow these guidelines:
        1. Analyze the document carefully and completely before making any determination
        2. Only consider official policy language - not examples or commentary
        3. Look for specific coverage language patterns
        4. When in doubt, classify as "Not Covered"
        5. Always provide your reasoning and document references

        Coverage Indicators:
        Covered: {', '.join(cls.COVERED_KEYWORDS)}
        Not Covered: {', '.join(cls.NOT_COVERED_KEYWORDS)}
        """

    @classmethod
    def get_cpt_coverage_prompt(cls, cpt_code: str, context: str):
        """Prompt template for CPT code coverage analysis"""
        return f"""
        {cls.get_base_prompt()}
        
        Analyze the following policy document excerpts for CPT code {cpt_code}:
        {context}

        Provide your response in EXACTLY this format:
        🔹 **CPT Code**: {cpt_code}
        🔹 **Coverage Status**: [Covered/Not Covered/Undetermined]
        🔹 **Key Phrases Found**: [List exact phrases that informed your decision]
        🔹 **Limitations**: [Any restrictions or special conditions]
        🔹 **Document Reference**: [Page/Section numbers if available]
        🔹 **Confidence Level**: [High/Medium/Low based on evidence found]
        🔹 **Additional Notes**: [Any other relevant information]
        """

    @classmethod
    def get_general_question_prompt(cls, question: str, context: str):
        """Prompt template for general insurance policy questions"""
        return f"""
        {cls.get_base_prompt()}
        
        You are now answering a general question about the insurance policy:
        Question: {question}

        Policy Context:
        {context}

        Provide your response in EXACTLY this format:
        🔹 **Question**: {question}
        🔹 **Answer**: [Clear, concise answer based on policy language]
        🔹 **Key Policy References**: [Exact phrases/sections that support your answer]
        🔹 **Confidence Level**: [High/Medium/Low based on evidence found]
        🔹 **Additional Notes**: [Any caveats or important considerations]
        """

    @classmethod
    def get_coverage_decision_rules(cls):
        """Returns the detailed coverage decision rules for programmatic use"""
        return {
            "covered_keywords": cls.COVERED_KEYWORDS,
            "not_covered_keywords": cls.NOT_COVERED_KEYWORDS,
            "rules": [
                {
                    "condition": "Any covered keyword present without contradictory language",
                    "decision": "Covered",
                    "confidence": "High"
                },
                {
                    "condition": "Covered and not-covered keywords both present",
                    "decision": "Covered with Limitations",
                    "confidence": "Medium"
                },
                {
                    "condition": "Only not-covered keywords present",
                    "decision": "Not Covered",
                    "confidence": "High"
                },
                {
                    "condition": "No relevant keywords found",
                    "decision": "Not Covered",
                    "confidence": "Low"
                },
                {
                    "condition": "LIMITATIONS heading present without coverage language",
                    "decision": "Not Covered",
                    "confidence": "Medium"
                }
            ]
        }


class InsuranceResponseFormatter:
    """Utility class for formatting LLM responses consistently"""
    
    @staticmethod
    def format_cpt_response(response_text: str):
        """Formats the CPT coverage response into a structured dict"""
        sections = {
            "CPT Code": "",
            "Coverage Status": "",
            "Key Phrases Found": "",
            "Limitations": "",
            "Document Reference": "",
            "Confidence Level": "",
            "Additional Notes": ""
        }
        
        current_section = None
        for line in response_text.split('\n'):
            if line.strip().startswith('🔹'):
                # New section found
                section_match = line.split('**')[1]
                if section_match in sections:
                    current_section = section_match
                    sections[current_section] = line.split(':', 1)[1].strip()
            elif current_section:
                # Continuation of current section
                sections[current_section] += "\n" + line.strip()
        
        return sections
    
    @staticmethod
    def format_general_response(response_text: str):
        """Formats the general question response into a structured dict"""
        sections = {
            "Question": "",
            "Answer": "",
            "Key Policy References": "",
            "Confidence Level": "",
            "Additional Notes": ""
        }
        
        current_section = None
        for line in response_text.split('\n'):
            if line.strip().startswith('🔹'):
                # New section found
                section_match = line.split('**')[1]
                if section_match in sections:
                    current_section = section_match
                    sections[current_section] = line.split(':', 1)[1].strip()
            elif current_section:
                # Continuation of current section
                sections[current_section] += "\n" + line.strip()
        
        return sections