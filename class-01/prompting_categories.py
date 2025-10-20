"""
Comprehensive Prompting Techniques for AI Models
Contains various categories of prompting strategies for testing
"""

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI()

# ==================== PROMPTING CATEGORIES ====================

# 1. ZERO-SHOT PROMPTING
# Direct instruction without examples
ZERO_SHOT_PROMPT = {
    "system": "You are a helpful assistant.",
    "user": "Translate 'Hello, how are you?' to French."
}

# 2. ONE-SHOT PROMPTING
# Single example provided
ONE_SHOT_PROMPT = {
    "system": "You are a great philosopher. Only push the user to think deeper.",
    "user": """Example:
        user: "Good morning"
        assistant: "Next morning maybe you can't see, life is unpredictable."

    Now let's move with this:
        user: "How is life"
        assistant:"""
}

# 3. FEW-SHOT PROMPTING
# Multiple examples provided
FEW_SHOT_PROMPT = {
    "system": "You are a sentiment classifier.",
    "user": """Examples:
Text: "I love this product!" → Sentiment: Positive
Text: "This is terrible." → Sentiment: Negative
Text: "It's okay, nothing special." → Sentiment: Neutral

Text: "Amazing quality and fast delivery!" → Sentiment:"""
}

# 4. CHAIN-OF-THOUGHT (COT) PROMPTING
# Step-by-step reasoning
COT_PROMPT = {
    "system": "You are a math tutor. Show your step-by-step reasoning.",
    "user": """Problem: A store has 15 apples. They sell 8 apples in the morning and 3 apples in the afternoon. How many apples are left?

Let me think step by step:"""
}

# 5. TREE-OF-THOUGHT PROMPTING
# Multiple reasoning paths
TOT_PROMPT = {
    "system": "You are a strategic problem solver.",
    "user": """Problem: How can we increase customer retention for an e-commerce website?

Consider 3 different approaches:
Approach 1: [Focus on user experience]
Approach 2: [Focus on loyalty programs]
Approach 3: [Focus on customer service]

Evaluate each approach and recommend the best strategy."""
}

# 6. ROLE-BASED PROMPTING
# Specific expert persona
ROLE_BASED_PROMPT = {
    "system": "You are a senior software architect with 15 years of experience in building scalable web applications.",
    "user": "Design a microservices architecture for a social media platform that needs to handle 1 million users."
}

# 7. INSTRUCTION-FOLLOWING PROMPTING
# Clear, specific instructions
INSTRUCTION_PROMPT = {
    "system": "Follow these instructions exactly:",
    "user": """Instructions:
1. Read the following text
2. Extract all email addresses
3. Format them as a numbered list
4. Count the total number

Text: "Contact john@example.com or mary.smith@company.org for support. You can also reach admin@website.net for technical issues."
"""
}

# 8. CONTEXTUAL PROMPTING
# Rich background context
CONTEXTUAL_PROMPT = {
    "system": "You are a financial advisor in 2024. The economy is recovering from inflation, interest rates are at 5.5%, and tech stocks have been volatile.",
    "user": "Should a 30-year-old invest in tech stocks or bonds right now?"
}

# 9. CONSTRAINT-BASED PROMPTING
# Specific limitations or requirements
CONSTRAINT_PROMPT = {
    "system": "You are a creative writer.",
    "user": """Write a story with these constraints:
- Exactly 100 words
- Must include: a cat, a storm, and a secret
- Setting: Victorian London
- Genre: Mystery"""
}

# 10. COMPARATIVE PROMPTING
# Comparing options or alternatives
COMPARATIVE_PROMPT = {
    "system": "You are a technology analyst.",
    "user": """Compare React vs Vue.js for building a large-scale web application:

Consider:
- Performance
- Learning curve
- Community support
- Ecosystem
- Long-term maintainability

Provide a recommendation with reasoning."""
}

# 11. SOCRATIC PROMPTING
# Question-based learning approach
SOCRATIC_PROMPT = {
    "system": "You are a Socratic tutor. Guide learning through questions.",
    "user": """I want to understand how machine learning works. Instead of explaining directly, ask me questions that will help me discover the concepts myself.

Start with: What do you think 'learning' means when we talk about machines?"""
}

# 12. METACOGNITIVE PROMPTING
# Thinking about thinking
METACOGNITIVE_PROMPT = {
    "system": "You are a reflective problem solver.",
    "user": """Problem: Design a mobile app for food delivery.

Before solving:
1. What assumptions am I making?
2. What information do I need?
3. What could go wrong with my approach?
4. How will I know if my solution is good?

Then provide your solution."""
}

# 13. EMOTIONAL PROMPTING
# Appeals to emotional intelligence
EMOTIONAL_PROMPT = {
    "system": "You are an empathetic counselor.",
    "user": """A team member seems disengaged during meetings and hasn't been contributing ideas lately. They used to be very active and creative. How would you approach this situation with sensitivity and care?"""
}

# 14. ITERATIVE PROMPTING
# Building on previous responses
ITERATIVE_PROMPT = {
    "system": "You are a product designer.",
    "user": """Design a mobile app feature for expense tracking.

First iteration: Basic concept
[Wait for response, then continue with:]

Second iteration: Improve the UI based on accessibility principles
[Wait for response, then continue with:]

Third iteration: Add gamification elements"""
}

# 15. NEGATIVE PROMPTING
# Explicitly stating what NOT to do
NEGATIVE_PROMPT = {
    "system": "You are a professional email writer.",
    "user": """Write a follow-up email to a client about a delayed project.

Do NOT:
- Make excuses
- Blame team members
- Use overly casual language
- Ignore the delay

DO:
- Take responsibility
- Provide clear next steps
- Maintain professionalism
- Offer solutions"""
}

# ==================== TESTING FUNCTIONS ====================

def test_prompting_technique(prompt_dict, technique_name):
    """Test a specific prompting technique"""
    print(f"\n{'='*50}")
    print(f"TESTING: {technique_name}")
    print(f"{'='*50}")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_dict["system"]},
            {"role": "user", "content": prompt_dict["user"]}
        ],
        temperature=0.7,
        max_tokens=500
    )

    print(f"System: {prompt_dict['system']}")
    print(f"User: {prompt_dict['user']}")
    print(f"\nResponse:\n{response.choices[0].message.content}")
    print("\n" + "="*50)

def run_all_tests():
    """Run all prompting technique tests"""
    techniques = [
        (ZERO_SHOT_PROMPT, "Zero-Shot Prompting"),
        (ONE_SHOT_PROMPT, "One-Shot Prompting"),
        (FEW_SHOT_PROMPT, "Few-Shot Prompting"),
        (COT_PROMPT, "Chain-of-Thought Prompting"),
        (TOT_PROMPT, "Tree-of-Thought Prompting"),
        (ROLE_BASED_PROMPT, "Role-Based Prompting"),
        (INSTRUCTION_PROMPT, "Instruction-Following Prompting"),
        (CONTEXTUAL_PROMPT, "Contextual Prompting"),
        (CONSTRAINT_PROMPT, "Constraint-Based Prompting"),
        (COMPARATIVE_PROMPT, "Comparative Prompting"),
        (SOCRATIC_PROMPT, "Socratic Prompting"),
        (METACOGNITIVE_PROMPT, "Metacognitive Prompting"),
        (EMOTIONAL_PROMPT, "Emotional Prompting"),
        (NEGATIVE_PROMPT, "Negative Prompting"),
    ]

    for prompt_dict, name in techniques:
        test_prompting_technique(prompt_dict, name)
        input("\nPress Enter to continue to next technique...")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("AI Prompting Techniques Testing Suite")
    print("Choose an option:")
    print("1. Test specific technique")
    print("2. Run all techniques")

    choice = input("\nEnter your choice (1 or 2): ")

    if choice == "1":
        print("\nAvailable techniques:")
        techniques = [
            "Zero-Shot", "One-Shot", "Few-Shot", "Chain-of-Thought",
            "Tree-of-Thought", "Role-Based", "Instruction-Following",
            "Contextual", "Constraint-Based", "Comparative",
            "Socratic", "Metacognitive", "Emotional", "Negative"
        ]

        for i, tech in enumerate(techniques, 1):
            print(f"{i}. {tech}")

        tech_choice = int(input("\nSelect technique number: ")) - 1

        if 0 <= tech_choice < len(techniques):
            technique_map = [
                (ZERO_SHOT_PROMPT, "Zero-Shot Prompting"),
                (ONE_SHOT_PROMPT, "One-Shot Prompting"),
                (FEW_SHOT_PROMPT, "Few-Shot Prompting"),
                (COT_PROMPT, "Chain-of-Thought Prompting"),
                (TOT_PROMPT, "Tree-of-Thought Prompting"),
                (ROLE_BASED_PROMPT, "Role-Based Prompting"),
                (INSTRUCTION_PROMPT, "Instruction-Following Prompting"),
                (CONTEXTUAL_PROMPT, "Contextual Prompting"),
                (CONSTRAINT_PROMPT, "Constraint-Based Prompting"),
                (COMPARATIVE_PROMPT, "Comparative Prompting"),
                (SOCRATIC_PROMPT, "Socratic Prompting"),
                (METACOGNITIVE_PROMPT, "Metacognitive Prompting"),
                (EMOTIONAL_PROMPT, "Emotional Prompting"),
                (NEGATIVE_PROMPT, "Negative Prompting"),
            ]

            prompt_dict, name = technique_map[tech_choice]
            test_prompting_technique(prompt_dict, name)

    elif choice == "2":
        run_all_tests()

    else:
        print("Invalid choice. Please run the script again.")