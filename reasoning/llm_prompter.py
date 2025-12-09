"""
LLM Prompter Module
Interface for Large Language Model queries
"""

from typing import Dict, Optional, Tuple
import json
import os


class LLMPrompter:
    """Interface for querying LLMs"""
    
    def __init__(self, gpt_version: str = "gpt-3.5-turbo", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.gpt_version = gpt_version
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        
        # Load prompt templates
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict:
        """Load prompt templates"""
        # Default prompts
        prompts = {
            'subgoal-verifier': {
                'template-system': 'You are a robot task analyzer. Verify if a subgoal was successfully completed and provide detailed explanations.',
                'template-user': 'Task: {task}\nSubgoal: {subgoal}\nObservation: {observation}\n\nWas the subgoal successful? Please answer:\n1. Yes or No\n2. A detailed explanation of why it succeeded or failed.\n\nIf it failed, explain what is missing or what went wrong based on the observation.'
            },
            'reasoning-execution': {
                'template-system': 'You are a robot failure analyzer. Explain why a robot action failed.',
                'template-user': 'Task: {task}\nAction: {action}\nObservation: {observation}\nWhy did the action fail?'
            },
            'reasoning-planning': {
                'template-system': 'You are a robot planner analyzer. Identify planning errors.',
                'template-user': 'Task: {task}\nPlan: {plan}\nFinal State: {final_state}\nExpected Goal: {goal}\nWhat went wrong in the plan?'
            },
            'constraint-generator': {
                'template-system': 'You are a constraint generator for robot tasks. Generate structured logical constraints in JSON format with executable AST expressions.',
                'template-user': '''Task: {task}
Scene Graph: {scene_graph}
Task Goal: {goal}

Generate logical constraints in the following JSON format:
{{
  "constraints": [
    {{
      "id": "C1",
      "type": "pre",  // "pre" (precondition), "post" (postcondition), "invariant", or "goal"
      "description": "Machine must be open before inserting a cup",
      "condition_expr": "(eq machine.door 'open')",  // Executable AST/DSL expression in LISP format
      "severity": "hard",  // "hard" or "soft"
      "eval_time": "pre"  // "pre" (before action), "post" (after action), "now" (current state), or "final" (task completion)
    }}
  ]
}}

AST Expression Format:
- Location: (inside obj container), (on_top_of obj surface)
- State: (eq obj.state 'open'), (eq obj.state 'closed'), (eq obj.state 'empty')
- Combined: (and (inside mug sink) (not (inside mug coffee_machine)))
- Negation: (not (inside obj container))

Generate constraints covering:
1. Preconditions (must be satisfied before actions)
   - For put_in actions: container must be empty (e.g., "Coffee machine must be empty before inserting mug")
   - For put_on actions: surface must be clear
   - For toggle actions: object must be accessible
2. Postconditions (must be satisfied after actions)
3. Invariants (must always be satisfied)
4. Goal constraints (final success conditions)
5. Causal chains (e.g., fill → has_water → heat)
6. **Occupancy constraints** (containers must be empty before insertion)

Example occupancy constraint for put_in:
{{
  "id": "C8",
  "type": "pre",
  "description": "Coffee machine must be empty before inserting mug",
  "condition_expr": "(empty coffee_machine)",
  "severity": "hard",
  "eval_time": "pre"
}}

Return ONLY valid JSON, no additional text.'''
            },
            'causal-verifier': {
                'template-system': 'You are a causal logic verifier. Verify if the causal relationships in a scene graph are logically consistent.',
                'template-user': 'Scene Graph: {scene_graph}\nAction: {action}\nExpected Effect: {expected_effect}\nActual Observation: {observation}\nVerify if the causal logic is consistent. Explain any inconsistencies.'
            },
            'consistency-verifier': {
                'template-system': 'You are a scene consistency verifier. Check if scene graphs at different time points are consistent.',
                'template-user': 'Previous Scene: {previous_scene}\nCurrent Scene: {current_scene}\nAction Performed: {action}\nCheck if the scene transition is consistent. Identify any inconsistencies.'
            }
        }
        return prompts
    
    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Tuple[str, Dict]:
        """
        Query the LLM
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            (response_text, metadata) tuple
        """
        if not self.api_key:
            # Mock response for testing
            return "Mock LLM response: The action failed because the target object was not found.", {}
        
        try:
            import openai
            
            # Configure client with base_url if provided (for poloapi or other providers)
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": 60.0  # 60 seconds timeout to prevent hanging
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            client = openai.OpenAI(**client_kwargs)
            
            response = client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            metadata = {
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            return response_text, metadata
            
        except ImportError:
            print("⚠️  openai package not installed. Install with: pip install openai")
            return "Mock LLM response (openai not installed)", {}
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Error calling LLM API: {error_msg}")
            # 如果是超时错误，提供更详细的提示
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print("   💡 Tip: If using poloapi, make sure API_KEY and POLOAPI_BASE_URL are correctly set")
                print("   💡 Tip: Check your network connection and try again")
            return f"Mock LLM response (API error: {error_msg})", {}
    
    def verify_subgoal(self, task: str, subgoal: str, observation: str) -> Tuple[bool, str]:
        """
        Verify if a subgoal was successfully completed
        
        Returns:
            (is_success, explanation) tuple
        """
        prompt_info = self.prompts['subgoal-verifier']
        user_prompt = prompt_info['template-user'].format(
            task=task,
            subgoal=subgoal,
            observation=observation
        )
        
        response, _ = self.query(
            prompt_info['template-system'], 
            user_prompt,
            max_tokens=300  # Increase tokens for detailed explanation
        )
        
        is_success = "yes" in response.lower()
        
        # If failed and explanation is too brief, get more detailed explanation
        if not is_success and len(response.strip()) < 50:
            # Use execution failure explanation for more details
            detailed_explanation = self.explain_execution_failure(
                task=task,
                action=subgoal,
                observation=observation
            )
            # Combine the verification result with detailed explanation
            explanation = f"{response.strip()}\n\nDetailed Analysis: {detailed_explanation}"
            return is_success, explanation
        
        return is_success, response
    
    def explain_execution_failure(self, task: str, action: str, observation: str) -> str:
        """Explain why an execution action failed"""
        prompt_info = self.prompts['reasoning-execution']
        user_prompt = prompt_info['template-user'].format(
            task=task,
            action=action,
            observation=observation
        )
        
        response, _ = self.query(prompt_info['template-system'], user_prompt)
        return response
    
    def explain_planning_failure(self, task: str, plan: str, final_state: str, goal: str) -> str:
        """Explain why a planning approach failed"""
        prompt_info = self.prompts['reasoning-planning']
        user_prompt = prompt_info['template-user'].format(
            task=task,
            plan=plan,
            final_state=final_state,
            goal=goal
        )
        
        response, _ = self.query(prompt_info['template-system'], user_prompt)
        return response

