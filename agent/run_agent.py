import os
import sys
import re
import json
import time
import hashlib
import argparse
import requests
import concurrent.futures
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Callable
from datetime import datetime

try:
    import boto3
    from botocore.config import Config as BotoConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from data_loader import QueryMeta, load_queries


TOOL_SCHEMAS = {
    "search_flights": {
        "name": "search_flights",
        "description": "Search for flights between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "departure_city": {
                    "type": "string",
                    "description": "The city to depart from"
                },
                "arrival_city": {
                    "type": "string",
                    "description": "The destination city"
                },
                "trip_type": {
                    "type": "string",
                    "enum": ["one_way", "round_trip"],
                    "description": "Type of trip"
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price filter"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["price", "departure_time"],
                    "description": "Sort results by this field"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10)"
                }
            },
            "required": ["departure_city", "arrival_city", "trip_type"]
        }
    },
    "search_hotels": {
        "name": "search_hotels",
        "description": "Search for hotels in one or multiple cities. Use comma-separated cities for batch query.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City or comma-separated cities (e.g., 'Paris' or 'Paris,London,Tokyo')"
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price per night"
                },
                "min_star": {
                    "type": "integer",
                    "description": "Minimum star rating (1-5)"
                },
                "amenity": {
                    "type": "string",
                    "description": "Semantic search for amenities"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["price", "rating", "star"],
                    "description": "Sort results by this field"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results per city (default 10)"
                }
            },
            "required": ["city"]
        }
    },
    "search_attractions": {
        "name": "search_attractions",
        "description": "Search for attractions in one or multiple cities. Use comma-separated cities for batch query.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City or comma-separated cities (e.g., 'Paris' or 'Paris,London,Tokyo')"
                },
                "max_ticket_price": {
                    "type": "number",
                    "description": "Maximum ticket price"
                },
                "facility": {
                    "type": "string",
                    "description": "Semantic search for facilities"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["price", "rating", "duration_of_visit"],
                    "description": "Sort results by this field"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results per city (default 10)"
                }
            },
            "required": ["city"]
        }
    },
    "search_cars": {
        "name": "search_cars",
        "description": "Search for rental cars in one or multiple cities. Use comma-separated cities for batch query.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City or comma-separated cities (e.g., 'Paris' or 'Paris,London,Tokyo')"
                },
                "min_capacity": {
                    "type": "integer",
                    "description": "Minimum passenger capacity"
                },
                "max_price_per_day": {
                    "type": "number",
                    "description": "Maximum price per day"
                },
                "car_type": {
                    "type": "string",
                    "description": "Type of car"
                },
                "extra_service": {
                    "type": "string",
                    "description": "Semantic search for extra services"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results per city (default 10)"
                }
            },
            "required": ["city"]
        }
    },
    "submit_plan": {
        "name": "submit_plan",
        "description": "Submit the final travel plan. Call this when you have gathered enough information.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_feasible": {
                    "type": "boolean",
                    "description": "Whether the travel plan is feasible"
                },
                "refusal_reason": {
                    "type": "string",
                    "description": "Reason for refusal if not feasible"
                },
                "plan": {
                    "type": "object",
                    "description": "The travel plan organized by day"
                }
            },
            "required": ["is_feasible", "plan"]
        }
    }
}


@dataclass
class PipelineConfig:
    llm_provider: str = "vllm"
    
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # Claude Haiku 4.5
    bedrock_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    
    anthropic_api_key: str = ""
    anthropic_model_id: str = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5
    
    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.0-flash"  # Gemini 2.0 Flash
    gemini_thinking_level: str = "low"  # low, medium, high for Gemini 3
    
    openai_api_key: str = ""
    openai_model_id: str = "gpt-5-mini-2025-08-07"
    
    travel_api_base_url: str = "http://localhost:5000"
    
    max_tool_calls: int = 15
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 42
    
    enable_cache: bool = True
    
    backup_dir: str = "outputs"
    backup_interval: int = 50
    output_dir: str = "outputs"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, usage: dict):
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)


class APICache:
    
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._call_log: list[dict] = []
        self._max_cache_size = 2000
    
    def _make_key(self, tool_name: str, params: dict) -> str:
        param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(f"{tool_name}:{param_str}".encode()).hexdigest()
    
    def get(self, tool_name: str, params: dict) -> Optional[Any]:
        key = self._make_key(tool_name, params)
        return self._cache.get(key)
    
    def set(self, tool_name: str, params: dict, result: Any):
        if len(self._cache) >= self._max_cache_size:
            keys_to_remove = list(self._cache.keys())[:self._max_cache_size // 2]
            for k in keys_to_remove:
                del self._cache[k]
            print(f"[Cache] Cleaned {len(keys_to_remove)} old entries, current size: {len(self._cache)}")
        
        key = self._make_key(tool_name, params)
        self._cache[key] = result
    
    def log_call(self, tool_name: str, params: dict, result: Any, 
                 cached: bool, error: Optional[str] = None):
        self._call_log.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "params": params,
            "result": result,
            "cached": cached,
            "error": error
        })
    
    @property
    def call_log(self) -> list[dict]:
        return self._call_log


class TravelAPIClient:
    
    def __init__(self, base_url: str, cache: APICache):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
    
    def _call_api(self, endpoint: str, params: dict, tool_name: str) -> dict:
        cached_result = self.cache.get(tool_name, params)
        if cached_result is not None:
            self.cache.log_call(tool_name, params, cached_result, cached=True)
            return cached_result
        
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        
        try:
            url = f"{self.base_url}{endpoint}"
            resp = requests.get(url, params=params, timeout=30)
            
            if resp.status_code == 400:
                error_data = resp.json()
                error_msg = error_data.get("error", "Bad Request") if isinstance(error_data, dict) else "Bad Request"
                result = {"error": error_msg, "type": "param_error"}
            elif resp.status_code != 200:
                result = {"error": f"HTTP {resp.status_code}", "type": "api_error"}
            else:
                raw_result = resp.json()
                if isinstance(raw_result, list):
                    key_map = {
                        "/flights2": "flights",
                        "/hotels2": "hotels",
                        "/attractions2": "attractions",
                        "/cars2": "cars"
                    }
                    key = key_map.get(endpoint, "results")
                    result = {key: raw_result}
                else:
                    result = raw_result
            
            has_error = isinstance(result, dict) and "error" in result
            if not has_error:
                self.cache.set(tool_name, params, result)
            
            error_msg = result.get("error") if isinstance(result, dict) else None
            self.cache.log_call(tool_name, params, result, cached=False, error=error_msg)
            return result
            
        except Exception as e:
            error_result = {"error": str(e), "type": "api_error"}
            self.cache.log_call(tool_name, params, None, cached=False, error=str(e))
            return error_result
    
    def search_flights(self, **params) -> dict:
        return self._call_api("/flights2", params, "search_flights")
    
    def search_hotels(self, **params) -> dict:
        city = params.get("city", "")
        if "," in city:
            cities = [c.strip() for c in city.split(",")]
            results = {}
            for c in cities:
                single_params = {**params, "city": c}
                result = self._call_api("/hotels2", single_params, f"search_hotels_{c}")
                results[c] = result.get("hotels", result.get("data", []))
            return {"hotels_by_city": results}
        return self._call_api("/hotels2", params, "search_hotels")
    
    def search_attractions(self, **params) -> dict:
        city = params.get("city", "")
        if "," in city:
            cities = [c.strip() for c in city.split(",")]
            results = {}
            for c in cities:
                single_params = {**params, "city": c}
                result = self._call_api("/attractions2", single_params, f"search_attractions_{c}")
                results[c] = result.get("attractions", result.get("data", []))
            return {"attractions_by_city": results}
        return self._call_api("/attractions2", params, "search_attractions")
    
    def search_cars(self, **params) -> dict:
        city = params.get("city", "")
        if "," in city:
            cities = [c.strip() for c in city.split(",")]
            results = {}
            for c in cities:
                single_params = {**params, "city": c}
                result = self._call_api("/cars2", single_params, f"search_cars_{c}")
                results[c] = result.get("cars", result.get("data", []))
            return {"cars_by_city": results}
        return self._call_api("/cars2", params, "search_cars")


# ============ System Prompt (Simplified) ============
SYSTEM_PROMPT = """You are a Travel Planning Agent.

## Available Tools
{tools_json}

## Rules
1. Maximum {max_calls} tool calls. Be EFFICIENT - do NOT repeat calls!
2. Database is the SOLE source of truth. Entity names MUST match exactly.
3. All times use HH:MM 24-hour format.
4. PRIORITIZE "Special Requirements" (e.g., accessibility) over saving budget.
5. Do NOT repeat attractions across days.
6. For hotels/attractions/cars: use comma-separated cities in ONE call (e.g., city="Paris,London").
7. MULTI-CITY TRIPS: If traveling "A to B, C", you MUST:
   - Visit ALL destination cities (B AND C)
   - Search flights: A→B, B→C, C→A (for round trip)
   - Split days between cities proportionally
8. If car rental is requested, search for cars in destination cities.

## CRITICAL OUTPUT RULES
- Output EXACTLY ONE JSON object per response
- Do NOT repeat a tool call - check "Collected Data" section first!
- After collecting flights, hotels, attractions (and cars if needed), call submit_plan

## Output Format
```json
{{
  "thought": "Brief reasoning",
  "action": "tool_name",
  "action_input": {{...}}
}}
```

## Submit Plan Format
```json
{{
  "thought": "Ready to submit",
  "action": "submit_plan",
  "action_input": {{
    "is_feasible": true,
    "plan": {{
      "day1": {{
        "current_city": "A to B",
        "flights": [{{"flight_number": "XX123", "departure_city": "A", "arrival_city": "B", "departure_time": "08:00", "arrival_time": "10:00", "price": 150}}],
        "attractions": [{{"name": "Museum", "city": "B", "visit_start": "14:00", "visit_end": "16:00"}}],
        "hotel": {{"name": "Hotel X", "city": "B", "price_per_night": 80, "check_in": "15:00"}},
        "car": {{"car_type": "Compact", "capacity": 4, "price_per_day": 50}}
      }}
    }}
  }}
}}
```

If impossible: {{"is_feasible": false, "refusal_reason": "...", "plan": {{}}}}
"""


# ============ ReAct Runner ============
class ReActRunner:
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = APICache()
        self.api_client = TravelAPIClient(config.travel_api_base_url, self.cache)
        self.token_usage = TokenUsage()
        self._bedrock_client = None
    
    def _get_bedrock_client(self):
        if self._bedrock_client is not None:
            return self._bedrock_client
        
        if not HAS_BOTO3:
            raise RuntimeError("boto3 is required for Bedrock. Install with: pip install boto3")
        
        boto_config = BotoConfig(
            region_name=self.config.bedrock_region,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )
        
        if self.config.aws_access_key_id and self.config.aws_secret_access_key:
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.config.bedrock_region,
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                config=boto_config
            )
        else:
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.config.bedrock_region,
                config=boto_config
            )
        
        return self._bedrock_client
    
    def _call_llm(self, messages: list[dict]) -> tuple[str, dict]:
        if self.config.llm_provider == "bedrock":
            return self._call_bedrock(messages)
        elif self.config.llm_provider == "anthropic":
            return self._call_anthropic(messages)
        elif self.config.llm_provider == "gemini":
            return self._call_gemini(messages)
        elif self.config.llm_provider == "openai":
            return self._call_openai(messages)
        else:
            return self._call_vllm(messages)
    
    def _call_bedrock(self, messages: list[dict]) -> tuple[str, dict]:
        
        """
        client = self._get_bedrock_client()
        
        system_prompts = []
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompts.append({"text": msg["content"]})
            else:
                conversation.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })
        
        converse_params = {
            "modelId": self.config.bedrock_model_id,
            "messages": conversation,
            "inferenceConfig": {
                "maxTokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
        }
        
        if system_prompts:
            converse_params["system"] = system_prompts
        
        # Retry logic for network instability
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = client.converse(**converse_params)
                break
            except Exception as e:
                last_error = e
                print(f"Bedrock call attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Bedrock call failed after {max_retries} attempts: {last_error}")
            
        content = ""
        output = response.get("output", {})
        if "message" in output:
            for block in output["message"].get("content", []):
                if "text" in block:
                    content += block["text"]
        
        usage_data = response.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("inputTokens", 0),
            "completion_tokens": usage_data.get("outputTokens", 0),
            "total_tokens": usage_data.get("inputTokens", 0) + usage_data.get("outputTokens", 0)
        }
        
        return content, usage
    
    
    def _call_anthropic(self, messages: list[dict]) -> tuple[str, dict]:
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic library not installed. Run: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        
        # Retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=self.config.anthropic_model_id,
                    max_tokens=self.config.max_tokens,
                    system=system_content,
                    messages=user_messages
                )
                break
            except Exception as e:
                last_error = e
                print(f"Anthropic call attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Anthropic call failed after {max_retries} attempts: {last_error}")
        
        content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                content += block.text
        
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        return content, usage
    
    
    def _call_gemini(self, messages: list[dict]) -> tuple[str, dict]:
        if not HAS_GEMINI:
            raise ImportError("google-genai library not installed. Run: pip install google-genai")
        
        client = genai.Client(api_key=self.config.gemini_api_key)
        
        system_instruction = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                contents.append(types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=msg["content"])]
                ))
        
        thinking_config = None
        if "gemini-3" in self.config.gemini_model_id.lower():
            thinking_config = types.ThinkingConfig(
                thinking_level=self.config.gemini_thinking_level  # minimal, low, medium, high
            )
        
        generate_config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            thinking_config=thinking_config
        )
        
        # Retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.config.gemini_model_id,
                    contents=contents,
                    config=generate_config
                )
                break
            except Exception as e:
                last_error = e
                print(f"Gemini call attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_error}")
        
        content = ""
        if response.text:
            content = response.text
        
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0
        }
        
        return content, usage


    def _call_openai(self, messages: list[dict]) -> tuple[str, dict]:
        if not HAS_OPENAI:
            raise ImportError("openai library not installed. Run: pip install openai")
        
        client = OpenAI(api_key=self.config.openai_api_key)
        
        # Retry logic
        max_retries = 3
        last_error = None
        
        create_params = {
            "model": self.config.openai_model_id,
            "messages": messages,
            "max_completion_tokens": self.config.max_tokens,
            "seed": self.config.seed
        }
        if "gpt-5" not in self.config.openai_model_id:
            create_params["temperature"] = self.config.temperature
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**create_params)
                break
            except Exception as e:
                last_error = e
                print(f"OpenAI call attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_error}")
        
        content = response.choices[0].message.content or ""
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }
        
        return content, usage


    def _call_vllm(self, messages: list[dict]) -> tuple[str, dict]:
        url = f"{self.config.vllm_base_url}/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            
            usage = result.get("usage", {})
            content = result["choices"][0]["message"]["content"]
            
            return content, usage
            
        except Exception as e:
            raise RuntimeError(f"VLLM call failed: {e}")
    
    def _parse_action(self, response: str) -> tuple[Optional[str], Optional[dict], Optional[str]]:
        response = response.strip()
        
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        
        start = response.find("{")
        if start >= 0:
            depth = 0
            end = start
            for i, c in enumerate(response[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            if end > start:
                try:
                    data = json.loads(response[start:end])
                    thought = data.get("thought", "")
                    action = data.get("action")
                    action_input = data.get("action_input", {})
                    return action, action_input, thought
                except json.JSONDecodeError:
                    pass
        
        return None, None, None
    
    def _execute_tool(self, action: str, params: dict) -> dict:
        if action == "search_flights":
            return self.api_client.search_flights(**params)
        elif action == "search_hotels":
            return self.api_client.search_hotels(**params)
        elif action == "search_attractions":
            return self.api_client.search_attractions(**params)
        elif action == "search_cars":
            return self.api_client.search_cars(**params)
        elif action == "submit_plan":
            return {"status": "plan_submitted", "plan": params}
        else:
            return {"error": f"Unknown tool: {action}", "type": "param_error"}
    
    def _validate_params(self, action: str, params: dict) -> Optional[str]:
        if action not in TOOL_SCHEMAS:
            return f"Unknown tool: {action}"
        
        schema = TOOL_SCHEMAS[action]["parameters"]
        required = schema.get("required", [])
        
        for req in required:
            if req not in params or params[req] is None:
                return f"Missing required parameter: {req}"
        
        return None
    
    def run(self, query_meta: QueryMeta) -> dict:
        tools_json = json.dumps(list(TOOL_SCHEMAS.values()), indent=2, ensure_ascii=False)
        
        system_prompt = SYSTEM_PROMPT.format(
            tools_json=tools_json,
            max_calls=self.config.max_tool_calls
        )
        
        user_prompt = self._build_user_prompt(query_meta)
        
        scratchpad = {
            "collected_flights": [],
            "collected_hotels": [],
            "collected_cars": [],
            "collected_attractions": [],
            "selected": {
                "outbound_flight": None,
                "return_flight": None,
                "hotel": None,
                "car": None
            },
            "budget_used": 0,
            "notes": "Starting search"
        }
        
        tool_call_count = 0
        param_error_count = 0
        max_param_errors = 3
        
        final_plan = None
        trajectory = []
        last_observation = None
        
        while tool_call_count < self.config.max_tool_calls:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_scratchpad_prompt(
                    user_prompt, scratchpad, last_observation, tool_call_count
                )}
            ]
            
            try:
                response, usage = self._call_llm(messages)
                self.token_usage.add(usage)
            except Exception as e:
                trajectory.append({"error": f"LLM call failed: {e}"})
                break
            
            action, params, thought, new_scratchpad = self._parse_scratchpad_response(response)
            
            if new_scratchpad:
                scratchpad = new_scratchpad
            
            trajectory.append({
                "step": tool_call_count + 1,
                "thought": thought,
                "action": action,
                "params": params,
                "raw_response": response[:500]
            })
            
            if action is None:
                param_error_count += 1
                last_observation = "Error: Could not parse your response. Please respond with valid JSON."
                if param_error_count >= max_param_errors:
                    break
                continue
            
            validation_error = self._validate_params(action, params or {})
            if validation_error:
                param_error_count += 1
                last_observation = f"Error: {validation_error}. Please fix and try again."
                if param_error_count >= max_param_errors:
                    break
                continue
            
            if action == "submit_plan":
                final_plan = params
                trajectory[-1]["result"] = "Plan submitted"
                break
            
            tool_call_count += 1
            result = self._execute_tool(action, params)
            
            trajectory[-1]["result"] = result
            
            if isinstance(result, dict) and result.get("type") == "api_error":
                tool_call_count -= 1
            
            scratchpad = self._update_scratchpad(scratchpad, action, result)
            
            last_observation = self._compress_observation(action, result)
        
        if final_plan is None and tool_call_count >= self.config.max_tool_calls:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_scratchpad_prompt(
                    user_prompt, scratchpad, 
                    "FINAL CALL: You have reached the maximum tool calls. Submit your best plan NOW using submit_plan.",
                    tool_call_count
                )}
            ]
            
            try:
                response, usage = self._call_llm(messages)
                self.token_usage.add(usage)
                action, params, thought, _ = self._parse_scratchpad_response(response)
                if action == "submit_plan":
                    final_plan = params
            except:
                pass
        
        if final_plan is None:
            final_plan = {
                "is_feasible": False,
                "refusal_reason": "Failed to generate a valid plan",
                "plan": {}
            }
        
        return {
            "plan": final_plan,
            "tool_call_count": tool_call_count,
            "param_error_count": param_error_count,
            "token_usage": asdict(self.token_usage),
            "api_call_log": self.cache.call_log,
            "trajectory": trajectory
        }
    
    def _build_scratchpad_prompt(self, user_prompt: str, scratchpad: dict, 
                                   last_observation: Optional[str], step: int) -> str:
        collected_summary = self._format_scratchpad_summary(scratchpad)
        
        prompt = f"""{user_prompt}

## Step {step + 1}/{self.config.max_tool_calls}

## Already Collected (DO NOT search again!)
{collected_summary}
"""
        if last_observation:
            prompt += f"""
## Last Result
{last_observation}
"""
        
        prompt += "\nDecide your next action (do NOT repeat previous searches):"
        return prompt
    
    def _format_scratchpad_summary(self, scratchpad: dict) -> str:
        lines = []
        
        flights = scratchpad.get("collected_flights", [])
        if flights:
            routes = set()
            for f in flights:
                if "->" in f:
                    parts = f.split("->")
                    if len(parts) >= 2:
                        dep = parts[0].split(":")[-1].strip()
                        arr = parts[1].split()[0].strip()
                        routes.add(f"{dep}->{arr}")
            if routes:
                lines.append(f"Flight routes searched: {', '.join(sorted(routes))}")
            lines.append(f"Flight options: {len(flights)}")
            for f in flights[:3]:
                lines.append(f"  - {f}")
        
        hotels = scratchpad.get("collected_hotels", [])
        if hotels:
            cities = set()
            for h in hotels:
                if "(" in h and ")" in h:
                    city = h.split("(")[1].split(")")[0]
                    cities.add(city)
            if cities:
                lines.append(f"Hotels searched in: {', '.join(sorted(cities))}")
            lines.append(f"Hotel options: {len(hotels)}")
            for h in hotels[:3]:
                lines.append(f"  - {h}")
        
        attractions = scratchpad.get("collected_attractions", [])
        if attractions:
            cities = set()
            for a in attractions:
                if "(" in a and ")" in a:
                    city = a.split("(")[1].split(")")[0]
                    cities.add(city)
            if cities:
                lines.append(f"Attractions searched in: {', '.join(sorted(cities))}")
            lines.append(f"Attraction options: {len(attractions)}")
            for a in attractions[:3]:
                lines.append(f"  - {a}")
        
        cars = scratchpad.get("collected_cars", [])
        if cars:
            cities = set()
            for c in cars:
                if "(" in c and ")" in c:
                    city = c.split("(")[1].split(")")[0]
                    cities.add(city)
            if cities:
                lines.append(f"Cars searched in: {', '.join(sorted(cities))}")
            lines.append(f"Car options: {len(cars)}")
            for c in cars[:3]:
                lines.append(f"  - {c}")
        
        if not lines:
            return "No data collected yet."
        
        return "\n".join(lines)
    
    def _parse_scratchpad_response(self, response: str) -> tuple:
        response = response.strip()
        
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        
        start = response.find("{")
        if start >= 0:
            depth = 0
            end = start
            for i, c in enumerate(response[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            if end > start:
                try:
                    data = json.loads(response[start:end])
                    thought = data.get("thought", "")
                    action = data.get("action")
                    action_input = data.get("action_input", {})
                    return action, action_input, thought, None
                except json.JSONDecodeError:
                    pass
        
        return None, None, None, None
    
    def _update_scratchpad(self, scratchpad: dict, action: str, result: dict) -> dict:
        if not isinstance(result, dict) or "error" in result:
            return scratchpad
        
        if action == "search_flights":
            flights = []
            one_way_flights = result.get("flights", [])
            depart_flights = result.get("depart_flights", [])
            return_flights = result.get("return_flights", [])
            
            for f in one_way_flights[:3]:
                flights.append(f"{f.get('flight_number')}: {f.get('departure_city')}->{f.get('arrival_city')} ${f.get('price')}")
            for f in depart_flights[:3]:
                flights.append(f"{f.get('flight_number')}: {f.get('departure_city')}->{f.get('arrival_city')} ${f.get('price')}")
            for f in return_flights[:3]:
                flights.append(f"(return) {f.get('flight_number')}: {f.get('departure_city')}->{f.get('arrival_city')} ${f.get('price')}")
            
            if flights:
                scratchpad["collected_flights"].extend(flights)
        
        elif action == "search_hotels" or action.startswith("search_hotels_"):
            hotels_by_city = result.get("hotels_by_city", {})
            if hotels_by_city:
                for city, hotels in hotels_by_city.items():
                    for h in hotels[:3]:
                        scratchpad["collected_hotels"].append(
                            f"{h.get('name')} ({city}): ${h.get('price')}/night, {h.get('star')}star"
                        )
            else:
                for h in result.get("hotels", [])[:3]:
                    scratchpad["collected_hotels"].append(
                        f"{h.get('name')}: ${h.get('price')}/night, {h.get('star')}star"
                    )
        
        elif action == "search_cars" or action.startswith("search_cars_"):
            cars_by_city = result.get("cars_by_city", {})
            if cars_by_city:
                for city, cars in cars_by_city.items():
                    for c in cars[:3]:
                        scratchpad["collected_cars"].append(
                            f"{c.get('car_type')} ({city}): ${c.get('price_per_day')}/day, {c.get('capacity')} seats"
                        )
            else:
                for c in result.get("cars", [])[:3]:
                    scratchpad["collected_cars"].append(
                        f"{c.get('car_type')}: ${c.get('price_per_day')}/day, {c.get('capacity')} seats"
                    )
        
        elif action == "search_attractions" or action.startswith("search_attractions_"):
            attractions_by_city = result.get("attractions_by_city", {})
            if attractions_by_city:
                for city, attractions in attractions_by_city.items():
                    for a in attractions[:3]:
                        scratchpad["collected_attractions"].append(
                            f"{a.get('attraction_name')} ({city}): ${a.get('ticket_price')}"
                        )
            else:
                for a in result.get("attractions", [])[:3]:
                    scratchpad["collected_attractions"].append(
                        f"{a.get('attraction_name')}: ${a.get('ticket_price')}"
                    )
        
        return scratchpad
        
        return scratchpad
    
    def _compress_observation(self, action: str, result: dict) -> str:
        if not isinstance(result, dict):
            return str(result)[:500]
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if action == "search_flights":
            one_way = result.get("flights", [])[:3]
            depart = result.get("depart_flights", [])[:3]
            ret = result.get("return_flights", [])[:3]
            compressed = []
            for f in one_way:
                compressed.append(f"{f.get('flight_number')}: {f.get('departure_city')}->{f.get('arrival_city')} ${f.get('price')}")
            for f in depart:
                compressed.append(f"{f.get('flight_number')}: {f.get('departure_city')}->{f.get('arrival_city')} ${f.get('price')}")
            if ret:
                compressed.append("Return flights:")
                for f in ret:
                    compressed.append(f"{f.get('flight_number')}: {f.get('departure_city')}->{f.get('arrival_city')} ${f.get('price')}")
            return "Flights found: " + "; ".join(compressed) if compressed else "No flights found"
        
        elif action == "search_hotels" or action.startswith("search_hotels_"):
            hotels_by_city = result.get("hotels_by_city", {})
            if hotels_by_city:
                compressed = []
                for city, hotels in hotels_by_city.items():
                    for h in hotels[:2]:
                        compressed.append(f"{h.get('name')} ({city}): ${h.get('price')}/night")
                return "Hotels found: " + "; ".join(compressed)
            hotels = result.get("hotels", [])[:3]
            compressed = [f"{h.get('name')}: ${h.get('price')}/night, {h.get('star')}star" for h in hotels]
            return "Hotels found: " + "; ".join(compressed)
        
        elif action == "search_cars" or action.startswith("search_cars_"):
            cars_by_city = result.get("cars_by_city", {})
            if cars_by_city:
                compressed = []
                for city, cars in cars_by_city.items():
                    for c in cars[:2]:
                        compressed.append(f"{c.get('car_type')} ({city}): ${c.get('price_per_day')}/day")
                return "Cars found: " + "; ".join(compressed)
            cars = result.get("cars", [])[:3]
            compressed = [f"{c.get('car_type')}: ${c.get('price_per_day')}/day, {c.get('capacity')} seats" for c in cars]
            return "Cars found: " + "; ".join(compressed)
        
        elif action == "search_attractions" or action.startswith("search_attractions_"):
            attractions_by_city = result.get("attractions_by_city", {})
            if attractions_by_city:
                compressed = []
                for city, attrs in attractions_by_city.items():
                    for a in attrs[:2]:
                        compressed.append(f"{a.get('attraction_name')} ({city}): ${a.get('ticket_price')}")
                return "Attractions found: " + "; ".join(compressed)
            attrs = result.get("attractions", [])[:3]
            compressed = [f"{a.get('attraction_name')}: ${a.get('ticket_price')}" for a in attrs]
            return "Attractions found: " + "; ".join(compressed)
        
        return json.dumps(result, ensure_ascii=False)[:500]
    
    def _build_user_prompt(self, meta: QueryMeta) -> str:
        prompt = f"""## Travel Request
{meta.query}

## Constraints
- Number of travelers: {meta.person_num}
- Total budget: {meta.budget} USD
- Duration: {meta.days} days
- Hotel rooms needed: {meta.rooms_count}

## Required Services
"""
        if meta.req_flight:
            dep = meta.req_flight.get("departure_city", "")
            arr = meta.req_flight.get("arrival_city", [])
            trip = meta.req_flight.get("trip_type", "one_way")
            arr_str = arr[0] if isinstance(arr, list) and arr else str(arr)
            prompt += f"- Flight: {dep} → {arr_str} ({trip})\n"
        
        if meta.implicit_keywords:
            prompt += "\n## Special Requirements (Implicit Needs)\n"
            for kw in meta.implicit_keywords:
                prompt += f"- **{kw}**\n"
        
        prompt += "\nPlease start by searching for available options, then create a detailed travel plan.\n"
        prompt += "REMEMBER:\n"
        prompt += "1. If there are 'Special Requirements' above (e.g., disabled traveler), you must INFER the necessary facilities (e.g., 'wheelchair space', 'accessibility features') and strictly select valid options. Do NOT ignore these needs for a lower price.\n"
        prompt += "2. Your itinerary must be DENSE and fully occupied for every single day. Do not leave any half-days or long gaps empty. You must find and schedule enough activities to fill the entire duration."
        
        return prompt
    



class LLMPipelineEngine:
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.total_token_usage = TokenUsage()
    
    def run_single_query(self, query_meta: QueryMeta) -> dict:
        runner = ReActRunner(self.config)
        result = runner.run(query_meta)
        
        usage = result.get("token_usage", {})
        self.total_token_usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_token_usage.completion_tokens += usage.get("completion_tokens", 0)
        self.total_token_usage.total_tokens += usage.get("total_tokens", 0)
        
        return result
    
    def run_batch(self, queries: list[QueryMeta], 
                  progress_callback: Callable = None,
                  concurrency: int = 1,
                  output_path: str = None,
                  initial_results: list = None,
                  skip_indices: set = None) -> list[dict]:
        results = list(initial_results) if initial_results is not None else []
        skip_set = skip_indices if skip_indices else set()
        
        print(f"[DEBUG] run_batch started with {len(results)} initial results, skipping {len(skip_set)} indices")
        
        def _process_query(args):
            i, query = args
            try:
                if progress_callback:
                    progress_callback(i, len(queries), query)
                
                result = self.run_single_query(query)
                return {
                    "query_index": i,
                    "query": query.query,
                    "level": query.level,
                    **result,
                    "metadata": {
                        "person_num": query.person_num,
                        "budget": query.budget,
                        "days": query.days,
                        "implicit_keywords": query.implicit_keywords
                    }
                }
            except Exception as e:
                print(f"Error processing query {i}: {e}")
                return {
                    "query_index": i,
                    "error": str(e)
                }

        def save_checkpoint():
            if output_path:
                try:
                    sorted_results = sorted(results, key=lambda x: x.get("query_index", 0))
                    
                    checkpoint_data = {
                        "config": asdict(self.config),
                        "total_queries": len(sorted_results),
                        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "total_token_usage": asdict(self.total_token_usage),
                        "results": sorted_results
                    }
                    
                    temp_path = f"{output_path}.tmp"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                    
                    if os.path.exists(output_path):
                        os.replace(temp_path, output_path)
                    else:
                        os.rename(temp_path, output_path)
                        
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
        
        def save_and_clear_memory():
            try:
                save_checkpoint()
                
                import gc
                for r in results:
                    if "trajectory" in r:
                        for step in r["trajectory"]:
                            if "result" in step:
                                step["result"] = None
                    if "api_call_log" in r:
                        r["api_call_log"] = []
                
                gc.collect()
                print(f"\n[Memory] Saved and cleared memory for {len(results)} results")
            except Exception as e:
                print(f"Failed to save and clear memory: {e}")
        
        def save_backup(count):
            try:
                backup_dir = self.config.backup_dir
                os.makedirs(backup_dir, exist_ok=True)
                
                if self.config.llm_provider == "gemini":
                    model_name = self.config.gemini_model_id.replace("/", "_").replace(":", "_")
                elif self.config.llm_provider == "anthropic":
                    model_name = self.config.anthropic_model_id.replace("/", "_").replace(":", "_")
                elif self.config.llm_provider == "bedrock":
                    model_name = self.config.bedrock_model_id.replace("/", "_").replace(":", "_")
                else:
                    model_name = self.config.model_name.replace("/", "_").replace(":", "_")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"backup_{model_name}_{count}_{timestamp}.json")
                
                sorted_results = sorted(results, key=lambda x: x.get("query_index", 0))
                backup_data = {
                    "config": asdict(self.config),
                    "total_queries": len(sorted_results),
                    "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_token_usage": asdict(self.total_token_usage),
                    "results": sorted_results
                }
                
                with open(backup_path, "w", encoding="utf-8") as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
                
                print(f"\n[BACKUP] Saved to {backup_path} ({count} queries)")
            except Exception as e:
                print(f"Failed to save backup: {e}")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                args_list = []
                for i, q in enumerate(queries):
                    if i in skip_set:
                        continue
                    args_list.append((i, q))
                
                if not args_list:
                    print("No new queries to run.")
                    return results

                future_to_query = {executor.submit(_process_query, args): args[0] for args in args_list}
                
                completed_count = 0
                for future in tqdm(concurrent.futures.as_completed(future_to_query), total=len(args_list), desc="Processing"):
                    res = future.result()
                    results.append(res)
                    completed_count += 1
                    
                    if completed_count % 5 == 0:
                        save_checkpoint()
                    
                    if completed_count % 100 == 0:
                        save_and_clear_memory()
                    
                    if len(results) % self.config.backup_interval == 0:
                        save_backup(len(results))
                        
                save_checkpoint()
                
        except KeyboardInterrupt:
            print("\nBatch processing interrupted! Saving partial results...")
            save_checkpoint()
            
        results.sort(key=lambda x: x.get("query_index", 0))
        return results


def main():
    parser = argparse.ArgumentParser(description="KDD Travel Planning ReAct LLM Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input query CSV file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    parser.add_argument("--provider", choices=["vllm", "bedrock", "anthropic", "gemini", "openai"], default="vllm",
                        help="LLM provider: vllm, bedrock, anthropic, gemini, or openai")
    
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="VLLM service URL")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="VLLM model name")
    
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Bedrock model ID")
    parser.add_argument("--bedrock-region", default="us-east-1", help="AWS region")
    parser.add_argument("--aws-key", default="", help="AWS access key ID")
    parser.add_argument("--aws-secret", default="", help="AWS secret access key")
    
    parser.add_argument("--anthropic-key", default="", help="Anthropic API key")
    parser.add_argument("--anthropic-model", default="claude-haiku-4-5-20251001", help="Anthropic model ID")
    
    parser.add_argument("--gemini-key", default="", help="Google Gemini API key")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model ID")
    parser.add_argument("--gemini-thinking", default="minimal", choices=["minimal", "low", "medium", "high"], help="Gemini 3 thinking level (minimal=almost off)")
    
    parser.add_argument("--openai-key", default="", help="OpenAI API key")
    parser.add_argument("--openai-model", default="gpt-5-mini-2025-08-07", help="OpenAI model ID")
    
    parser.add_argument("--api-url", default="http://localhost:5000", help="Travel API service URL")
    parser.add_argument("--max-calls", type=int, default=15, help="Max tool calls per query")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--limit", type=int, help="Limit number of queries to process")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency level (threads)")
    parser.add_argument("--backup-interval", type=int, default=50, help="Backup every N queries")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        llm_provider=args.provider,
        vllm_base_url=args.vllm_url,
        model_name=args.model,
        bedrock_model_id=args.bedrock_model,
        bedrock_region=args.bedrock_region,
        aws_access_key_id=args.aws_key,
        aws_secret_access_key=args.aws_secret,
        anthropic_api_key=args.anthropic_key,
        anthropic_model_id=args.anthropic_model,
        gemini_api_key=args.gemini_key,
        gemini_model_id=args.gemini_model,
        gemini_thinking_level=args.gemini_thinking,
        openai_api_key=args.openai_key,
        openai_model_id=args.openai_model,
        travel_api_base_url=args.api_url,
        max_tool_calls=args.max_calls,
        max_tokens=args.max_tokens,
        temperature=0.0,
        seed=args.seed,
        backup_dir="outputs",
        backup_interval=args.backup_interval
    )
    
    print(f"Loading queries from {args.input}...")
    queries = load_queries(args.input)
    print(f"Loaded {len(queries)} queries")
    
    if args.limit:
        queries = queries[args.start:args.start + args.limit]
        print(f"Processing queries {args.start} to {args.start + len(queries)}")
    else:
        queries = queries[args.start:]
        print(f"Processing queries {args.start} to {len(queries) + args.start}")
    
    
    engine = LLMPipelineEngine(config)
    
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = config.model_name.split("/")[-1].replace("-", "_")
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"output_{input_name}_{model_short}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    previous_results = []
    finished_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "results" in data:
                    previous_results = data["results"]
                    finished_indices = {r.get("query_index") for r in previous_results if r.get("query_index") is not None}
                    print(f"Found existing output at {output_path} with {len(previous_results)} results. Resuming...")
        except Exception as e:
            print(f"Failed to load existing output from {output_path}: {e}")

    skip_indices = set()
    for global_index in finished_indices:
        if global_index >= args.start:
            skip_indices.add(global_index - args.start)
            
    if skip_indices:
        print(f"Skipping {len(skip_indices)} already completed queries.")
            
    print(f"\n{'='*50}")
    print(f"ReAct Pipeline Configuration")
    print(f"{'='*50}")
    print(f"Provider: {config.llm_provider}")
    if config.llm_provider == "bedrock":
        print(f"Model: {config.bedrock_model_id}")
        print(f"Region: {config.bedrock_region}")
    elif config.llm_provider == "anthropic":
        print(f"Model: {config.anthropic_model_id}")
    elif config.llm_provider == "gemini":
        print(f"Model: {config.gemini_model_id}")
        print(f"Thinking: {config.gemini_thinking_level}")
    else:
        print(f"Model: {config.model_name}")
    print(f"Max tool calls: {config.max_tool_calls}")
    print(f"Temperature: {config.temperature}")
    print(f"Seed: {config.seed}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {output_path}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    results = engine.run_batch(queries, 
                             concurrency=args.concurrency,
                             output_path=output_path,
                             initial_results=previous_results,
                             skip_indices=skip_indices)
    elapsed = time.time() - start_time
    
    print(f"\nCompleted {len(results)} queries in {elapsed:.1f}s")
    
    print(f"Final results saved to {output_path}")
    
    
    final_output = {
        "config": asdict(config),
        "input_file": args.input,
        "concurrency": args.concurrency,
        "total_queries": len(results),
        "elapsed_seconds": elapsed,
        "total_token_usage": asdict(engine.total_token_usage),
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")
    
    feasible = sum(1 for r in results if r.get("plan", {}).get("is_feasible", False))
    avg_calls = sum(r.get("tool_call_count", 0) for r in results) / len(results) if results else 0
    
    print(f"\n{'='*50}")
    print("Statistics")
    print(f"{'='*50}")
    print(f"  Feasible plans: {feasible}/{len(results)} ({100*feasible/len(results):.1f}%)")
    print(f"  Average tool calls: {avg_calls:.2f}")
    print(f"  Total tokens: {engine.total_token_usage.total_tokens:,}")
    print(f"    - Prompt tokens: {engine.total_token_usage.prompt_tokens:,}")
    print(f"    - Completion tokens: {engine.total_token_usage.completion_tokens:,}")


if __name__ == "__main__":
    main()
