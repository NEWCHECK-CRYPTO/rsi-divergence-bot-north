"""
RAG Module using Google Gemini
"""

import json
from typing import Dict, Optional
import google.generativeai as genai
from config import GEMINI_API_KEY, RAG_KNOWLEDGE_PATH, GEMINI_MODEL


class TradingKnowledgeRAG:
    def __init__(self, knowledge_path: str = RAG_KNOWLEDGE_PATH):
        self.knowledge_path = knowledge_path
        self.knowledge_base = None
        self.model = None
        self._initialize()
    
    def _initialize(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={"temperature": 0.3, "top_p": 0.95, "max_output_tokens": 1024}
        )
        self.knowledge_base = self._load_knowledge_base()
        self.system_context = self._create_system_context()
    
    def _load_knowledge_base(self) -> Dict:
        try:
            with open(self.knowledge_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _create_system_context(self) -> str:
        if not self.knowledge_base:
            return "You are a trading assistant specializing in RSI divergence analysis."
        
        parts = ["You are an expert trading assistant using EmperorBTC methodology.", "", "=== KNOWLEDGE ===", ""]
        
        if "timeframe_mappings" in self.knowledge_base:
            parts.append("TIMEFRAME MAPPINGS:")
            for m in self.knowledge_base["timeframe_mappings"]["mappings"]:
                parts.append(f"- {m['signal_name']} ({m['signal_tf']}) → {m['confirm_name']} ({m['confirmation_tf']})")
            parts.append("")
        
        if "rsi_divergence" in self.knowledge_base:
            rsi = self.knowledge_base["rsi_divergence"]
            parts.append("BULLISH DIVERGENCE (Look at LOWS):")
            for dtype, d in rsi.get("bullish_divergence", {}).get("types", {}).items():
                parts.append(f"- {d['name']}: {d['price_action']} + {d['rsi_action']} = {d['confidence']*100}%")
            parts.append("")
            parts.append("BEARISH DIVERGENCE (Look at HIGHS):")
            for dtype, d in rsi.get("bearish_divergence", {}).get("types", {}).items():
                parts.append(f"- {d['name']}: {d['price_action']} + {d['rsi_action']} = {d['confidence']*100}%")
            parts.append("")
        
        if "market_structure" in self.knowledge_base:
            ms = self.knowledge_base["market_structure"]
            parts.append("MARKET STRUCTURE:")
            for term, defn in ms.get("vocabulary", {}).items():
                parts.append(f"- {term}: {defn}")
            parts.append("")
        
        return "\n".join(parts)
    
    def query(self, question: str) -> str:
        if self.model is None:
            return "RAG not initialized."
        
        prompt = f"{self.system_context}\n\n=== QUESTION ===\n{question}\n\nAnswer based on the knowledge above. Be specific and practical."
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"


class SimpleKnowledgeBase:
    def __init__(self, knowledge_path: str = RAG_KNOWLEDGE_PATH):
        self.knowledge = self._load(knowledge_path)
    
    def _load(self, path: str) -> Dict:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def get_divergence_info(self, divergence_type: str) -> Optional[Dict]:
        div_type = divergence_type.lower().replace(" ", "_")
        if "rsi_divergence" in self.knowledge:
            for section in ["bullish_divergence", "bearish_divergence"]:
                types = self.knowledge["rsi_divergence"].get(section, {}).get("types", {})
                if div_type in types:
                    return types[div_type]
        return None
    
    def get_confirmation_tf(self, signal_tf: str) -> Optional[str]:
        if "timeframe_mappings" in self.knowledge:
            for m in self.knowledge["timeframe_mappings"]["mappings"]:
                if m["signal_tf"].lower() == signal_tf.lower():
                    return m["confirmation_tf"]
        return None
    
    def get_vocabulary(self) -> Dict:
        return self.knowledge.get("market_structure", {}).get("vocabulary", {})
