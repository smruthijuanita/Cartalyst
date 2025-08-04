#!/usr/bin/env python3
"""
Test script to demonstrate the new LLM-powered query refinement system
"""

from rag import PartsRAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_llm_query_refinement():
    print("🤖 Testing LLM-Powered Query Refinement System")
    print("=" * 60)
    
    # Initialize RAG system
    rag_system = PartsRAG()
    
    # Load data from database
    if not rag_system.load_data_from_db('parts.db'):
        print("❌ Failed to load database")
        return
    
    # Test queries with various complexity levels
    test_queries = [
        "brak pads for honda",
        "alt not working",
        "headlite bulb burnt out", 
        "car won't start battery issue",
        "AC not cooling",
        "I need new shocks for my car",
        "looking for engine oil filter",
        "transmission fluid leak",
        "steering wheel hard to turn",
        "check engine light on"
    ]
    
    print("🔍 Testing Query Refinement Process:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Original Query: '{query}'")
        
        # Test LLM refinement
        if rag_system.groq_client:
            refined_query = rag_system.refine_query_with_llm(query)
            print(f"   LLM Refined: '{refined_query}'")
            
            # Test search with refinement
            results_with_llm = rag_system.search(query, top_k=3, use_llm_refinement=True)
            results_without_llm = rag_system.search(query, top_k=3, use_llm_refinement=False)
            
            print(f"   Results with LLM: {len(results_with_llm)} parts found")
            print(f"   Results without LLM: {len(results_without_llm)} parts found")
            
            if results_with_llm:
                print("   Top result with LLM:")
                top_result = results_with_llm[0]
                print(f"     • {top_result.PartDescription} ({top_result.PartNo})")
        else:
            print("   ⚠️  LLM not available (GROQ_API_KEY not set)")
    
    print("\n" + "=" * 60)
    print("🧪 Testing Complete Search Workflow:")
    print("-" * 40)
    
    # Test the complete search workflow
    test_search_queries = [
        "brake pads honda civic",
        "engine oil filter",
        "headlight bulb replacement"
    ]
    
    for query in test_search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = rag_system.search_with_fallback(query, top_k=5)
        
        if results:
            print(f"   ✅ Found {len(results)} relevant parts:")
            for j, part in enumerate(results[:3], 1):
                print(f"   {j}. {part.PartDescription} - ₹{part.Rate:.2f} ({part.PartNo})")
        else:
            print("   ❌ No parts found")

def test_chatbot_response():
    print("\n" + "=" * 60)
    print("💬 Testing Chatbot Response with LLM Refinement:")
    print("-" * 40)
    
    rag_system = PartsRAG()
    if not rag_system.load_data_from_db('parts.db'):
        print("❌ Failed to load database")
        return
    
    test_chat_queries = [
        "I need brake pads for my Honda",
        "My car's alternator is not working",
        "Looking for headlight bulbs"
    ]
    
    for query in test_chat_queries:
        print(f"\n👤 User: {query}")
        if rag_system.groq_client:
            response = rag_system.generate_llm_response(query)
            print(f"🤖 Caren: {response}")
        else:
            print("🤖 Caren: ⚠️ LLM not available (GROQ_API_KEY not set)")

if __name__ == "__main__":
    test_llm_query_refinement()
    test_chatbot_response()