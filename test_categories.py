#!/usr/bin/env python3
"""
Test script to verify the new 5-category classification system
"""

from rag import PartsRAG

def test_categorization():
    print("🔍 Testing the new 5-category classification system...")
    
    # Initialize RAG system
    rag_system = PartsRAG()
    
    # Load data from database
    if not rag_system.load_data_from_db('parts.db'):
        print("❌ Failed to load database")
        return
    
    # Get categorized parts
    categorized_parts = rag_system.get_categorized_parts()
    
    print(f"\n📊 Found {len(categorized_parts)} categories:")
    print("=" * 60)
    
    total_parts = 0
    for category, parts in categorized_parts.items():
        print(f"📁 {category}: {len(parts)} parts")
        total_parts += len(parts)
        
        # Show a few example parts from each category
        print("   Examples:")
        for i, part in enumerate(parts[:3]):
            print(f"   • {part.PartDescription} (Original: {part.Category})")
        if len(parts) > 3:
            print(f"   ... and {len(parts) - 3} more")
        print()
    
    print(f"📈 Total unique parts classified: {total_parts}")
    
    # Test some specific mappings
    print("\n🧪 Testing specific category mappings:")
    test_cases = [
        ("Engine", "Engine Oil Filter"),
        ("Brake", "Brake Pad Set"),
        ("Suspension", "Shock Absorber"),
        ("Electrical", "Headlight Bulb"),
        ("Body", "Door Handle"),
        ("Transmission", "Clutch Kit"),
        ("", "Unknown Part Description")
    ]
    
    for original_cat, description in test_cases:
        mapped = rag_system._map_to_main_category(original_cat, description)
        print(f"   '{original_cat}' + '{description}' → {mapped}")

if __name__ == "__main__":
    test_categorization()