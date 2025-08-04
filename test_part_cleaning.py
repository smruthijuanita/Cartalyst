#!/usr/bin/env python3
"""
Test script to demonstrate the part name cleaning and standardization functionality
"""

from rag import PartsRAG

def test_part_name_cleaning():
    print("🧹 Testing Part Name Cleaning and Standardization")
    print("=" * 60)
    
    # Test various messy part names
    test_names = [
        "BRK_PAD_SET_FRT_HONDA_CIVIC",
        "alt__12v__replacement__kit",
        "SHOCK_ABSORBER_REAR_LEFT_OEM",
        "headlite_bulb_h4_55w_12v",
        "eng_oil_flt_premium_quality",
        "FUEL_PMP_ASSY_COMPLETE_KIT",
        "strut_mnt_upper_frnt_right",
        "cv_jnt_outer_drv_shaft",
        "ign_coil_pack_set_4cyl",
        "rad_cap_16psi_universal",
        "AC_COMP_CLUTCH_KIT_R134A",
        "ps_pump_rebuild_kit_oem",
        "exh_manifold_gasket_set",
        "timing_belt_kit_complete",
        "wtr_pump_thermostat_kit"
    ]
    
    print("🔧 Original vs Cleaned Part Names:")
    print("-" * 60)
    
    for i, original_name in enumerate(test_names, 1):
        cleaned_name = PartsRAG.clean_part_name(original_name)
        print(f"{i:2d}. Original: {original_name}")
        print(f"    Cleaned:  {cleaned_name}")
        print()
    
    print("✨ Benefits of Part Name Cleaning:")
    print("-" * 40)
    print("• Removes underscores and excessive punctuation")
    print("• Expands common abbreviations (BRK → Brake, ALT → Alternator)")
    print("• Standardizes terminology (FLT → Filter, PMP → Pump)")
    print("• Proper title case formatting")
    print("• More attractive and professional appearance")
    print("• Better readability for customers")

def test_with_real_database():
    print("\n" + "=" * 60)
    print("📊 Testing with Real Database Parts:")
    print("-" * 40)
    
    # Initialize RAG system
    rag_system = PartsRAG()
    
    # Load data from database
    if not rag_system.load_data_from_db('parts.db'):
        print("❌ Failed to load database")
        return
    
    # Get a sample of parts from each category
    categorized_parts = rag_system.get_categorized_parts()
    
    for category, parts in list(categorized_parts.items())[:3]:  # Show first 3 categories
        print(f"\n🏷️  {category} Category:")
        print("-" * 30)
        
        for i, part in enumerate(parts[:5], 1):  # Show first 5 parts
            original = part.PartDescription
            cleaned = part.cleaned_description
            print(f"{i}. Original: {original}")
            print(f"   Cleaned:  {cleaned}")
            print()

if __name__ == "__main__":
    test_part_name_cleaning()
    test_with_real_database()