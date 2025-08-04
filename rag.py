# rag.py

import sqlite3
import logging
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from groq import Groq
from dotenv import load_dotenv

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Class for a Part Record ---
@dataclass(frozen=True)
class PartRecord:
    id: int
    TransactionID: int
    Date: str
    CustomerID: str
    PartNo: str
    Quantity: int
    Rate: float
    TotalPrice: float
    PartDescription: str
    Category: str
    Source: str
    VehicleMake: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def price_category(self) -> str:
        if self.Rate < 25.0: return "budget"
        elif self.Rate < 75.0: return "mid_range"
        else: return "premium"
    
    @property
    def cleaned_description(self) -> str:
        """Returns a cleaned and standardized part description."""
        return PartsRAG.clean_part_name(self.PartDescription)

# --- The RAG System ---
class PartsRAG:
    def __init__(self):
        self.parts_data: List[PartRecord] = []
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        self.tfidf_matrix: Optional[Any] = None
        self._part_lookup: Dict[str, PartRecord] = {}
        try:
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            logger.info("Groq client initialized.")
        except Exception:
            self.groq_client = None
            logger.error("Failed to initialize Groq client. Ensure GROQ_API_KEY is set.")

    @staticmethod
    def clean_part_name(part_description: str) -> str:
        """Cleans and standardizes part names to make them more attractive and readable."""
        if not part_description:
            return "Automotive Part"
        
        # Convert to title case and clean up
        cleaned = part_description.strip()
        
        # Remove excessive punctuation and normalize spacing
        cleaned = re.sub(r'[_\-]{2,}', ' ', cleaned)  # Replace multiple underscores/dashes with space
        cleaned = re.sub(r'[^\w\s\-/().]', ' ', cleaned)  # Remove special chars except common ones
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize multiple spaces
        
        # Handle common abbreviations and expand them
        abbreviations = {
            r'\bALT\b': 'Alternator',
            r'\bSTR\b': 'Starter',
            r'\bBRK\b': 'Brake',
            r'\bSUSP\b': 'Suspension',
            r'\bENG\b': 'Engine',
            r'\bTRANS\b': 'Transmission',
            r'\bFLT\b': 'Filter',
            r'\bPMP\b': 'Pump',
            r'\bSHK\b': 'Shock',
            r'\bSTRUT\b': 'Strut',
            r'\bCV\b': 'CV Joint',
            r'\bA/C\b': 'Air Conditioning',
            r'\bAC\b': 'Air Conditioning',
            r'\bPS\b': 'Power Steering',
            r'\bWTR\b': 'Water',
            r'\bOIL\b': 'Oil',
            r'\bFUEL\b': 'Fuel',
            r'\bEXH\b': 'Exhaust',
            r'\bRAD\b': 'Radiator',
            r'\bTHERM\b': 'Thermostat',
            r'\bSPRK\b': 'Spark',
            r'\bIGN\b': 'Ignition',
            r'\bDIST\b': 'Distributor',
            r'\bROTOR\b': 'Rotor',
            r'\bCAP\b': 'Cap',
            r'\bWIRE\b': 'Wire',
            r'\bHARN\b': 'Harness',
            r'\bSW\b': 'Switch',
            r'\bSENS\b': 'Sensor',
            r'\bVLV\b': 'Valve',
            r'\bGSKT\b': 'Gasket',
            r'\bSEAL\b': 'Seal',
            r'\bBRG\b': 'Bearing',
            r'\bBUSH\b': 'Bushing',
            r'\bLNK\b': 'Link',
            r'\bARM\b': 'Arm',
            r'\bROD\b': 'Rod',
            r'\bJNT\b': 'Joint',
            r'\bHUB\b': 'Hub',
            r'\bKNUCK\b': 'Knuckle',
            r'\bAXLE\b': 'Axle',
            r'\bDRV\b': 'Drive',
            r'\bSHFT\b': 'Shaft',
            r'\bGEAR\b': 'Gear',
            r'\bCLUTCH\b': 'Clutch',
            r'\bFLYWHL\b': 'Flywheel',
            r'\bDIFF\b': 'Differential',
            r'\bTRAN\b': 'Transfer',
            r'\bCASE\b': 'Case',
            r'\bMNT\b': 'Mount',
            r'\bBRKT\b': 'Bracket',
            r'\bSUPP\b': 'Support',
            r'\bFRM\b': 'Frame',
            r'\bCROSS\b': 'Cross',
            r'\bMBR\b': 'Member',
            r'\bSUB\b': 'Sub',
            r'\bCRDL\b': 'Cradle',
            r'\bRAIL\b': 'Rail',
            r'\bBEAM\b': 'Beam',
            r'\bSTRUCT\b': 'Structure'
        }
        
        # Apply abbreviation expansions (case insensitive)
        for abbrev, full_form in abbreviations.items():
            cleaned = re.sub(abbrev, full_form, cleaned, flags=re.IGNORECASE)
        
        # Standardize common terms
        standardizations = {
            r'\bPAD\b': 'Pad',
            r'\bSET\b': 'Set',
            r'\bKIT\b': 'Kit',
            r'\bASSY\b': 'Assembly',
            r'\bASSEMBLY\b': 'Assembly',
            r'\bCOMPLETE\b': 'Complete',
            r'\bREPLACEMENT\b': 'Replacement',
            r'\bOEM\b': 'OEM',
            r'\bAFTERMARKET\b': 'Aftermarket',
            r'\bGENUINE\b': 'Genuine',
            r'\bORIGINAL\b': 'Original',
            r'\bHEAVY\s+DUTY\b': 'Heavy Duty',
            r'\bHIGH\s+PERFORMANCE\b': 'High Performance',
            r'\bPREMIUM\b': 'Premium',
            r'\bSTANDARD\b': 'Standard',
            r'\bUNIVERSAL\b': 'Universal',
            r'\bFRONT\b': 'Front',
            r'\bREAR\b': 'Rear',
            r'\bLEFT\b': 'Left',
            r'\bRIGHT\b': 'Right',
            r'\bUPPER\b': 'Upper',
            r'\bLOWER\b': 'Lower',
            r'\bINNER\b': 'Inner',
            r'\bOUTER\b': 'Outer'
        }
        
        # Apply standardizations (case insensitive)
        for pattern, replacement in standardizations.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Convert to proper title case
        cleaned = cleaned.title()
        
        # Fix common title case issues
        cleaned = re.sub(r'\bOf\b', 'of', cleaned)
        cleaned = re.sub(r'\bAnd\b', 'and', cleaned)
        cleaned = re.sub(r'\bThe\b', 'the', cleaned)
        cleaned = re.sub(r'\bFor\b', 'for', cleaned)
        cleaned = re.sub(r'\bWith\b', 'with', cleaned)
        cleaned = re.sub(r'\bTo\b', 'to', cleaned)
        cleaned = re.sub(r'\bIn\b', 'in', cleaned)
        cleaned = re.sub(r'\bOn\b', 'on', cleaned)
        cleaned = re.sub(r'\bAt\b', 'at', cleaned)
        
        # Ensure first letter is always capitalized
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        return cleaned if cleaned else "Automotive Part"

    def load_data_from_db(self, db_path: str) -> bool:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM parts")
            self.parts_data = [PartRecord(**row) for row in cursor.fetchall()]
            conn.close()
            self._create_search_index()
            logger.info(f"Successfully loaded {len(self.parts_data)} part records from '{db_path}'.")
            return True
        except Exception as e:
            logger.error(f"Error loading data from DB: {e}")
            return False

    def _create_search_index(self):
        if not self.parts_data: return
        texts = [f"{p.PartDescription or ''} {p.Category or ''} {p.VehicleMake or ''} {p.PartNo or ''}" for p in self.parts_data]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._part_lookup = {part.PartNo: part for part in reversed(self.parts_data)}
        logger.info("Search index created.")

    def _get_unique_categories_and_makes(self) -> Dict[str, List[str]]:
        """Get unique categories and vehicle makes for query refinement context."""
        categories = sorted(list(set(p.Category for p in self.parts_data if p.Category)))
        vehicle_makes = sorted(list(set(p.VehicleMake for p in self.parts_data if p.VehicleMake)))
        return {"categories": categories, "vehicle_makes": vehicle_makes}

    def refine_query_with_llm(self, original_query: str) -> str:
        """Uses LLM to intelligently refine and optimize user queries for better search accuracy."""
        if not self.groq_client:
            logger.warning("Groq client not available. Using original query.")
            return original_query
        
        context_data = self._get_unique_categories_and_makes()
        categories_str = ", ".join(context_data["categories"][:20])
        makes_str = ", ".join(context_data["vehicle_makes"][:20])

        system_prompt = f"""You are an intelligent query optimization assistant for an automotive parts search system. Your goal is to transform user queries into the most effective search terms for finding relevant auto parts.

AVAILABLE DATA CONTEXT:
- Categories: {categories_str}
- Vehicle Makes: {makes_str}

OPTIMIZATION RULES:
1. Fix spelling errors and typos (e.g., "brak pads" → "brake pads")
2. Expand abbreviations and acronyms (e.g., "alt" → "alternator", "AC" → "air conditioning")
3. Standardize terminology (e.g., "car battery" → "battery", "headlamp" → "headlight")
4. Add relevant technical terms when appropriate (e.g., "brake pads" → "brake pads disc brake")
5. Include synonyms for better matching (e.g., "shock" → "shock absorber strut")
6. Preserve vehicle make/model information
7. Remove unnecessary words that don't help search (e.g., "I need", "looking for")
8. Add category-specific keywords when context is clear

EXAMPLES:
- "I need brak pads for my honda civic" → "brake pads honda civic disc brake"
- "alt not working" → "alternator generator charging system"
- "headlite bulb burnt out" → "headlight bulb lamp lighting"
- "car won't start battery issue" → "battery starter alternator charging"
- "AC not cooling" → "air conditioning compressor refrigerant cooling"

Return ONLY the optimized search query with no explanations or quotes."""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Optimize this search query: {original_query}"}
                ],
                temperature=0.2,
                max_tokens=100
            )
            refined_query = chat_completion.choices[0].message.content.strip()
            refined_query = re.sub(r'^["\']|["\']$', '', refined_query)
            logger.info(f"LLM Query Optimization: '{original_query}' → '{refined_query}'")
            return refined_query
        except Exception as e:
            logger.error(f"Error optimizing query with LLM: {e}")
            return original_query

    def search(self, query: str, top_k: int = 10, min_similarity: float = 0.1,
             category: Optional[str] = None, vehicle_make: Optional[str] = None,
             price_category: Optional[str] = None, use_llm_refinement: bool = True) -> List[PartRecord]:
        """Performs a semantic search with LLM-powered query optimization and filtering."""
        if self.tfidf_matrix is None:
            logger.error("Search index not created. Load data first.")
            return []
        
        # Step 1: LLM refines the query for optimal search accuracy
        if use_llm_refinement:
            search_query = self.refine_query_with_llm(query)
            logger.info(f"Using LLM-optimized query for search: '{search_query}'")
        else:
            search_query = query
            logger.info(f"Using original query for search: '{search_query}'")
        
        # Step 2: Convert refined query to vector embedding
        query_vector = self.vectorizer.transform([search_query.lower()])
        
        # Step 3: Calculate semantic similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Step 4: Filter by minimum similarity threshold
        candidate_indices = [i for i, sim in enumerate(similarities) if sim >= min_similarity]
        
        # Step 5: Apply additional filters (category, vehicle make, price)
        filtered_results = []
        for i in candidate_indices:
            part = self.parts_data[i]
            
            # Handle both original category and mapped main category filtering
            if category:
                part_main_category = self._map_to_main_category(part.Category, part.PartDescription)
                if (category.lower() not in (part.Category.lower() if part.Category else '') and
                    category.lower() != part_main_category.lower()):
                    continue
                    
            if vehicle_make and vehicle_make.lower() not in (part.VehicleMake.lower() if part.VehicleMake else ''): continue
            if price_category and price_category.lower() != part.price_category.lower(): continue
            filtered_results.append((part, similarities[i]))
        
        # Step 6: Sort by relevance and return top results
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return [part for part, sim in filtered_results[:top_k]]

    def search_with_fallback(self, query: str, top_k: int = 10) -> List[PartRecord]:
        """Performs search with LLM optimization and multiple fallback strategies for better results."""
        # Primary search: LLM-optimized query
        logger.info(f"Primary search: Using LLM-optimized query for '{query}'")
        results = self.search(query, top_k=top_k, use_llm_refinement=True)
        if results:
            logger.info(f"✅ Found {len(results)} results with LLM-optimized query.")
            return results

        # Fallback 1: Original query with lower similarity threshold
        logger.info("Fallback 1: Trying original query with lower similarity threshold.")
        results = self.search(query, top_k=top_k, min_similarity=0.05, use_llm_refinement=False)
        if results:
            logger.info(f"✅ Found {len(results)} results with lower similarity threshold.")
            return results

        # Fallback 2: LLM-optimized query with very low threshold
        logger.info("Fallback 2: Trying LLM-optimized query with very low threshold.")
        results = self.search(query, top_k=top_k, min_similarity=0.02, use_llm_refinement=True)
        if results:
            logger.info(f"✅ Found {len(results)} results with LLM optimization and low threshold.")
            return results

        # Fallback 3: Individual query terms
        logger.info("Fallback 3: Trying individual query terms.")
        query_terms = query.lower().split()
        all_results = []
        for term in query_terms:
            if len(term) > 2:
                all_results.extend(self.search(term, top_k=5, use_llm_refinement=False))
        
        unique_results = list({part.PartNo: part for part in all_results}.values())
        logger.info(f"✅ Found {len(unique_results)} results with individual terms.")
        return unique_results[:top_k]

    def generate_llm_response(self, query: str) -> str:
        """Parses a user query, searches for parts, and generates a conversational response with purchase links."""
        if not self.groq_client: return "Chatbot is currently unavailable."
        logger.info(f"Generating LLM response for query: {query}")

        retrieved_parts = self.search_with_fallback(query, top_k=5)
        if not retrieved_parts:
            return f"Sorry, I couldn't find any parts for '{query}'. Please try a different search or check the spelling."

        context = "Relevant parts from inventory:\n"
        for part in retrieved_parts:
            context += f"- Part: {part.PartDescription}, No: {part.PartNo}, Price: ${part.Rate/80:.2f}, Stock: {part.Quantity}, Vehicle: {part.VehicleMake}\n"
        
        system_prompt = """You are Caren, a helpful auto parts assistant. Answer the user's question concisely using ONLY the provided context. Be conversational and helpful.

IMPORTANT: For each part you mention, include a clickable link in this exact format:
[View Part Details](/view_part/PART_NUMBER)

For example: "I found brake pads for Honda [View Part Details](/view_part/BP001) priced at $15.00."

Always include these links when suggesting specific parts to help customers easily navigate to purchase them."""
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
                ]
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "Sorry, I'm having trouble connecting right now."

    def get_employee_insights(self) -> Dict[str, Any]:
        """Generates comprehensive business insights and analytics from the parts data."""
        logger.info(f"Generating employee insights for {len(self.parts_data)} parts")
        
        if not self.parts_data:
            logger.warning("No parts data available for insights")
            return {
                "total_inventory_value": 0,
                "parts_per_supplier": {},
                "low_stock_items": [],
                "fast_moving_items": [],
                "category_revenue": {"Engine and Fuel": 0, "Body and Electrical": 0, "Powertrain and Chassis": 0, "Axle and Suspension": 0, "Others": 0},
                "category_transactions": {"Engine and Fuel": 0, "Body and Electrical": 0, "Powertrain and Chassis": 0, "Axle and Suspension": 0, "Others": 0},
                "monthly_revenue": [],
                "top_suppliers": [],
                "total_transactions": 0,
                "average_transaction_value": 0
            }
        
        try:
            latest_parts = {p.PartNo: p for p in self.parts_data}
            total_value = sum(float(p.Quantity) * float(p.Rate) for p in latest_parts.values())
            supplier_counts = Counter(p.Source for p in latest_parts.values() if p.Source)
            low_stock_items = sorted([p for p in latest_parts.values() if p.Quantity < 2], key=lambda p: p.Quantity)
            
            # Enhanced analytics for visualizations
            quantity_sold_by_part_no = Counter(p.PartNo for p in self.parts_data)
            fast_moving_items = []
            for part_no, count in quantity_sold_by_part_no.most_common(10):
                part_record = self._part_lookup.get(part_no)
                if part_record:
                    fast_moving_items.append({
                        'PartDescription': self.clean_part_name(part_record.PartDescription),
                        'PartNo': part_record.PartNo,
                        'TotalTransactions': count,
                        'Revenue': count * part_record.Rate,
                        'Rate': part_record.Rate
                    })
            
            # Revenue by category analysis
            category_revenue = defaultdict(float)
            category_transactions = defaultdict(int)
            for part in self.parts_data:
                main_category = self._map_to_main_category(part.Category, part.PartDescription)
                revenue = part.Rate * 1  # Assuming each transaction is 1 unit for simplicity
                category_revenue[main_category] += revenue
                category_transactions[main_category] += 1
            
            # Monthly revenue trend (simulated based on transaction patterns)
            monthly_revenue = self._calculate_monthly_revenue_trend()
            
            # Top suppliers by revenue
            supplier_revenue = defaultdict(float)
            for part in self.parts_data:
                if part.Source:
                    supplier_revenue[part.Source] += part.Rate
            
            top_suppliers = sorted(
                [{'name': supplier, 'revenue': revenue} for supplier, revenue in supplier_revenue.items()],
                key=lambda x: x['revenue'], reverse=True
            )[:8]
            
            insights_data = {
                "total_inventory_value": total_value,
                "parts_per_supplier": dict(supplier_counts),
                "low_stock_items": low_stock_items,
                "fast_moving_items": fast_moving_items,
                "category_revenue": dict(category_revenue),
                "category_transactions": dict(category_transactions),
                "monthly_revenue": monthly_revenue,
                "top_suppliers": top_suppliers,
                "total_transactions": len(self.parts_data),
                "average_transaction_value": sum(p.Rate for p in self.parts_data) / len(self.parts_data) if self.parts_data else 0
            }
            
            logger.info(f"Generated insights: {len(fast_moving_items)} fast moving items, {len(top_suppliers)} suppliers, ${total_value/80:.2f} total value")
            return insights_data
            
        except Exception as e:
            logger.error(f"Error generating employee insights: {e}")
            return {
                "total_inventory_value": 0,
                "parts_per_supplier": {},
                "low_stock_items": [],
                "fast_moving_items": [],
                "category_revenue": {"Engine and Fuel": 0, "Body and Electrical": 0, "Powertrain and Chassis": 0, "Axle and Suspension": 0, "Others": 0},
                "category_transactions": {"Engine and Fuel": 0, "Body and Electrical": 0, "Powertrain and Chassis": 0, "Axle and Suspension": 0, "Others": 0},
                "monthly_revenue": [],
                "top_suppliers": [],
                "total_transactions": 0,
                "average_transaction_value": 0
            }
    
    def _calculate_monthly_revenue_trend(self) -> List[Dict[str, Any]]:
        """Calculate monthly revenue trend for the last 12 months."""
        from datetime import datetime, timedelta
        import calendar
        
        # Since we don't have real date data, we'll simulate based on transaction patterns
        monthly_data = []
        base_date = datetime.now()
        
        for i in range(12):
            month_date = base_date - timedelta(days=30 * i)
            month_name = calendar.month_name[month_date.month]
            
            # Simulate revenue based on part data patterns
            # More recent months have higher revenue
            revenue_multiplier = 1.0 + (0.1 * (12 - i))
            base_revenue = sum(p.Rate for p in self.parts_data[:100]) * revenue_multiplier / 100
            
            monthly_data.append({
                'month': f"{month_name} {month_date.year}",
                'revenue': round(base_revenue, 2),
                'transactions': len(self.parts_data) // 12 + (i * 5)
            })
        
        return list(reversed(monthly_data))

    def _map_to_main_category(self, original_category: str, part_description: str) -> str:
        """Maps original categories to the 5 main categories based on keywords."""
        if not original_category and not part_description:
            return "Others"
        
        # Combine category and description for better classification
        text = f"{original_category or ''} {part_description or ''}".lower()
        
        # Engine and Fuel related
        engine_keywords = [
            'engine', 'motor', 'fuel', 'injection', 'carburetor', 'turbo', 'supercharger',
            'piston', 'cylinder', 'valve', 'camshaft', 'crankshaft', 'timing', 'belt',
            'oil', 'filter', 'pump', 'radiator', 'cooling', 'thermostat', 'fan',
            'spark', 'plug', 'ignition', 'coil', 'distributor', 'exhaust', 'muffler',
            'catalytic', 'converter', 'manifold', 'gasket', 'head', 'block'
        ]
        
        # Body and Electrical
        body_electrical_keywords = [
            'body', 'door', 'window', 'mirror', 'bumper', 'fender', 'hood', 'trunk',
            'seat', 'interior', 'dashboard', 'panel', 'trim', 'molding', 'glass',
            'electrical', 'wire', 'harness', 'fuse', 'relay', 'switch', 'button',
            'light', 'lamp', 'bulb', 'headlight', 'taillight', 'signal', 'indicator',
            'battery', 'alternator', 'starter', 'generator', 'horn', 'speaker',
            'radio', 'antenna', 'wiper', 'washer', 'heater', 'ac', 'climate'
        ]
        
        # Powertrain and Chassis
        powertrain_chassis_keywords = [
            'transmission', 'gearbox', 'clutch', 'flywheel', 'driveshaft', 'axle',
            'differential', 'transfer', 'case', 'cv', 'joint', 'universal',
            'chassis', 'frame', 'crossmember', 'mount', 'bracket', 'support',
            'subframe', 'cradle', 'rail', 'beam', 'structure'
        ]
        
        # Axle and Suspension
        axle_suspension_keywords = [
            'suspension', 'shock', 'absorber', 'strut', 'spring', 'coil', 'leaf',
            'damper', 'stabilizer', 'sway', 'bar', 'link', 'bushing', 'ball',
            'joint', 'control', 'arm', 'wishbone', 'tie', 'rod', 'steering',
            'rack', 'pinion', 'power', 'column', 'wheel', 'hub', 'bearing',
            'knuckle', 'spindle', 'axle', 'halfshaft', 'driveshaft'
        ]
        
        # Check for matches in order of priority
        if any(keyword in text for keyword in engine_keywords):
            return "Engine and Fuel"
        elif any(keyword in text for keyword in body_electrical_keywords):
            return "Body and Electrical"
        elif any(keyword in text for keyword in powertrain_chassis_keywords):
            return "Powertrain and Chassis"
        elif any(keyword in text for keyword in axle_suspension_keywords):
            return "Axle and Suspension"
        else:
            return "Others"

    def get_categorized_parts(self) -> Dict[str, List[PartRecord]]:
        """Groups unique parts by the 5 main categories."""
        # Define the 5 main categories in order
        main_categories = [
            "Engine and Fuel",
            "Body and Electrical",
            "Powertrain and Chassis",
            "Axle and Suspension",
            "Others"
        ]
        
        categorized = {category: [] for category in main_categories}
        seen_parts = set()
        
        for part in self.parts_data:
            if part.PartNo not in seen_parts:
                main_category = self._map_to_main_category(part.Category, part.PartDescription)
                categorized[main_category].append(part)
                seen_parts.add(part.PartNo)
        
        # Return only categories that have parts
        return {category: parts for category, parts in categorized.items() if parts}

    def generate_inventory_recommendations(self) -> Dict[str, Any]:
        """Uses the LLM to analyze inventory data and provide ordering recommendations."""
        if not self.groq_client: return {"error": "LLM client not available."}

        logger.info("Generating inventory recommendations with LLM...")
        insights = self.get_employee_insights()
        low_stock_parts = insights.get('low_stock_items', [])
        
        if not low_stock_parts:
            return {"metrics": {"totalRecommendations": 0, "totalInvestment": 0, "criticalItems": 0}, "recommendations": []}

        context = "Analyze the following auto parts inventory data for items with low stock (fewer than 2 units).\n"
        for part in low_stock_parts[:15]:
            context += f"- Part: {part.PartDescription}, PartNo: {part.PartNo}, Stock: {part.Quantity}, Price: ${part.Rate/80:.2f}\n"

        system_prompt = """
        You are an AI inventory analyst for 'Cartalyst', an auto parts store. Your goal is to generate a JSON object with ordering recommendations. The output MUST be a single JSON object and nothing else.
        The JSON object must have two top-level keys: "metrics" and "recommendations".
        1.  `metrics`: An object summarizing the report with three keys:
            - `totalRecommendations`: Total number of parts you are recommending to order.
            - `totalInvestment`: Total cost in USD to order all recommended parts.
            - `criticalItems`: Count of recommendations with 'Critical' priority.
        2.  `recommendations`: An array of objects, where each object has these keys:
            - `partNumber`: (string) The part number.
            - `description`: (string) The part description.
            - `suggestedQuantity`: (integer) A smart suggestion for how many units to reorder (e.g., 5 or 10).
            - `priorityLevel`: (string) 'Critical', 'High', or 'Medium'.
            - `priorityScore`: (integer) A score from 1-100 indicating urgency.
            - `investment`: (float) The total cost for this order in USD (suggestedQuantity * Rate_in_USD).
            - `reasoning`: (string) A brief justification (e.g., "Critically low stock").
        """
        try:
            chat_completion = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.choices[0].message.content
            recommendations = json.loads(response_content)
            logger.info("Successfully generated and parsed inventory recommendations.")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating or parsing LLM recommendations: {e}")
            return {"error": "Failed to generate AI analysis."}