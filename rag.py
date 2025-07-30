# rag.py

import sqlite3
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from groq import Groq

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Class for a Part Record ---
@dataclass(frozen=True)
class PartRecord:
    """A structured representation of a single part transaction."""
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
        """Converts the dataclass instance to a dictionary."""
        return asdict(self)
    
    @property
    def price_category(self) -> str:
        """Categorizes the part into 'budget', 'mid_range', or 'premium' based on its rate."""
        if self.Rate < 25.0:
            return "budget"
        elif self.Rate < 75.0:
            return "mid_range"
        else:
            return "premium"

# --- The RAG (Search & Analytics) System ---
class PartsRAG:
    """
    A system to handle loading, searching, and analyzing auto parts data.
    It uses a TF-IDF vectorizer for semantic search and provides methods
    for generating business insights from the inventory data.
    """
    def __init__(self):
        """Initializes the PartsRAG system."""
        self.parts_data: List[PartRecord] = []
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        self.tfidf_matrix: Optional[Any] = None
        self._part_lookup: Dict[str, PartRecord] = {} 
            # --- Initialize the Groq LLM Client ---
        try:
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            logger.info("Groq client initialized successfully.")
        except Exception as e:
            self.groq_client = None
            logger.error("Failed to initialize Groq client. Ensure GROQ_API_KEY is set.", exc_info=True)

    def load_data_from_db(self, db_path: str) -> bool:
        """
        Loads parts data from a SQLite database, builds the search index,
        and returns True on success.
        
        Args:
            db_path: The file path to the SQLite database.

        Returns:
            True if data was loaded successfully, False otherwise.
        """
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Access columns by name
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM parts")
            
            self.parts_data = [PartRecord(**row) for row in cursor.fetchall()]
            conn.close()
            
            self._create_search_index()
            logger.info(f"Successfully loaded {len(self.parts_data)} parts from '{db_path}'.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database error while loading from '{db_path}': {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}")
            return False

    def _create_search_index(self):
        """
        Builds the TF-IDF matrix for semantic search and a part lookup dictionary.
        This internal method is called by load_data_from_db.
        """
        if not self.parts_data:
            logger.warning("No data available to build search index.")
            return

        # Create combined text for vectorization
        texts = [
            f"{p.PartDescription} {p.Category} {p.VehicleMake} {p.PartNo}" 
            for p in self.parts_data
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Create a lookup for quick access to part details by PartNo
        self._part_lookup = {part.PartNo: part for part in reversed(self.parts_data)}
        logger.info("Search index and part lookup created successfully.")

    def search(self, query: str, top_k: int = 10, min_similarity: float = 0.1) -> List[PartRecord]:
        """
        Performs a semantic search for parts based on a text query.

        Args:
            query: The user's search term (e.g., "brake pads for honda").
            top_k: The maximum number of results to return.
            min_similarity: The minimum cosine similarity score to consider a match.

        Returns:
            A list of matching PartRecord objects, sorted by relevance.
        """
        if self.tfidf_matrix is None:
            logger.error("Search index not created. Load data first.")
            return []
            
        query_vector = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices of parts sorted by similarity, filtering by the threshold
        top_indices = [
            i for i in similarities.argsort()[::-1] 
            if similarities[i] >= min_similarity
        ][:top_k]
        
        return [self.parts_data[i] for i in top_indices]

    def generate_search_response(self, query: str) -> str:
        """
        Searches for parts and formats the results into a user-friendly markdown string.

        Args:
            query: The user's search query.

        Returns:
            A formatted string with search results or a 'not found' message.
        """
        results = self.search(query)
        if not results:
            return f" **No parts found for '{query}'.**\n\nTry using more general terms or check your spelling."
        
        response_parts = [f"**Found {len(results)} relevant parts for '{query}':**\n"]
        for i, part in enumerate(results, 1):
            response_parts.append(
                f"**{i}. {part.PartDescription}**\n"
                f"   • **Part No:** `{part.PartNo}`\n"
                f"   • **Price:** ₹{part.Rate:.2f}\n"
                f"   • **Category:** {part.Category}\n"
                f"   • **Vehicle:** {part.VehicleMake}\n"
                f"   • **In Stock:** {part.Quantity} units\n"
            )
        return "\n".join(response_parts)

    def get_employee_insights(self, fast_moving_count: int = 5) -> Dict[str, Any]:
        """
        Generates a dictionary of key business insights for an employee dashboard.

        Args:
            fast_moving_count: The number of top-selling items to identify.

        Returns:
            A dictionary containing total inventory value, supplier stats,
            price range distribution, low stock items, and fast-moving items.
        """
        if not self.parts_data:
            return {}

        total_value = sum(float(p.TotalPrice) for p in self.parts_data)
        supplier_counts = Counter(p.Source for p in self.parts_data)
        price_range_counts = Counter(p.price_category for p in self.parts_data)
        low_stock_items = sorted([p for p in self.parts_data if p.Quantity < 2], key=lambda p: p.Quantity)

        # Identify fast-moving items efficiently
        quantity_by_part_no = Counter()
        for part in self.parts_data:
            quantity_by_part_no[part.PartNo] += part.Quantity
            
        fast_moving_items = []
        for part_no, total_quantity in quantity_by_part_no.most_common(fast_moving_count):
            # Use the pre-built lookup for fast access to part details
            part_record = self._part_lookup.get(part_no)
            if part_record:
                fast_moving_items.append({
                    'PartNo': part_no,
                    'PartDescription': part_record.PartDescription,
                    'TotalQuantitySold': total_quantity
                })
        
        return {
            "total_inventory_value": total_value,
            "parts_per_supplier": dict(supplier_counts),
            "parts_in_price_ranges": dict(price_range_counts),
            "low_stock_items": low_stock_items,
            "fast_moving_items": fast_moving_items,
        }
    def get_categorized_parts(self) -> Dict[str, List[PartRecord]]:
        """
        Groups all parts by their category for Browse.

        Returns:
            A dictionary where keys are category names and values are lists of parts.
        """
        categorized = defaultdict(list)
        for part in self.parts_data:
            categorized[part.Category].append(part)
        return dict(sorted(categorized.items()))
    def generate_llm_response(self, query: str) -> str:
        """
        Uses the Groq LLM to genreate a response to a user query."""

        if not self.groq_client:
            return "Chatbot is currently unavailable. Please try again later."
        logger.info(f"Generating LLM response for query: {query}")

        # 1. Retrieve top 5 relevant parts
        retrieved_parts = self.search(query , top_k=5, min_similarity=0.1)
        if not retrieved_parts:
            return f"Sorry, I couldn't find any relevant parts for '{query}'. Please try a different search."
        
        # 2. Format the retrieved parts for the LLM
        context = "Here are the most relevant parts found in our inventory:\n"
        for part in retrieved_parts:
            context += (
                f"- Part: {part.PartDescription}\n"
                f"  - Part Number: {part.PartNo}\n"
                f"  - Price: ₹{part.Rate:.2f}\n"
                f"  - Stock: {part.Quantity} units\n"
                f"  - Part Description: {part.PartDescription}\n"
                f"  - For Vehicle: {part.VehicleMake}\n\n"
                f"  - View Details:{part_url}\n\n"
            )
        #3. Calling and Prompt for the LLM
        system_prompt = (
            "You are a helpful assistant specialized in auto parts and vehicle maintenance. "
            "Your task is to provide accurate and helpful responses based on the provided context. "
            "Answer the user's question concisely using ONLY the information from the context below. "
            "Do not mention that you are an AI and your name is caren . Be conversational."

        )

        try:
            chat_completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
                ],
            
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "Sorry, I'm having trouble connecting to my brain right now. Please try again later."
        logger.info("LLM response generated successfully.")