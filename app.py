# app.py

import math
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from rag import PartsRAG # Make sure your updated rag.py is in the same directory

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask Application ---
app = Flask(__name__)
app.secret_key = 'dev-secret-key-change-me' 

PARTS_PER_PAGE = 12 # Define how many items to show per page

# --- RAG System Initialization ---
# This creates a single, global instance of the RAG system when the app starts.
# It's efficient because the data and search index are loaded only once.
logging.info("Initializing RAG system...")
rag_system = PartsRAG()

# Default part images mapping based on categories
DEFAULT_PART_IMAGES = {
    'engine': 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=300&fit=crop',
    'brake': 'https://images.unsplash.com/photo-1619642751034-765dfdf7c58e?w=400&h=300&fit=crop',
    'suspension': 'https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=400&h=300&fit=crop',
    'transmission': 'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400&h=300&fit=crop',
    'electrical': 'https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400&h=300&fit=crop',
    'exhaust': 'https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=400&h=300&fit=crop',
    'cooling': 'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400&h=300&fit=crop',
    'fuel': 'https://images.unsplash.com/photo-1574781330855-d0db9cc6a4c3?w=400&h=300&fit=crop',
    'body': 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=400&h=300&fit=crop',
    'interior': 'https://images.unsplash.com/photo-1555215695-3004980ad54e?w=400&h=300&fit=crop',
    'default': 'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400&h=300&fit=crop'
}

def get_part_image_url(part, category_name):
    """
    Get the image URL for a part. Since your PartRecord doesn't have ImageURL,
    we'll use category-based default images.
    """
    # Use category-based default since your database doesn't have image URLs
    category_lower = category_name.lower()
    return DEFAULT_PART_IMAGES.get(category_lower, DEFAULT_PART_IMAGES['default'])

# --- User Authentication & Role Management ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login and role assignment."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        # This is a simple authentication system. Replace with a real one for production.
        if password == 'password':
            if role == 'employee' and username == 'employee':
                session['user_role'] = 'employee'
                session['username'] = username
                flash('Welcome, Employee!', 'success')
                return redirect(url_for('dashboard'))
            elif role == 'customer' and username == 'customer':
                session['user_role'] = 'customer'
                session['username'] = username
                flash('Welcome, Customer!', 'success')
                return redirect(url_for('dashboard'))
        
        flash('Invalid username, password, or role.', 'error')
        return redirect(url_for('login'))

    if 'user_role' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logs the user out by clearing the session."""
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# --- Main Application Routes ---
@app.route('/')
def dashboard():
    """Renders the main dashboard based on the user's role."""
    if 'user_role' not in session:
        return redirect(url_for('login'))
        
    is_employee = (session.get('user_role') == 'employee')
    
    try:
        # Pass different data to the template based on the user's role
        if is_employee:
            # For employees, show the business insights dashboard
            insights = rag_system.get_employee_insights()
            return render_template('dashboard.html', is_employee=True, insights=insights)
        else:
            # For customers, show the category browser and chatbot
            # The chatbot will fetch data dynamically via the /api/chat endpoint
            categorized_parts = rag_system.get_categorized_parts()
            return render_template('dashboard.html', is_employee=False, categorized_parts=categorized_parts)
    except Exception as e:
        logging.error(f"Error loading dashboard: {e}")
        flash('Error loading dashboard. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route('/view_parts_by_category/<category_name>')
def view_parts_by_category(category_name):
    """Display parts for a specific category with pagination."""
    if 'user_role' not in session:
        return redirect(url_for('login'))

    try:
        page = request.args.get('page', 1, type=int)
        
        # Get all parts in the specified category (case-insensitive)
        all_parts_in_category = [
            p for p in rag_system.parts_data 
            if p.Category and p.Category.lower() == category_name.lower()
        ]
        
        if not all_parts_in_category:
            flash(f'No parts found in category "{category_name}".', 'error')
            return redirect(url_for('dashboard'))
        
        total_parts = len(all_parts_in_category)
        total_pages = math.ceil(total_parts / PARTS_PER_PAGE)
        
        # Ensure page number is valid
        if page < 1:
            page = 1
        elif page > total_pages:
            page = total_pages
        
        start_index = (page - 1) * PARTS_PER_PAGE
        end_index = start_index + PARTS_PER_PAGE
        
        parts_to_display = all_parts_in_category[start_index:end_index]
        
        # Create a list of parts with image URLs (since PartRecord is frozen)
        parts_with_images = []
        for part in parts_to_display:
            # Create a dictionary with part data + image URL
            part_dict = part.to_dict()
            part_dict['ImageURL'] = get_part_image_url(part, category_name)
            parts_with_images.append(part_dict)

        return render_template(
            'category_parts.html', 
            category_name=category_name.title(), 
            parts=parts_with_images,
            page=page,
            total_pages=total_pages,
            total_parts=total_parts
        )
        
    except Exception as e:
        logging.error(f"Error loading category {category_name}: {e}")
        flash('Error loading parts. Please try again.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/categories')
def categories():
    """Display all available categories."""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    try:
        categorized_parts = rag_system.get_categorized_parts()
        return render_template('categories.html', categorized_parts=categorized_parts)
    except Exception as e:
        logging.error(f"Error loading categories: {e}")
        flash('Error loading categories. Please try again.', 'error')
        return redirect(url_for('dashboard'))

# --- ✨ API Route for the LLM Chatbot ---
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Handles conversational queries from the user by using the RAG + LLM system.
    """
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized access.'}), 401
    
    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({'reply': 'Please type a message to get started.'})

        # This single call performs the entire RAG process:
        # 1. Searches for relevant parts (Retrieve)
        # 2. Creates a context for the LLM (Augment)
        # 3. Gets a conversational response from the LLM (Generate)
        bot_response = rag_system.generate_llm_response(user_message)
        
        return jsonify({'reply': bot_response})
        
    except Exception as e:
        logging.error(f"Error in chat API: {e}")
        return jsonify({'reply': 'Sorry, I encountered an error. Please try again.'}), 500

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    return "Page not found", 404

@app.errorhandler(500)
def internal_error(error):
    return "Internal server error", 500

# --- Application Startup ---
def initialize_app(app_instance):
    """Load data into the RAG system within the app context."""
    with app_instance.app_context():
        logging.info("Loading database into RAG system...")
        try:
            # Make sure your database file is named 'parts.db' or update the path here
            if not rag_system.load_data_from_db('parts.db'):
                logging.error("CRITICAL: Failed to load data. The application might not work as expected.")
                return False
            else:
                logging.info(f"Database loaded successfully. Total parts: {len(rag_system.parts_data)}")
                return True
        except Exception as e:
            logging.error(f"CRITICAL: Exception while loading data: {e}")
            return False

# Initialize the application
if not initialize_app(app):
    logging.error("Failed to initialize application. Exiting.")
    exit(1)

if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000)