# app.py

import math
import logging
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from rag import PartsRAG

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask Application ---
app = Flask(__name__)
app.secret_key = 'dev-secret-key-change-me' 

PARTS_PER_PAGE = 12

# --- RAG System Initialization ---
logging.info("Initializing RAG system...")
rag_system = PartsRAG()

DEFAULT_PART_IMAGES = {
    'engine': 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=300&fit=crop',
    'brake': 'https://images.unsplash.com/photo-1619642751034-765dfdf7c58e?w=400&h=300&fit=crop',
    'suspension': 'https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=400&h=300&fit=crop',
    'default': 'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400&h=300&fit=crop'
}

def get_part_image_url(part_category):
    category_key = next((key for key in DEFAULT_PART_IMAGES if key in part_category.lower()), 'default')
    return DEFAULT_PART_IMAGES[category_key]

# --- Database Helper Functions ---
def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('parts.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_user_id(username):
    """Get user ID from username."""
    conn = get_db_connection()
    user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return user['id'] if user else None

def get_part_by_number(part_no):
    """Get part details by part number."""
    conn = get_db_connection()
    part = conn.execute('''
        SELECT * FROM parts WHERE PartNo = ?
        ORDER BY id DESC LIMIT 1
    ''', (part_no,)).fetchone()
    conn.close()
    return dict(part) if part else None

# --- Cart Management Functions ---
def add_to_cart(user_id, part_no, quantity=1):
    """Add item to user's cart."""
    conn = get_db_connection()
    try:
        # Check if item already exists in cart
        existing = conn.execute('''
            SELECT quantity FROM cart WHERE user_id = ? AND part_no = ?
        ''', (user_id, part_no)).fetchone()
        
        if existing:
            # Update quantity
            new_quantity = existing['quantity'] + quantity
            conn.execute('''
                UPDATE cart SET quantity = ?, added_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND part_no = ?
            ''', (new_quantity, user_id, part_no))
        else:
            # Insert new item
            conn.execute('''
                INSERT INTO cart (user_id, part_no, quantity)
                VALUES (?, ?, ?)
            ''', (user_id, part_no, quantity))
        
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error adding to cart: {e}")
        return False
    finally:
        conn.close()

def get_cart_items(user_id):
    """Get all items in user's cart with part details."""
    conn = get_db_connection()
    items = conn.execute('''
        SELECT c.*, p.PartDescription, p.Rate, p.Category, p.VehicleMake, p.Quantity as StockQuantity
        FROM cart c
        JOIN parts p ON c.part_no = p.PartNo
        WHERE c.user_id = ?
        GROUP BY c.part_no
        HAVING p.id = MAX(p.id)
        ORDER BY c.added_at DESC
    ''', (user_id,)).fetchall()
    conn.close()
    return [dict(item) for item in items]

def remove_from_cart(user_id, part_no):
    """Remove item from user's cart."""
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM cart WHERE user_id = ? AND part_no = ?', (user_id, part_no))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error removing from cart: {e}")
        return False
    finally:
        conn.close()

def update_cart_quantity(user_id, part_no, quantity):
    """Update quantity of item in cart."""
    conn = get_db_connection()
    try:
        if quantity <= 0:
            conn.execute('DELETE FROM cart WHERE user_id = ? AND part_no = ?', (user_id, part_no))
        else:
            conn.execute('''
                UPDATE cart SET quantity = ?, added_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND part_no = ?
            ''', (quantity, user_id, part_no))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error updating cart quantity: {e}")
        return False
    finally:
        conn.close()

def create_order(user_id, cart_items):
    """Create order from cart items."""
    conn = get_db_connection()
    try:
        # Calculate total amount
        total_amount = sum(item['Rate'] * item['quantity'] for item in cart_items)
        
        # Create order
        cursor = conn.execute('''
            INSERT INTO orders (user_id, total_amount, status)
            VALUES (?, ?, 'completed')
        ''', (user_id, total_amount))
        order_id = cursor.lastrowid
        
        # Add order items
        for item in cart_items:
            conn.execute('''
                INSERT INTO order_items (order_id, part_no, quantity, unit_price, total_price)
                VALUES (?, ?, ?, ?, ?)
            ''', (order_id, item['part_no'], item['quantity'], item['Rate'],
                  item['Rate'] * item['quantity']))
        
        # Clear cart
        conn.execute('DELETE FROM cart WHERE user_id = ?', (user_id,))
        
        conn.commit()
        return order_id
    except Exception as e:
        logging.error(f"Error creating order: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_user_orders(user_id):
    """Get user's order history."""
    conn = get_db_connection()
    orders = conn.execute('''
        SELECT o.*, COUNT(oi.id) as item_count
        FROM orders o
        LEFT JOIN order_items oi ON o.id = oi.order_id
        WHERE o.user_id = ?
        GROUP BY o.id
        ORDER BY o.order_date DESC
    ''', (user_id,)).fetchall()
    conn.close()
    return [dict(order) for order in orders]

def get_order_details(order_id, user_id):
    """Get detailed order information."""
    conn = get_db_connection()
    order = conn.execute('''
        SELECT * FROM orders WHERE id = ? AND user_id = ?
    ''', (order_id, user_id)).fetchone()
    
    if not order:
        conn.close()
        return None
    
    items = conn.execute('''
        SELECT oi.*, p.PartDescription, p.Category, p.VehicleMake
        FROM order_items oi
        JOIN parts p ON oi.part_no = p.PartNo
        WHERE oi.order_id = ?
        GROUP BY oi.part_no
        HAVING p.id = MAX(p.id)
    ''', (order_id,)).fetchall()
    
    conn.close()
    return {
        'order': dict(order),
        'items': [dict(item) for item in items]
    }

# --- User Authentication & Role Management ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        if password == 'password':
            if role == 'employee' and username == 'employee':
                session['user_role'] = 'employee'
                return redirect(url_for('dashboard'))
            elif role == 'customer' and username == 'customer':
                session['user_role'] = 'customer'
                return redirect(url_for('dashboard'))
        
        flash('Invalid username, password, or role.', 'error')
        return redirect(url_for('login'))

    if 'user_role' in session: return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# --- Main Application Routes ---
@app.route('/')
def dashboard():
    if 'user_role' not in session: return redirect(url_for('login'))
    is_employee = (session.get('user_role') == 'employee')
    try:
        if is_employee:
            insights = rag_system.get_employee_insights()
            return render_template('dashboard.html', is_employee=True, insights=insights)
        else:
            categorized_parts = rag_system.get_categorized_parts()
            return render_template('dashboard.html', is_employee=False, categorized_parts=categorized_parts)
    except Exception as e:
        logging.error(f"Error loading dashboard: {e}")
        flash('Error loading dashboard. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route('/view_parts_by_category/<category_name>')
def view_parts_by_category(category_name):
    if 'user_role' not in session: return redirect(url_for('login'))
    try:
        page = request.args.get('page', 1, type=int)
        
        # Filter parts by mapped main category
        all_parts = []
        seen_parts = set()
        for part in rag_system.parts_data:
            if part.PartNo not in seen_parts:
                mapped_category = rag_system._map_to_main_category(part.Category, part.PartDescription)
                if mapped_category.lower() == category_name.lower():
                    all_parts.append(part)
                    seen_parts.add(part.PartNo)
        
        total_parts = len(all_parts)
        total_pages = math.ceil(total_parts / PARTS_PER_PAGE)
        page = max(1, min(page, total_pages))
        
        start_index = (page - 1) * PARTS_PER_PAGE
        parts_to_display = all_parts[start_index : start_index + PARTS_PER_PAGE]
        
        parts_with_images = []
        for part in parts_to_display:
            part_dict = part.to_dict()
            part_dict['ImageURL'] = get_part_image_url(part.Category)
            # Use cleaned description for better presentation
            part_dict['CleanedDescription'] = part.cleaned_description
            parts_with_images.append(part_dict)

        return render_template('category_parts.html', category_name=category_name.title(), parts=parts_with_images, page=page, total_pages=total_pages)
    except Exception as e:
        logging.error(f"Error loading category {category_name}: {e}")
        return redirect(url_for('dashboard'))

# --- Part Details Route ---
@app.route('/view_part/<part_no>')
def view_part(part_no):
    """View individual part details with add to cart option."""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    part = get_part_by_number(part_no)
    if not part:
        flash('Part not found.', 'error')
        return redirect(url_for('dashboard'))
    
    part['ImageURL'] = get_part_image_url(part['Category'])
    # Add cleaned description for better presentation
    part['CleanedDescription'] = rag_system.clean_part_name(part['PartDescription'])
    return render_template('part_detail.html', part=part)

# --- Cart Routes ---
@app.route('/cart')
def cart():
    """Display user's shopping cart."""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    username = 'customer' if session.get('user_role') == 'customer' else 'employee'
    user_id = get_user_id(username)
    
    if not user_id:
        flash('User not found.', 'error')
        return redirect(url_for('dashboard'))
    
    cart_items = get_cart_items(user_id)
    total_amount = sum(item['Rate'] * item['quantity'] for item in cart_items)
    
    return render_template('cart.html', cart_items=cart_items, total_amount=total_amount)

@app.route('/orders')
def orders():
    """Display user's order history."""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    username = 'customer' if session.get('user_role') == 'customer' else 'employee'
    user_id = get_user_id(username)
    
    if not user_id:
        flash('User not found.', 'error')
        return redirect(url_for('dashboard'))
    
    user_orders = get_user_orders(user_id)
    return render_template('orders.html', orders=user_orders)

@app.route('/order/<int:order_id>')
def order_detail(order_id):
    """Display detailed order information."""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    username = 'customer' if session.get('user_role') == 'customer' else 'employee'
    user_id = get_user_id(username)
    
    if not user_id:
        flash('User not found.', 'error')
        return redirect(url_for('dashboard'))
    
    order_details = get_order_details(order_id, user_id)
    if not order_details:
        flash('Order not found.', 'error')
        return redirect(url_for('orders'))
    
    return render_template('order_detail.html', **order_details)

# --- Cart API Endpoints ---
@app.route('/api/cart/add', methods=['POST'])
def api_add_to_cart():
    """Add item to cart."""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        part_no = data.get('part_no')
        quantity = int(data.get('quantity', 1))
        
        if not part_no:
            return jsonify({'error': 'Part number required'}), 400
        
        # Check if part exists
        part = get_part_by_number(part_no)
        if not part:
            return jsonify({'error': 'Part not found'}), 404
        
        username = 'customer' if session.get('user_role') == 'customer' else 'employee'
        user_id = get_user_id(username)
        
        if not user_id:
            return jsonify({'error': 'User not found'}), 404
        
        if add_to_cart(user_id, part_no, quantity):
            return jsonify({'success': True, 'message': 'Item added to cart'})
        else:
            return jsonify({'error': 'Failed to add item to cart'}), 500
            
    except Exception as e:
        logging.error(f"Error adding to cart: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/remove', methods=['POST'])
def api_remove_from_cart():
    """Remove item from cart."""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        part_no = data.get('part_no')
        
        if not part_no:
            return jsonify({'error': 'Part number required'}), 400
        
        username = 'customer' if session.get('user_role') == 'customer' else 'employee'
        user_id = get_user_id(username)
        
        if not user_id:
            return jsonify({'error': 'User not found'}), 404
        
        if remove_from_cart(user_id, part_no):
            return jsonify({'success': True, 'message': 'Item removed from cart'})
        else:
            return jsonify({'error': 'Failed to remove item from cart'}), 500
            
    except Exception as e:
        logging.error(f"Error removing from cart: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/update', methods=['POST'])
def api_update_cart():
    """Update cart item quantity."""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        part_no = data.get('part_no')
        quantity = int(data.get('quantity', 1))
        
        if not part_no:
            return jsonify({'error': 'Part number required'}), 400
        
        username = 'customer' if session.get('user_role') == 'customer' else 'employee'
        user_id = get_user_id(username)
        
        if not user_id:
            return jsonify({'error': 'User not found'}), 404
        
        if update_cart_quantity(user_id, part_no, quantity):
            return jsonify({'success': True, 'message': 'Cart updated'})
        else:
            return jsonify({'error': 'Failed to update cart'}), 500
            
    except Exception as e:
        logging.error(f"Error updating cart: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/checkout', methods=['POST'])
def api_checkout():
    """Process checkout and create order."""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        username = 'customer' if session.get('user_role') == 'customer' else 'employee'
        user_id = get_user_id(username)
        
        if not user_id:
            return jsonify({'error': 'User not found'}), 404
        
        cart_items = get_cart_items(user_id)
        if not cart_items:
            return jsonify({'error': 'Cart is empty'}), 400
        
        order_id = create_order(user_id, cart_items)
        if order_id:
            return jsonify({
                'success': True,
                'message': 'Order placed successfully',
                'order_id': order_id
            })
        else:
            return jsonify({'error': 'Failed to create order'}), 500
            
    except Exception as e:
        logging.error(f"Error during checkout: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# --- ✨ NEW ROUTE FOR THE AI REPORT PAGE ---
@app.route('/insights')
def insights():
    """Renders the main AI analysis report page, which then fetches data."""
    if session.get('user_role') != 'employee':
        return render_template('access_denied.html'), 403
    return render_template('report.html')

# --- ✨ NEW API ROUTE FOR THE AI ANALYSIS ---
@app.route('/api/get_analysis_results')
def get_analysis_results():
    """Provides the AI-generated inventory analysis data."""
    if session.get('user_role') != 'employee':
        return jsonify({'error': 'Access Denied'}), 403
    try:
        results = rag_system.generate_inventory_recommendations()
        if "error" in results: return jsonify(results), 500
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in analysis results API: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# --- API Route for the LLM Chatbot ---
@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user_role' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        user_message = request.json.get('message', '').strip()
        if not user_message: return jsonify({'reply': 'Please type a message.'})
        bot_response = rag_system.generate_llm_response(user_message)
        return jsonify({'reply': bot_response})
    except Exception as e:
        logging.error(f"Error in chat API: {e}")
        return jsonify({'reply': 'Sorry, an error occurred.'}), 500

# --- Application Startup ---
def initialize_app(app_instance):
    with app_instance.app_context():
        # Load database for customer functionality
        logging.info("Loading database into RAG system...")
        if not rag_system.load_data_from_db('parts.db'):
            logging.error("CRITICAL: Failed to load database data.")
            return False
        logging.info(f"Database loaded. Total parts: {len(rag_system.parts_data)}")
        
        # Load Excel data for employee insights
        logging.info("Loading Excel data for employee insights...")
        if not rag_system.load_data_from_excel('model-data.xlsx'):
            logging.warning("Failed to load Excel data. Employee insights will use database data.")
        else:
            logging.info(f"Excel data loaded. Total parts for insights: {len(rag_system.excel_parts_data)}")
        
        return True

if not initialize_app(app):
    exit(1)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)