"""
Web Frameworks: Flask Basics and Secure Web Development
"""

# ===== INTRODUCTION TO FLASK =====
print("\n===== INTRODUCTION TO FLASK =====")
"""
Flask is a lightweight WSGI web application framework designed to make getting started quick
and easy, with the ability to scale up to complex applications.

Key Features:
1. Lightweight and modular design
2. Built-in development server and debugger
3. RESTful request dispatching
4. Jinja2 templating
5. Support for secure cookies
6. Extensive documentation
7. Google App Engine compatibility
8. Extensions available for form validation, upload handling, authentication, etc.

Installation:
```bash
pip install flask
```

Basic Flask application:
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

Running a Flask application:
```bash
# Method 1: Using Python
python app.py

# Method 2: Using Flask CLI
export FLASK_APP=app.py
export FLASK_ENV=development  # For development mode
flask run
```
"""

# ===== ROUTING =====
print("\n===== ROUTING =====")
"""
Flask uses decorators to bind functions to URLs:

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

# Variable rules
@app.route('/user/<username>')
def show_user_profile(username):
    return f'User {username}'

# Converter types
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post {post_id}'

# Multiple rules
@app.route('/projects/')
@app.route('/projects/<path:subpath>')
def projects(subpath=None):
    if subpath:
        return f'Subpath: {subpath}'
    return 'Projects index'
```

URL converters:
- string: (default) accepts any text without a slash
- int: accepts positive integers
- float: accepts positive floating point values
- path: like string but also accepts slashes
- uuid: accepts UUID strings
"""

# ===== REQUEST HANDLING =====
print("\n===== REQUEST HANDLING =====")
"""
Flask provides access to request data through the request object:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Process login
        return f'Logged in as {username}'
    else:
        return 'Please log in'

@app.route('/api/data', methods=['POST'])
def receive_data():
    # Access JSON data
    data = request.get_json()
    return jsonify({'received': data})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save the file
        file.save('/path/to/uploads/' + file.filename)
        return 'File uploaded successfully'
```

Request data access:
- request.form: form data (POST or PUT)
- request.args: query string parameters
- request.files: uploaded files
- request.json: parsed JSON data (with Content-Type: application/json)
- request.cookies: cookies
- request.headers: headers
"""

# ===== RESPONSES =====
print("\n===== RESPONSES =====")
"""
Flask provides multiple ways to create responses:

```python
from flask import Flask, jsonify, render_template, redirect, url_for, make_response

app = Flask(__name__)

# Simple string response
@app.route('/')
def index():
    return 'Hello, World!'

# JSON response
@app.route('/api/data')
def get_data():
    data = {'name': 'John', 'age': 30}
    return jsonify(data)

# HTML template response
@app.route('/user/<username>')
def user_profile(username):
    return render_template('profile.html', username=username)

# Redirect
@app.route('/redirect')
def redirect_example():
    return redirect(url_for('index'))

# Custom response
@app.route('/cookie')
def set_cookie():
    resp = make_response('Cookie set!')
    resp.set_cookie('username', 'flask-user')
    return resp

# Custom status code
@app.route('/error')
def error():
    return 'Error occurred', 500
```

Response types:
- Strings: converted to HTML response
- Dictionaries/lists: converted to JSON response
- Tuples: (response, status_code) or (response, status_code, headers)
- Response objects: for more control
"""

# ===== TEMPLATES =====
print("\n===== TEMPLATES =====")
"""
Flask uses Jinja2 for templating:

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)
```

Example template (hello.html):
```html
<!doctype html>
<html>
  <head>
    <title>Hello from Flask</title>
  </head>
  <body>
    {% if name %}
      <h1>Hello {{ name }}!</h1>
    {% else %}
      <h1>Hello, World!</h1>
    {% endif %}
    
    <ul>
      {% for item in items %}
        <li>{{ item }}</li>
      {% endfor %}
    </ul>
  </body>
</html>
```

Template features:
- Variable substitution: {{ variable }}
- Control structures: {% if condition %} ... {% endif %}
- Loops: {% for item in items %} ... {% endfor %}
- Template inheritance: {% extends "base.html" %}
- Blocks: {% block content %} ... {% endblock %}
- Filters: {{ name|capitalize }}
- Macros: {% macro input(name, value='', type='text') %} ... {% endmacro %}
"""

# ===== STATIC FILES =====
print("\n===== STATIC FILES =====")
"""
Flask serves static files from the 'static' folder:

Project structure:
```
/app
  /static
    /css
      style.css
    /js
      script.js
    /images
      logo.png
  /templates
    index.html
  app.py
```

Referencing static files in templates:
```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
<script src="{{ url_for('static', filename='js/script.js') }}"></script>
<img src="{{ url_for('static', filename='images/logo.png') }}">
```
"""

# ===== SESSIONS AND COOKIES =====
print("\n===== SESSIONS AND COOKIES =====")
"""
Flask provides support for cookies and sessions:

```python
from flask import Flask, session, request, make_response

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for sessions

# Working with sessions
@app.route('/set_session')
def set_session():
    session['username'] = 'flask-user'
    return 'Session variable set!'

@app.route('/get_session')
def get_session():
    username = session.get('username', 'Guest')
    return f'Hello, {username}!'

# Working with cookies
@app.route('/set_cookie')
def set_cookie():
    resp = make_response('Cookie set!')
    resp.set_cookie('user', 'flask-cookie-user', max_age=60*60*24*30)  # 30 days
    return resp

@app.route('/get_cookie')
def get_cookie():
    user = request.cookies.get('user', 'Guest')
    return f'Cookie value: {user}'
```

Session security:
- Always set a strong secret key
- Consider using a server-side session store for sensitive data
- Set secure and httponly flags for cookies in production
"""

# ===== DATABASE INTEGRATION =====
print("\n===== DATABASE INTEGRATION =====")
"""
Flask can work with any database. Here's an example with SQLAlchemy:

```python
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    
    def __repr__(self):
        return f'<User {self.username}>'

# Create tables
with app.app_context():
    db.create_all()

@app.route('/users')
def list_users():
    users = User.query.all()
    return render_template('users.html', users=users)

@app.route('/users/add', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        user = User(username=username, email=email)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('list_users'))
    return render_template('add_user.html')
```

Database best practices:
- Use an ORM like SQLAlchemy for security and convenience
- Use migrations for schema changes (Flask-Migrate)
- Parameterize queries to prevent SQL injection
- Use connection pooling for better performance
"""

# ===== FORMS =====
print("\n===== FORMS =====")
"""
Flask-WTF provides form handling and validation:

```python
from flask import Flask, render_template, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Log In')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Process the form data
        email = form.email.data
        password = form.password.data
        # Authenticate user
        flash('Login successful!')
        return redirect(url_for('index'))
    return render_template('login.html', form=form)
```

Form template (login.html):
```html
{% extends "base.html" %}

{% block content %}
  <h1>Login</h1>
  <form method="POST">
    {{ form.hidden_tag() }}
    <div>
      {{ form.email.label }}
      {{ form.email }}
      {% if form.email.errors %}
        <ul class="errors">
          {% for error in form.email.errors %}
            <li>{{ error }}</li>
          {% endfor %}
        </ul>
      {% endif %}
    </div>
    <div>
      {{ form.password.label }}
      {{ form.password }}
      {% if form.password.errors %}
        <ul class="errors">
          {% for error in form.password.errors %}
            <li>{{ error }}</li>
          {% endfor %}
        </ul>
      {% endif %}
    </div>
    {{ form.submit }}
  </form>
{% endblock %}
```

Form features:
- CSRF protection
- Field validation
- Custom validators
- File uploads
- Form rendering
"""

# ===== BLUEPRINTS =====
print("\n===== BLUEPRINTS =====")
"""
Blueprints help organize Flask applications:

```python
# auth.py
from flask import Blueprint, render_template, redirect, url_for

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login')
def login():
    return render_template('auth/login.html')

@auth_bp.route('/register')
def register():
    return render_template('auth/register.html')

# api.py
from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/users')
def get_users():
    users = [{'id': 1, 'name': 'John'}, {'id': 2, 'name': 'Jane'}]
    return jsonify(users)

# app.py
from flask import Flask
from auth import auth_bp
from api import api_bp

app = Flask(__name__)
app.register_blueprint(auth_bp)
app.register_blueprint(api_bp)

@app.route('/')
def index():
    return 'Home Page'
```

Blueprint benefits:
- Modular application structure
- Reusable components
- Separate URL prefixes
- Separate static files and templates
- Easier maintenance for large applications
"""

# ===== AUTHENTICATION =====
print("\n===== AUTHENTICATION =====")
"""
Flask-Login provides user authentication:

```python
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')
```

Authentication best practices:
- Never store plain text passwords
- Use strong password hashing (bcrypt, Argon2)
- Implement proper session management
- Use HTTPS in production
- Consider two-factor authentication for sensitive applications
"""

# ===== ERROR HANDLING =====
print("\n===== ERROR HANDLING =====")
"""
Flask provides ways to handle errors:

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Custom error pages
@app.route('/error')
def trigger_error():
    # Simulate an error
    1 / 0
```

Error handling best practices:
- Create custom error pages for common HTTP errors
- Log errors properly
- Don't expose sensitive information in error messages
- Return appropriate status codes
"""

# ===== SECURITY BEST PRACTICES =====
print("\n===== SECURITY BEST PRACTICES =====")
"""
Securing Flask applications:

1. Cross-Site Scripting (XSS) Protection
   - Jinja2 templates escape content by default
   - Use safe filter only when necessary: {{ content|safe }}
   - Set Content-Security-Policy headers

2. Cross-Site Request Forgery (CSRF) Protection
   - Use Flask-WTF for CSRF protection
   - Include csrf_token in all forms: {{ form.csrf_token }}

3. SQL Injection Prevention
   - Use SQLAlchemy or other ORMs
   - Use parameterized queries
   - Never build SQL queries with string concatenation

4. Secure Headers
   - Use Flask-Talisman to set security headers
   - Set appropriate CORS policies
   - Enable HSTS in production

5. Authentication and Authorization
   - Use Flask-Login for session management
   - Implement proper password hashing
   - Use role-based access control

6. Session Security
   - Set a strong SECRET_KEY
   - Use secure and httponly flags for cookies
   - Consider server-side sessions for sensitive data

7. Input Validation
   - Validate all user inputs
   - Use WTForms validators
   - Sanitize data before storing or displaying

8. Dependency Management
   - Keep dependencies updated
   - Use tools like safety to check for vulnerabilities

9. Deployment
   - Always use HTTPS in production
   - Set DEBUG=False in production
   - Use a production WSGI server (Gunicorn, uWSGI)
   - Set appropriate file permissions
"""

# ===== DEPLOYMENT =====
print("\n===== DEPLOYMENT =====")
"""
Deploying Flask applications:

1. Production WSGI Server
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

2. Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

3. Deployment platforms:
   - Heroku
   - AWS (Elastic Beanstalk, EC2)
   - Google Cloud Run
   - Azure App Service
   - Digital Ocean App Platform

4. Nginx as a reverse proxy:
```nginx
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Deployment best practices:
- Use environment variables for configuration
- Set up proper logging
- Implement health checks
- Use a process manager (e.g., Supervisor)
- Set up monitoring and alerting
"""

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Flask is a versatile and lightweight web framework that allows you to build anything from simple websites to complex web applications and APIs.

Key advantages of Flask:
1. Simplicity: Easy to learn and use
2. Flexibility: No enforced structure or dependencies
3. Extensibility: Many extensions available for common tasks
4. Minimalism: Only includes what you need
5. Control: Fine-grained control over components

To build secure and efficient web applications with Flask:
1. Use extensions for common functionality
2. Organize code with blueprints for larger applications
3. Follow security best practices
4. Use proper error handling
5. Test your application thoroughly
6. Deploy with a production WSGI server

Flask is an excellent choice when you need flexibility and control over your web application architecture.
""")

print("\n===== END OF FLASK BASICS =====")