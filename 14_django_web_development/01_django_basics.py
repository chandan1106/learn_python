"""
Django Web Development: Getting Started with Django
"""

# This file provides an introduction to Django web framework
# In a real environment, you would need to install Django: pip install django

# ===== INTRODUCTION TO DJANGO =====
print("\n===== INTRODUCTION TO DJANGO =====")
"""
Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design.

Key Features:
1. Batteries-included: Comes with many built-in features
2. ORM (Object-Relational Mapping): Work with databases using Python objects
3. Admin interface: Automatic admin interface for content management
4. URL routing: Flexible URL configuration
5. Template system: Powerful template language
6. Form handling: Built-in form validation and processing
7. Authentication: User authentication and authorization
8. Security: Protection against common web vulnerabilities
9. Scalability: Can scale from small projects to large applications

Django follows the MVT (Model-View-Template) architecture:
- Model: Defines data structure and database interactions
- View: Contains the logic that processes requests and returns responses
- Template: Defines how the data is presented to the user
"""

# ===== SETTING UP A DJANGO PROJECT =====
print("\n===== SETTING UP A DJANGO PROJECT =====")
"""
To create a new Django project, you would use the following commands:

```bash
# Install Django
pip install django

# Create a new project
django-admin startproject myproject

# Navigate to the project directory
cd myproject

# Create a new app within the project
python manage.py startapp myapp

# Run the development server
python manage.py runserver
```

Project Structure:
```
myproject/
├── manage.py              # Command-line utility for administrative tasks
├── myproject/             # Project package
│   ├── __init__.py        # Empty file that makes the directory a Python package
│   ├── settings.py        # Project settings/configuration
│   ├── urls.py            # URL declarations for the project
│   ├── asgi.py            # Entry point for ASGI-compatible web servers
│   └── wsgi.py            # Entry point for WSGI-compatible web servers
└── myapp/                 # Application package
    ├── __init__.py        # Empty file that makes the directory a Python package
    ├── admin.py           # Register models with the Django admin
    ├── apps.py            # Application configuration
    ├── migrations/        # Database migrations
    ├── models.py          # Data models
    ├── tests.py           # Test cases
    └── views.py           # Views/controllers
```
"""

# ===== MODELS AND DATABASE CONFIGURATION =====
print("\n===== MODELS AND DATABASE CONFIGURATION =====")
"""
Django's ORM allows you to define your data models as Python classes.

Example model definition:
```python
# myapp/models.py
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)
    bio = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='books')
    publication_date = models.DateField()
    price = models.DecimalField(max_digits=6, decimal_places=2)
    is_published = models.BooleanField(default=True)
    
    def __str__(self):
        return self.title
```

Database Configuration:
Django supports multiple database backends including PostgreSQL, MySQL, SQLite, and Oracle.

Configure your database in settings.py:
```python
# myproject/settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydatabase',
        'USER': 'mydatabaseuser',
        'PASSWORD': 'mypassword',
        'HOST': '127.0.0.1',
        'PORT': '5432',
    }
}
```

After defining models, create and apply migrations:
```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate
```
"""

# ===== VIEWS AND URL PATTERNS =====
print("\n===== VIEWS AND URL PATTERNS =====")
"""
Views handle the logic of processing requests and returning responses.

Function-based views:
```python
# myapp/views.py
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import Book

def index(request):
    return HttpResponse("Hello, world!")

def book_list(request):
    books = Book.objects.all()
    return render(request, 'myapp/book_list.html', {'books': books})

def book_detail(request, book_id):
    book = get_object_or_404(Book, pk=book_id)
    return render(request, 'myapp/book_detail.html', {'book': book})
```

Class-based views:
```python
# myapp/views.py
from django.views.generic import ListView, DetailView
from .models import Book

class BookListView(ListView):
    model = Book
    template_name = 'myapp/book_list.html'
    context_object_name = 'books'

class BookDetailView(DetailView):
    model = Book
    template_name = 'myapp/book_detail.html'
    context_object_name = 'book'
```

URL patterns connect URLs to views:
```python
# myapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('books/', views.book_list, name='book_list'),
    path('books/<int:book_id>/', views.book_detail, name='book_detail'),
    
    # For class-based views
    path('books/', views.BookListView.as_view(), name='book_list'),
    path('books/<int:pk>/', views.BookDetailView.as_view(), name='book_detail'),
]
```

Include app URLs in the project URLs:
```python
# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('myapp/', include('myapp.urls')),
]
```
"""

# ===== TEMPLATES =====
print("\n===== TEMPLATES =====")
"""
Templates define how data is presented to the user.

Template structure:
```
myapp/
└── templates/
    └── myapp/
        ├── base.html
        ├── book_list.html
        └── book_detail.html
```

Base template:
```html
<!-- myapp/templates/myapp/base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}My Django App{% endblock %}</title>
</head>
<body>
    <header>
        <h1>My Book Collection</h1>
        <nav>
            <a href="{% url 'index' %}">Home</a>
            <a href="{% url 'book_list' %}">Books</a>
        </nav>
    </header>
    
    <main>
        {% block content %}
        {% endblock %}
    </main>
    
    <footer>
        <p>&copy; 2023 My Django App</p>
    </footer>
</body>
</html>
```

List template:
```html
<!-- myapp/templates/myapp/book_list.html -->
{% extends 'myapp/base.html' %}

{% block title %}Book List{% endblock %}

{% block content %}
    <h2>All Books</h2>
    
    {% if books %}
        <ul>
            {% for book in books %}
                <li>
                    <a href="{% url 'book_detail' book.id %}">{{ book.title }}</a>
                    by {{ book.author.name }}
                </li>
            {% endfor %}
        </ul>
    {% else %}
        <p>No books available.</p>
    {% endif %}
{% endblock %}
```

Detail template:
```html
<!-- myapp/templates/myapp/book_detail.html -->
{% extends 'myapp/base.html' %}

{% block title %}{{ book.title }}{% endblock %}

{% block content %}
    <h2>{{ book.title }}</h2>
    <p>Author: {{ book.author.name }}</p>
    <p>Publication Date: {{ book.publication_date }}</p>
    <p>Price: ${{ book.price }}</p>
    
    <a href="{% url 'book_list' %}">Back to Book List</a>
{% endblock %}
```
"""

# ===== FORMS =====
print("\n===== FORMS =====")
"""
Django provides a powerful form library for handling user input.

Form definition:
```python
# myapp/forms.py
from django import forms
from .models import Book, Author

class AuthorForm(forms.ModelForm):
    class Meta:
        model = Author
        fields = ['name', 'bio']

class BookForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ['title', 'author', 'publication_date', 'price', 'is_published']
        widgets = {
            'publication_date': forms.DateInput(attrs={'type': 'date'}),
        }
```

Using forms in views:
```python
# myapp/views.py
from django.shortcuts import render, redirect
from .forms import BookForm
from .models import Book

def create_book(request):
    if request.method == 'POST':
        form = BookForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = BookForm()
    
    return render(request, 'myapp/book_form.html', {'form': form})

def update_book(request, book_id):
    book = get_object_or_404(Book, pk=book_id)
    
    if request.method == 'POST':
        form = BookForm(request.POST, instance=book)
        if form.is_valid():
            form.save()
            return redirect('book_detail', book_id=book.id)
    else:
        form = BookForm(instance=book)
    
    return render(request, 'myapp/book_form.html', {'form': form})
```

Form template:
```html
<!-- myapp/templates/myapp/book_form.html -->
{% extends 'myapp/base.html' %}

{% block title %}{% if form.instance.id %}Edit Book{% else %}New Book{% endif %}{% endblock %}

{% block content %}
    <h2>{% if form.instance.id %}Edit Book{% else %}New Book{% endif %}</h2>
    
    <form method="post">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit">Save</button>
    </form>
    
    <a href="{% url 'book_list' %}">Cancel</a>
{% endblock %}
```
"""

# ===== ADMIN INTERFACE =====
print("\n===== ADMIN INTERFACE =====")
"""
Django comes with a built-in admin interface for managing your data.

Register models with the admin:
```python
# myapp/admin.py
from django.contrib import admin
from .models import Author, Book

@admin.register(Author)
class AuthorAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'publication_date', 'price', 'is_published')
    list_filter = ('is_published', 'publication_date')
    search_fields = ('title', 'author__name')
    date_hierarchy = 'publication_date'
```

Create a superuser to access the admin:
```bash
python manage.py createsuperuser
```

Access the admin interface at http://localhost:8000/admin/
"""

print("\n===== END OF DJANGO BASICS PART 1 =====")