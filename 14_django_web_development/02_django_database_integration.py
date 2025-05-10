"""
Django Web Development: Database Integration with Django
"""

# This file covers connecting Django to SQL databases and performing database operations
# In a real environment, you would need to install Django and database drivers

# ===== DJANGO DATABASE INTEGRATION =====
print("\n===== DJANGO DATABASE INTEGRATION =====")
"""
Django's ORM (Object-Relational Mapper) provides a powerful abstraction layer for working with databases.
It allows you to interact with your database using Python objects instead of writing raw SQL.

Key benefits:
1. Database independence - switch between database backends with minimal code changes
2. Automatic SQL generation - Django generates optimized SQL for you
3. Migration system - track and apply database schema changes
4. Query API - powerful and intuitive API for querying data
5. Transaction support - manage database transactions easily
"""

# ===== SUPPORTED DATABASES =====
print("\n===== SUPPORTED DATABASES =====")
"""
Django officially supports several database backends:

1. PostgreSQL (recommended for production)
   - Full feature support
   - Advanced data types (JSON, arrays, etc.)
   - Installation: pip install psycopg2-binary

2. MySQL / MariaDB
   - Widely used
   - Installation: pip install mysqlclient

3. SQLite (default)
   - No setup required
   - Good for development and small applications
   - Built into Python

4. Oracle
   - Enterprise database
   - Installation: pip install cx_Oracle

Third-party backends are available for:
- Microsoft SQL Server
- IBM DB2
- SAP SQL Anywhere
- Firebird
- ODBC connections
"""

# ===== DATABASE CONFIGURATION =====
print("\n===== DATABASE CONFIGURATION =====")
"""
Database configuration is defined in the DATABASES setting in your project's settings.py file.

Example configurations:

SQLite (default):
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',  # Path to SQLite file
    }
}
```

PostgreSQL:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydatabase',             # Database name
        'USER': 'mydatabaseuser',         # Database user
        'PASSWORD': 'mypassword',         # Database password
        'HOST': 'localhost',              # Database host (or IP address)
        'PORT': '5432',                   # Database port
        'CONN_MAX_AGE': 600,              # Connection lifetime in seconds
        'OPTIONS': {
            'sslmode': 'require',         # Use SSL
        },
    }
}
```

MySQL:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'mydatabase',
        'USER': 'mydatabaseuser',
        'PASSWORD': 'mypassword',
        'HOST': 'localhost',
        'PORT': '3306',
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
            'charset': 'utf8mb4',
        },
    }
}
```

Multiple databases:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'main_db',
        # ... other settings
    },
    'analytics': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'analytics_db',
        # ... other settings
    },
    'archive': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'archive_db',
        # ... other settings
    }
}
```

Best practice: Use environment variables for sensitive information:
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}
```
"""

# ===== DEFINING MODELS =====
print("\n===== DEFINING MODELS =====")
"""
Models are Python classes that define the structure of your database tables.

Basic model example:
```python
from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
```

Common field types:
- CharField: Short text
- TextField: Long text
- IntegerField: Integer
- DecimalField: Decimal numbers
- BooleanField: True/False
- DateField: Date
- DateTimeField: Date and time
- EmailField: Email address
- FileField: File upload
- ImageField: Image upload
- ForeignKey: One-to-many relationship
- ManyToManyField: Many-to-many relationship
- OneToOneField: One-to-one relationship

Field options:
- null: Allow NULL values in the database
- blank: Allow empty values in forms
- choices: Limit values to predefined choices
- default: Default value
- help_text: Help text for forms
- unique: Enforce uniqueness
- verbose_name: Human-readable name
- validators: Custom validation functions

Example with relationships:
```python
class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    
    class Meta:
        verbose_name_plural = 'Categories'
    
    def __str__(self):
        return self.name

class Product(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    image = models.ImageField(upload_to='products/', blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    tags = models.ManyToManyField('Tag', blank=True)
    
    def __str__(self):
        return self.name

class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    
    def __str__(self):
        return self.name

class ProductDetail(models.Model):
    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name='detail')
    weight = models.FloatField(null=True, blank=True)
    dimensions = models.CharField(max_length=100, blank=True)
    material = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"{self.product.name} details"
```

Model inheritance:
```python
# Abstract base class
class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True  # This model won't be created in the database

# Concrete models inheriting from BaseModel
class Product(BaseModel):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return self.name

class Customer(BaseModel):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    
    def __str__(self):
        return self.name
```

Meta options:
```python
class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    class Meta:
        verbose_name = 'Product'
        verbose_name_plural = 'Products'
        ordering = ['-price', 'name']  # Order by price (desc) then name (asc)
        unique_together = ['name', 'price']  # Enforce uniqueness on combination
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['price']),
        ]
        db_table = 'store_product'  # Custom table name
```
"""

# ===== MIGRATIONS =====
print("\n===== MIGRATIONS =====")
"""
Migrations are Django's way of propagating changes to your models into your database schema.

Key migration commands:
```bash
# Create migrations based on model changes
python manage.py makemigrations

# Apply pending migrations
python manage.py migrate

# Show migration status
python manage.py showmigrations

# Generate SQL for a migration without applying it
python manage.py sqlmigrate app_name migration_name

# Create an empty migration
python manage.py makemigrations app_name --empty --name=migration_name
```

Migration workflow:
1. Make changes to your models
2. Run `makemigrations` to create migration files
3. Review the generated migration files
4. Run `migrate` to apply the migrations to your database

Custom migrations:
```python
# migrations/0002_custom_migration.py
from django.db import migrations, models

def set_default_prices(apps, schema_editor):
    # Get the historical model
    Product = apps.get_model('myapp', 'Product')
    # Update all products with zero price
    Product.objects.filter(price=0).update(price=9.99)

def reverse_default_prices(apps, schema_editor):
    # This is the reverse operation (for migrations rollback)
    pass

class Migration(migrations.Migration):
    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(set_default_prices, reverse_default_prices),
    ]
```

Migration best practices:
1. Always review migrations before applying them
2. Keep migrations small and focused
3. Test migrations on a copy of production data before deploying
4. Never edit or delete migration files after they've been applied
5. Use `--name` to give migrations descriptive names
6. Include data migrations when necessary
"""

# ===== QUERYING THE DATABASE =====
print("\n===== QUERYING THE DATABASE =====")
"""
Django's ORM provides a powerful API for querying the database.

Basic queries:
```python
# Get all products
products = Product.objects.all()

# Get a single product by primary key
product = Product.objects.get(pk=1)

# Get the first product
first_product = Product.objects.first()

# Get the last product
last_product = Product.objects.last()

# Count products
product_count = Product.objects.count()
```

Filtering queries:
```python
# Filter by exact match
active_products = Product.objects.filter(is_active=True)

# Filter by multiple conditions (AND)
featured_active = Product.objects.filter(is_active=True, is_featured=True)

# Price greater than
expensive = Product.objects.filter(price__gt=100)

# Price less than or equal to
affordable = Product.objects.filter(price__lte=50)

# Price between
mid_range = Product.objects.filter(price__range=(50, 100))

# String contains (case-sensitive)
contains_phone = Product.objects.filter(name__contains='phone')

# String contains (case-insensitive)
contains_phone_i = Product.objects.filter(name__icontains='phone')

# Starts with
starts_with = Product.objects.filter(name__startswith='Apple')

# Ends with
ends_with = Product.objects.filter(name__endswith='Pro')

# In a list
in_categories = Product.objects.filter(category__in=[1, 2, 3])

# Date greater than
recent = Product.objects.filter(created_at__gt='2023-01-01')

# Exclude
not_phones = Product.objects.exclude(category__name='Phones')
```

Complex queries with Q objects:
```python
from django.db.models import Q

# OR condition
phones_or_tablets = Product.objects.filter(
    Q(category__name='Phones') | Q(category__name='Tablets')
)

# AND and OR combined
complex_query = Product.objects.filter(
    Q(price__lt=100) | Q(is_on_sale=True),
    is_active=True
)

# NOT condition
not_phones = Product.objects.filter(~Q(category__name='Phones'))
```

Queries across relationships:
```python
# Forward relationship (ForeignKey)
category_products = Product.objects.filter(category__name='Electronics')

# Backward relationship
electronics = Category.objects.get(name='Electronics')
category_products = electronics.products.all()  # Using related_name

# Many-to-many
products_with_tag = Product.objects.filter(tags__name='Wireless')
tags_for_product = product.tags.all()

# Nested relationships
products_by_country = Product.objects.filter(category__parent__country__name='USA')
```

Ordering results:
```python
# Ascending order
by_name = Product.objects.order_by('name')

# Descending order
by_price_desc = Product.objects.order_by('-price')

# Multiple fields
ordered = Product.objects.order_by('category__name', '-price')

# Random order
random = Product.objects.order_by('?')
```

Limiting results:
```python
# First 5 products
top_5 = Product.objects.all()[:5]

# Products 6-10
next_5 = Product.objects.all()[5:10]

# All except first 5
rest = Product.objects.all()[5:]
```

Field lookups:
```python
# Exact match (default)
Product.objects.filter(name='iPhone')

# Case-insensitive exact match
Product.objects.filter(name__iexact='iphone')

# Contains
Product.objects.filter(name__contains='phone')

# Case-insensitive contains
Product.objects.filter(name__icontains='phone')

# In list
Product.objects.filter(id__in=[1, 2, 3])

# Greater than
Product.objects.filter(price__gt=100)

# Greater than or equal to
Product.objects.filter(price__gte=100)

# Less than
Product.objects.filter(price__lt=100)

# Less than or equal to
Product.objects.filter(price__lte=100)

# Range
Product.objects.filter(price__range=(50, 100))

# Date range
Product.objects.filter(created_at__date__range=('2023-01-01', '2023-12-31'))

# Starts with
Product.objects.filter(name__startswith='i')

# Ends with
Product.objects.filter(name__endswith='Phone')

# Regex
Product.objects.filter(name__regex=r'^i.*e$')

# Is null
Product.objects.filter(description__isnull=True)
```
"""

# ===== ADVANCED QUERIES =====
print("\n===== ADVANCED QUERIES =====")
"""
Django's ORM supports advanced query operations like annotations, aggregations, and more.

Annotations:
```python
from django.db.models import Count, Sum, Avg, Min, Max, F, Value, CharField
from django.db.models.functions import Concat

# Count related objects
categories_with_counts = Category.objects.annotate(
    product_count=Count('products')
)
for category in categories_with_counts:
    print(f"{category.name}: {category.product_count} products")

# Calculate total value
products_with_total = Product.objects.annotate(
    total_value=F('price') * F('stock')
)
for product in products_with_total:
    print(f"{product.name}: ${product.total_value}")

# Combine fields
products_with_full_name = Product.objects.annotate(
    full_name=Concat('category__name', Value(' - '), 'name', output_field=CharField())
)
for product in products_with_full_name:
    print(product.full_name)  # "Electronics - iPhone"
```

Aggregations:
```python
from django.db.models import Count, Sum, Avg, Min, Max

# Count all products
product_count = Product.objects.count()

# Sum of all product prices
total_price = Product.objects.aggregate(Sum('price'))
print(f"Total price: ${total_price['price__sum']}")

# Average price
avg_price = Product.objects.aggregate(Avg('price'))
print(f"Average price: ${avg_price['price__avg']}")

# Min and max price
price_range = Product.objects.aggregate(
    min_price=Min('price'),
    max_price=Max('price')
)
print(f"Price range: ${price_range['min_price']} - ${price_range['max_price']}")

# Multiple aggregations by category
category_stats = Category.objects.annotate(
    product_count=Count('products'),
    avg_price=Avg('products__price'),
    total_stock=Sum('products__stock')
)
for category in category_stats:
    print(f"{category.name}: {category.product_count} products, "
          f"avg price: ${category.avg_price}, "
          f"total stock: {category.total_stock}")
```

F expressions:
```python
from django.db.models import F

# Update prices by 10%
Product.objects.update(price=F('price') * 1.1)

# Increase stock
Product.objects.filter(name='iPhone').update(stock=F('stock') + 10)

# Compare fields
low_stock = Product.objects.filter(stock__lt=F('min_stock'))

# Arithmetic
Product.objects.annotate(margin=F('price') - F('cost'))

# Date arithmetic
from datetime import timedelta
from django.utils import timezone
recent = Product.objects.filter(created_at__gte=timezone.now() - timedelta(days=30))
```

Subqueries:
```python
from django.db.models import Subquery, OuterRef

# Get products in categories with more than 5 products
product_count_subquery = (
    Product.objects.filter(category=OuterRef('pk'))
    .values('category')
    .annotate(count=Count('id'))
    .values('count')
)
categories = Category.objects.annotate(
    product_count=Subquery(product_count_subquery)
).filter(product_count__gt=5)

# Get latest order for each customer
latest_order_subquery = (
    Order.objects.filter(customer=OuterRef('pk'))
    .order_by('-created_at')
    .values('created_at')[:1]
)
customers = Customer.objects.annotate(
    latest_order_date=Subquery(latest_order_subquery)
)
```

Raw SQL:
```python
# Raw SQL query
products = Product.objects.raw('SELECT * FROM myapp_product WHERE price > %s', [100])
for product in products:
    print(product.name, product.price)

# Custom SQL with connection
from django.db import connection

def get_expensive_products():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT p.name, p.price, c.name as category
            FROM myapp_product p
            JOIN myapp_category c ON p.category_id = c.id
            WHERE p.price > %s
            ORDER BY p.price DESC
        """, [100])
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

expensive_products = get_expensive_products()
for product in expensive_products:
    print(f"{product['name']} (${product['price']}) - {product['category']}")
```
"""

# ===== DATABASE TRANSACTIONS =====
print("\n===== DATABASE TRANSACTIONS =====")
"""
Transactions ensure that a series of database operations either all succeed or all fail.

Atomic transactions:
```python
from django.db import transaction

# Transaction as a context manager
def transfer_funds(from_account, to_account, amount):
    with transaction.atomic():
        from_account.balance -= amount
        from_account.save()
        
        # If this fails, the entire transaction is rolled back
        to_account.balance += amount
        to_account.save()

# Transaction as a decorator
@transaction.atomic
def create_order(customer, products):
    # Create order
    order = Order.objects.create(customer=customer)
    
    # Create order items
    for product_id, quantity in products.items():
        product = Product.objects.get(id=product_id)
        OrderItem.objects.create(
            order=order,
            product=product,
            quantity=quantity,
            price=product.price
        )
        
        # Update stock
        product.stock -= quantity
        product.save()
    
    return order

# Savepoints
def complex_operation():
    with transaction.atomic():
        # First operation
        operation_1()
        
        # Create a savepoint
        sid = transaction.savepoint()
        try:
            # Second operation
            operation_2()
        except Exception:
            # Roll back to savepoint if operation_2 fails
            transaction.savepoint_rollback(sid)
        else:
            # Commit savepoint if operation_2 succeeds
            transaction.savepoint_commit(sid)
        
        # Third operation (always executed)
        operation_3()
```

Transaction management:
```python
# Default behavior (autocommit)
# Each query is committed immediately

# Disable autocommit
transaction.set_autocommit(False)
try:
    # Operations...
    transaction.commit()
except Exception:
    transaction.rollback()
finally:
    transaction.set_autocommit(True)

# Non-atomic requests (for long-running processes)
@transaction.non_atomic_requests
def long_running_view(request):
    # This view won't be wrapped in a transaction
    pass
```

Database-level constraints:
```python
class Order(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    total = models.DecimalField(max_digits=10, decimal_places=2)
    
    class Meta:
        constraints = [
            models.CheckConstraint(
                check=models.Q(total__gte=0),
                name='total_gte_0'
            )
        ]
```
"""

# ===== OPTIMIZING DATABASE PERFORMANCE =====
print("\n===== OPTIMIZING DATABASE PERFORMANCE =====")
"""
Optimizing database performance is crucial for scalable Django applications.

Indexing:
```python
class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    class Meta:
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['price']),
            models.Index(fields=['name', 'price']),
        ]
```

Select related (for ForeignKey and OneToOneField):
```python
# Without select_related (generates multiple queries)
product = Product.objects.get(id=1)
category_name = product.category.name  # Additional query

# With select_related (generates a single query with JOIN)
product = Product.objects.select_related('category').get(id=1)
category_name = product.category.name  # No additional query
```

Prefetch related (for ManyToManyField and reverse ForeignKey):
```python
# Without prefetch_related (generates multiple queries)
category = Category.objects.get(id=1)
for product in category.products.all():  # Additional query
    print(product.name)

# With prefetch_related (generates two efficient queries)
category = Category.objects.prefetch_related('products').get(id=1)
for product in category.products.all():  # No additional query
    print(product.name)
```

Combining select_related and prefetch_related:
```python
# Complex example
orders = Order.objects.select_related('customer').prefetch_related(
    'items__product',
    'items__product__category'
).filter(created_at__gte='2023-01-01')
```

Defer and only:
```python
# Defer loads all fields except those specified
products = Product.objects.defer('description', 'created_at')

# Only loads only the specified fields
products = Product.objects.only('id', 'name', 'price')
```

Bulk operations:
```python
# Bulk create
Product.objects.bulk_create([
    Product(name='Product 1', price=10.99),
    Product(name='Product 2', price=20.99),
    Product(name='Product 3', price=30.99),
], batch_size=100)

# Bulk update
products = list(Product.objects.filter(price__lt=10))
for product in products:
    product.price *= 1.1
Product.objects.bulk_update(products, ['price'], batch_size=100)

# Bulk delete
Product.objects.filter(is_active=False).delete()
```

Database connection pooling:
```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydatabase',
        'USER': 'mydatabaseuser',
        'PASSWORD': 'mypassword',
        'HOST': 'localhost',
        'PORT': '5432',
        'CONN_MAX_AGE': 600,  # Keep connections open for 10 minutes
    }
}
```

Query optimization tips:
1. Use specific field lookups instead of filtering in Python
2. Avoid N+1 query problems with select_related and prefetch_related
3. Use values() or values_list() when you only need specific fields
4. Add appropriate indexes to fields used in filtering and ordering
5. Use explain() to analyze query performance
6. Consider denormalization for read-heavy operations
7. Use database-specific features when appropriate
"""

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Django's database integration provides a powerful and flexible way to work with SQL databases:

1. The ORM abstracts away database-specific details while still allowing raw SQL when needed
2. Models provide a clean, Pythonic way to define your database schema
3. Migrations make it easy to evolve your database schema over time
4. The query API offers a rich set of tools for retrieving and manipulating data
5. Transactions ensure data integrity for complex operations
6. Performance optimization tools help your application scale

Best practices:
1. Design your models carefully to represent your domain
2. Use appropriate field types and relationships
3. Create and apply migrations systematically
4. Write efficient queries using select_related and prefetch_related
5. Use transactions to ensure data integrity
6. Add indexes to frequently queried fields
7. Monitor and optimize database performance as your application grows
""")

print("\n===== END OF DJANGO DATABASE INTEGRATION =====")