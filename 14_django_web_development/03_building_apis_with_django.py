"""
Django Web Development: Building APIs with Django REST Framework
"""

# This file covers building RESTful APIs with Django REST Framework
# In a real environment, you would need to install: pip install djangorestframework

# ===== INTRODUCTION TO DJANGO REST FRAMEWORK =====
print("\n===== INTRODUCTION TO DJANGO REST FRAMEWORK =====")
"""
Django REST Framework (DRF) is a powerful toolkit for building Web APIs with Django.

Key features:
1. Serialization - Convert Django models to JSON/XML and vice versa
2. Authentication - OAuth, Token, Session, JWT authentication
3. Viewsets and Routers - Quickly build CRUD APIs
4. Browsable API - Interactive API documentation
5. Permissions - Fine-grained access control
6. Content negotiation - Support multiple formats (JSON, XML, etc.)
7. Pagination - Handle large result sets
8. Filtering - Allow clients to filter results
9. Throttling - Rate limiting for API endpoints
10. Testing - Utilities for API testing
"""

# ===== INSTALLATION AND SETUP =====
print("\n===== INSTALLATION AND SETUP =====")
"""
To use Django REST Framework, you need to install it and add it to your Django project:

```bash
# Install Django REST Framework
pip install djangorestframework

# Optional packages
pip install markdown       # Markdown support for the browsable API
pip install django-filter  # Filtering support
```

Add REST Framework to your Django settings:
```python
# settings.py
INSTALLED_APPS = [
    # Django apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party apps
    'rest_framework',
    'rest_framework.authtoken',  # If using token authentication
    
    # Your apps
    'myapp',
]

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10
}
```

Include REST Framework URLs in your project:
```python
# urls.py
from django.urls import path, include

urlpatterns = [
    path('api-auth/', include('rest_framework.urls')),  # Browsable API login
    # Your API URLs will go here
]
```
"""

# ===== SERIALIZERS =====
print("\n===== SERIALIZERS =====")
"""
Serializers convert Django models to JSON/XML and vice versa.

Basic serializer:
```python
# serializers.py
from rest_framework import serializers
from .models import Product, Category

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'price', 'category', 'created_at']
```

Custom serializer:
```python
class CategorySerializer(serializers.ModelSerializer):
    products_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = Category
        fields = ['id', 'name', 'products_count']
```

Nested serializers:
```python
class ProductDetailSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)
    
    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'price', 'category', 'created_at']
```

Serializer with custom fields:
```python
class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.ReadOnlyField(source='category.name')
    formatted_price = serializers.SerializerMethodField()
    
    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'price', 'formatted_price', 
                  'category', 'category_name', 'created_at']
    
    def get_formatted_price(self, obj):
        return f"${obj.price:.2f}"
```

Serializer validation:
```python
class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'price', 'category']
    
    def validate_price(self, value):
        if value <= 0:
            raise serializers.ValidationError("Price must be greater than zero")
        return value
    
    def validate(self, data):
        # Cross-field validation
        if data['name'] == data['description']:
            raise serializers.ValidationError("Name and description must be different")
        return data
```

Creating and updating with serializers:
```python
# Create
serializer = ProductSerializer(data=request.data)
if serializer.is_valid():
    product = serializer.save()
    return Response(serializer.data, status=status.HTTP_201_CREATED)
return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Update
product = Product.objects.get(pk=pk)
serializer = ProductSerializer(product, data=request.data, partial=True)
if serializer.is_valid():
    serializer.save()
    return Response(serializer.data)
return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```
"""

# ===== VIEWS AND VIEWSETS =====
print("\n===== VIEWS AND VIEWSETS =====")
"""
DRF provides several ways to build API views, from function-based to class-based.

Function-based views:
```python
# views.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import Product
from .serializers import ProductSerializer

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def product_list(request):
    if request.method == 'GET':
        products = Product.objects.all()
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = ProductSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAuthenticated])
def product_detail(request, pk):
    try:
        product = Product.objects.get(pk=pk)
    except Product.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        serializer = ProductSerializer(product)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = ProductSerializer(product, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        product.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

Class-based views:
```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Product
from .serializers import ProductSerializer

class ProductList(APIView):
    def get(self, request):
        products = Product.objects.all()
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = ProductSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ProductDetail(APIView):
    def get_object(self, pk):
        try:
            return Product.objects.get(pk=pk)
        except Product.DoesNotExist:
            raise Http404
    
    def get(self, request, pk):
        product = self.get_object(pk)
        serializer = ProductSerializer(product)
        return Response(serializer.data)
    
    def put(self, request, pk):
        product = self.get_object(pk)
        serializer = ProductSerializer(product, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk):
        product = self.get_object(pk)
        product.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

Generic class-based views:
```python
from rest_framework import generics
from .models import Product
from .serializers import ProductSerializer

class ProductList(generics.ListCreateAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class ProductDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
```

ViewSets:
```python
from rest_framework import viewsets
from .models import Product, Category
from .serializers import ProductSerializer, CategorySerializer

class ProductViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing products.
    """
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """
    A viewset for viewing categories.
    """
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
```

Custom ViewSet actions:
```python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Product
from .serializers import ProductSerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    
    @action(detail=True, methods=['post'])
    def mark_featured(self, request, pk=None):
        product = self.get_object()
        product.is_featured = True
        product.save()
        return Response({'status': 'product marked as featured'})
    
    @action(detail=False)
    def featured(self, request):
        featured_products = Product.objects.filter(is_featured=True)
        serializer = self.get_serializer(featured_products, many=True)
        return Response(serializer.data)
```
"""

# ===== ROUTING =====
print("\n===== ROUTING =====")
"""
DRF provides routers to automatically generate URL patterns for ViewSets.

Basic routing:
```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('products/', views.ProductList.as_view(), name='product-list'),
    path('products/<int:pk>/', views.ProductDetail.as_view(), name='product-detail'),
]
```

Router for ViewSets:
```python
# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'products', views.ProductViewSet)
router.register(r'categories', views.CategoryViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

Nested routers:
```python
# With drf-nested-routers package
from rest_framework_nested import routers
from . import views

router = routers.DefaultRouter()
router.register(r'categories', views.CategoryViewSet)

products_router = routers.NestedDefaultRouter(router, r'categories', lookup='category')
products_router.register(r'products', views.ProductViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('', include(products_router.urls)),
]
```
"""

# ===== AUTHENTICATION AND PERMISSIONS =====
print("\n===== AUTHENTICATION AND PERMISSIONS =====")
"""
DRF provides various authentication and permission classes.

Authentication classes:
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.BasicAuthentication',
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ]
}
```

Token authentication setup:
```python
# settings.py
INSTALLED_APPS = [
    # ...
    'rest_framework.authtoken',
]

# After migrating, create a token
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User

user = User.objects.get(username='admin')
token, created = Token.objects.get_or_create(user=user)
print(token.key)
```

JWT authentication:
```python
# Install djangorestframework-simplejwt
# pip install djangorestframework-simplejwt

# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
}

# urls.py
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
```

Permission classes:
```python
# Built-in permissions
from rest_framework.permissions import (
    IsAuthenticated,
    IsAdminUser,
    IsAuthenticatedOrReadOnly,
    AllowAny,
    DjangoModelPermissions,
)

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
```

Custom permissions:
```python
# permissions.py
from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object to edit it.
    """
    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions are only allowed to the owner
        return obj.owner == request.user

# views.py
from .permissions import IsOwnerOrReadOnly

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]
```
"""

print("\n===== END OF DJANGO REST FRAMEWORK PART 1 =====")