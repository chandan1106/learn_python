"""
Django Web Development: Advanced Django REST Framework Techniques
"""

# This file covers advanced techniques for Django REST Framework
# In a real environment, you would need to install: pip install djangorestframework

# ===== FILTERING AND SEARCHING =====
print("\n===== FILTERING AND SEARCHING =====")
"""
DRF provides powerful filtering capabilities for your API endpoints.

Basic filtering:
```python
# views.py
from rest_framework import generics
from .models import Product
from .serializers import ProductSerializer

class ProductList(generics.ListAPIView):
    serializer_class = ProductSerializer
    
    def get_queryset(self):
        queryset = Product.objects.all()
        category = self.request.query_params.get('category')
        if category is not None:
            queryset = queryset.filter(category__name=category)
        return queryset
```

Using django-filter:
```python
# Install django-filter
# pip install django-filter

# settings.py
INSTALLED_APPS = [
    # ...
    'django_filters',
]

REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ]
}

# views.py
from rest_framework import viewsets
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from .models import Product
from .serializers import ProductSerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['category', 'in_stock', 'price']
    search_fields = ['name', 'description']
    ordering_fields = ['price', 'created_at']
```

Custom filter backends:
```python
# filters.py
from django_filters import rest_framework as filters
from .models import Product

class ProductFilter(filters.FilterSet):
    min_price = filters.NumberFilter(field_name="price", lookup_expr='gte')
    max_price = filters.NumberFilter(field_name="price", lookup_expr='lte')
    category_name = filters.CharFilter(field_name="category__name", lookup_expr='icontains')
    
    class Meta:
        model = Product
        fields = ['category', 'in_stock', 'min_price', 'max_price', 'category_name']

# views.py
from .filters import ProductFilter

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    filterset_class = ProductFilter
```
"""

# ===== PAGINATION =====
print("\n===== PAGINATION =====")
"""
Pagination is essential for handling large result sets in your API.

Global pagination settings:
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10
}
```

Pagination classes:
```python
# Page number pagination (/?page=2)
from rest_framework.pagination import PageNumberPagination

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100

# Limit-offset pagination (/?limit=10&offset=20)
from rest_framework.pagination import LimitOffsetPagination

class LargeResultsSetPagination(LimitOffsetPagination):
    default_limit = 100
    max_limit = 1000

# Cursor pagination (for very large datasets)
from rest_framework.pagination import CursorPagination

class ProductCursorPagination(CursorPagination):
    ordering = '-created_at'
    page_size = 25
```

Using pagination in views:
```python
class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    pagination_class = StandardResultsSetPagination
```

Custom pagination response:
```python
class CustomPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_paginated_response(self, data):
        return Response({
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link()
            },
            'count': self.page.paginator.count,
            'total_pages': self.page.paginator.num_pages,
            'current_page': self.page.number,
            'results': data
        })
```
"""

# ===== VERSIONING =====
print("\n===== VERSIONING =====")
"""
API versioning helps you evolve your API without breaking client applications.

Versioning settings:
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
    'DEFAULT_VERSION': 'v1',
    'ALLOWED_VERSIONS': ['v1', 'v2'],
    'VERSION_PARAM': 'version',
}
```

Versioning schemes:
```python
# URL path versioning (/api/v1/products/)
from rest_framework.versioning import URLPathVersioning

# Query parameter versioning (/api/products/?version=v1)
from rest_framework.versioning import QueryParameterVersioning

# Header versioning (Accept: application/json; version=v1)
from rest_framework.versioning import AcceptHeaderVersioning

# Hostname versioning (v1.api.example.com)
from rest_framework.versioning import HostNameVersioning

# Namespace versioning (URLs included in 'v1:' namespace)
from rest_framework.versioning import NamespaceVersioning
```

Using versioning in views:
```python
class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    
    def get_serializer_class(self):
        if self.request.version == 'v1':
            return ProductSerializerV1
        return ProductSerializerV2
```

URL configuration for versioning:
```python
# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router_v1 = DefaultRouter()
router_v1.register(r'products', views.ProductViewSetV1)

router_v2 = DefaultRouter()
router_v2.register(r'products', views.ProductViewSetV2)

urlpatterns = [
    path('api/v1/', include(router_v1.urls)),
    path('api/v2/', include(router_v2.urls)),
]
```
"""

# ===== CONTENT NEGOTIATION =====
print("\n===== CONTENT NEGOTIATION =====")
"""
Content negotiation allows your API to serve different formats based on client requests.

Content negotiation settings:
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.MultiPartParser',
    ],
}
```

Available renderers:
```python
# JSON renderer (default)
from rest_framework.renderers import JSONRenderer

# Browsable API renderer
from rest_framework.renderers import BrowsableAPIRenderer

# XML renderer
from rest_framework_xml.renderers import XMLRenderer

# YAML renderer
from rest_framework_yaml.renderers import YAMLRenderer

# CSV renderer
from rest_framework_csv.renderers import CSVRenderer
```

Using renderers in views:
```python
from rest_framework.renderers import JSONRenderer, BrowsableAPIRenderer
from rest_framework_csv.renderers import CSVRenderer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    renderer_classes = [JSONRenderer, BrowsableAPIRenderer, CSVRenderer]
```

Custom renderer:
```python
from rest_framework.renderers import BaseRenderer

class PlainTextRenderer(BaseRenderer):
    media_type = 'text/plain'
    format = 'txt'
    
    def render(self, data, accepted_media_type=None, renderer_context=None):
        if isinstance(data, dict):
            return '\n'.join([f"{key}: {value}" for key, value in data.items()])
        return str(data)
```
"""

# ===== THROTTLING =====
print("\n===== THROTTLING =====")
"""
Throttling limits the rate at which clients can access your API.

Throttling settings:
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/day',
        'user': '1000/day'
    }
}
```

Throttling classes:
```python
# Anonymous users throttling
from rest_framework.throttling import AnonRateThrottle

# Authenticated users throttling
from rest_framework.throttling import UserRateThrottle

# Scoped throttling
from rest_framework.throttling import ScopedRateThrottle
```

Custom throttle rates:
```python
class BurstRateThrottle(UserRateThrottle):
    scope = 'burst'

class SustainedRateThrottle(UserRateThrottle):
    scope = 'sustained'

# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_RATES': {
        'burst': '60/min',
        'sustained': '1000/day'
    }
}
```

Using throttling in views:
```python
class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    throttle_classes = [AnonRateThrottle, UserRateThrottle]
```

Throttling specific actions:
```python
from rest_framework.decorators import action
from rest_framework.throttling import ScopedRateThrottle

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    
    @action(detail=True, methods=['post'], throttle_classes=[ScopedRateThrottle], throttle_scope='product_purchase')
    def purchase(self, request, pk=None):
        # Purchase logic here
        return Response({'status': 'purchase successful'})
```
"""

# ===== TESTING =====
print("\n===== TESTING =====")
"""
DRF provides tools for testing your API endpoints.

Basic API test:
```python
# tests.py
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from .models import Product

class ProductTests(APITestCase):
    def setUp(self):
        self.product = Product.objects.create(name='Test Product', price=9.99)
    
    def test_get_product_list(self):
        url = reverse('product-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
    
    def test_create_product(self):
        url = reverse('product-list')
        data = {'name': 'New Product', 'price': 19.99}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Product.objects.count(), 2)
        self.assertEqual(Product.objects.get(id=2).name, 'New Product')
```

Testing with authentication:
```python
from rest_framework.test import APIClient
from django.contrib.auth.models import User

class AuthenticatedProductTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.product = Product.objects.create(name='Test Product', price=9.99)
    
    def test_authenticated_get(self):
        url = reverse('product-detail', args=[self.product.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_unauthenticated_get(self):
        self.client.force_authenticate(user=None)
        url = reverse('product-detail', args=[self.product.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
```

Testing with token authentication:
```python
from rest_framework.authtoken.models import Token

class TokenAuthTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.token = Token.objects.create(user=self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token.key}')
    
    def test_token_auth(self):
        url = reverse('product-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
```

Testing ViewSets:
```python
from rest_framework.test import APIRequestFactory
from .views import ProductViewSet

class ViewSetTests(APITestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.product = Product.objects.create(name='Test Product', price=9.99)
    
    def test_list(self):
        request = self.factory.get('/api/products/')
        view = ProductViewSet.as_view({'get': 'list'})
        response = view(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_retrieve(self):
        request = self.factory.get('/api/products/1/')
        view = ProductViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=self.product.id)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
```
"""

# ===== DOCUMENTATION =====
print("\n===== DOCUMENTATION =====")
"""
Documenting your API helps developers understand how to use it.

Using drf-yasg for Swagger documentation:
```python
# Install drf-yasg
# pip install drf-yasg

# settings.py
INSTALLED_APPS = [
    # ...
    'drf_yasg',
]

# urls.py
from django.urls import path, include, re_path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
    openapi.Info(
        title="Products API",
        default_version='v1',
        description="API for managing products",
        terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    # Your API URLs
    path('api/', include('myapp.urls')),
    
    # Swagger documentation
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
```

Documenting ViewSets and serializers:
```python
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

class ProductSerializer(serializers.ModelSerializer):
    """
    Serializer for Product objects.
    """
    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'price']

class ProductViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing products.
    """
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    
    @swagger_auto_schema(
        operation_description="List all products",
        responses={200: ProductSerializer(many=True)}
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)
    
    @swagger_auto_schema(
        operation_description="Create a new product",
        request_body=ProductSerializer,
        responses={201: ProductSerializer}
    )
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)
    
    @swagger_auto_schema(
        operation_description="Mark a product as featured",
        responses={200: openapi.Response('Product marked as featured')}
    )
    @action(detail=True, methods=['post'])
    def mark_featured(self, request, pk=None):
        product = self.get_object()
        product.is_featured = True
        product.save()
        return Response({'status': 'product marked as featured'})
```
"""

# ===== BEST PRACTICES =====
print("\n===== BEST PRACTICES =====")
"""
Best practices for building RESTful APIs with Django REST Framework:

1. API Design:
   - Use nouns for resource names (e.g., /products, not /getProducts)
   - Use plural resource names (e.g., /products, not /product)
   - Use nested resources for relationships (e.g., /categories/1/products)
   - Use HTTP methods appropriately (GET, POST, PUT, PATCH, DELETE)
   - Use appropriate status codes (200, 201, 204, 400, 401, 403, 404, etc.)

2. Serializers:
   - Keep serializers focused and cohesive
   - Use different serializers for different use cases
   - Validate data at the serializer level
   - Use nested serializers for related objects
   - Use read_only and write_only fields appropriately

3. Views and ViewSets:
   - Use ViewSets for standard CRUD operations
   - Use custom actions for non-CRUD operations
   - Override get_queryset for dynamic filtering
   - Use appropriate permission classes
   - Use throttling for rate limiting

4. Performance:
   - Use select_related and prefetch_related to optimize queries
   - Use pagination for large result sets
   - Use filtering to reduce data transfer
   - Cache responses when appropriate
   - Use database indexes for frequently queried fields

5. Security:
   - Always use HTTPS in production
   - Use appropriate authentication (Token, JWT, OAuth)
   - Implement proper permission checks
   - Validate and sanitize all input
   - Use throttling to prevent abuse

6. Versioning:
   - Version your API from the beginning
   - Use a consistent versioning scheme
   - Maintain backward compatibility when possible
   - Document changes between versions

7. Documentation:
   - Document your API thoroughly
   - Include examples for each endpoint
   - Document error responses
   - Keep documentation up to date
   - Use tools like Swagger/OpenAPI

8. Testing:
   - Write tests for all endpoints
   - Test both positive and negative cases
   - Test with different authentication scenarios
   - Test performance with large datasets
   - Use continuous integration for automated testing
"""

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Django REST Framework provides a powerful toolkit for building Web APIs:

1. Serializers convert between Django models and API representations
2. Views and ViewSets handle API requests and responses
3. Routers automatically generate URL patterns
4. Authentication and permissions control access to your API
5. Filtering, pagination, and throttling help manage large datasets
6. Content negotiation supports multiple formats
7. Documentation tools help developers understand your API
8. Testing utilities ensure your API works correctly

By following best practices and leveraging DRF's features, you can build robust, 
scalable, and maintainable APIs that meet the needs of your clients.
""")

print("\n===== END OF ADVANCED DJANGO REST FRAMEWORK =====")