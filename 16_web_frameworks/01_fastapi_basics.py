"""
Web Frameworks: FastAPI Basics and Secure API Development
"""

# ===== INTRODUCTION TO FASTAPI =====
print("\n===== INTRODUCTION TO FASTAPI =====")
"""
FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.

Key Features:
1. Fast: Very high performance, on par with NodeJS and Go
2. Fast to code: Increases development speed by ~200-300%
3. Fewer bugs: Reduces about 40% of human-induced errors
4. Intuitive: Great editor support with auto-completion
5. Easy: Designed to be easy to use and learn
6. Short: Minimizes code duplication
7. Robust: Production-ready code with automatic interactive documentation
8. Standards-based: Based on (and fully compatible with) OpenAPI and JSON Schema

Installation:
```bash
pip install fastapi uvicorn
```

Running a FastAPI application:
```bash
uvicorn main:app --reload
```
"""

# ===== BASIC FASTAPI APPLICATION =====
print("\n===== BASIC FASTAPI APPLICATION =====")
"""
Here's a simple FastAPI application:

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

This creates:
1. A root endpoint at "/"
2. An item endpoint at "/items/{item_id}" with:
   - A path parameter `item_id` that must be an integer
   - An optional query parameter `q`

FastAPI automatically:
- Validates the types of parameters
- Converts the parameters to the declared types
- Generates OpenAPI documentation
"""

# ===== REQUEST BODY =====
print("\n===== REQUEST BODY =====")
"""
FastAPI makes it easy to work with request bodies using Pydantic models:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

Benefits:
- Automatic request body parsing
- Data validation
- Editor support with auto-completion
- Automatic documentation
"""

# ===== PATH PARAMETERS AND QUERY PARAMETERS =====
print("\n===== PATH PARAMETERS AND QUERY PARAMETERS =====")
"""
FastAPI handles path parameters and query parameters with type validation:

```python
from fastapi import FastAPI, Path, Query

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(
    item_id: int = Path(..., title="The ID of the item", ge=1),
    q: str = Query(None, max_length=50),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
):
    return {
        "item_id": item_id,
        "q": q,
        "skip": skip,
        "limit": limit
    }
```

Features:
- Path parameters are part of the URL path
- Query parameters are the key-value pairs after the ? in the URL
- Both can have validation rules (min/max values, regex patterns, etc.)
- Both can have metadata (title, description) for documentation
"""

# ===== RESPONSE MODELS =====
print("\n===== RESPONSE MODELS =====")
"""
FastAPI allows you to declare the model used for the response:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: List[str] = []

@app.post("/items/", response_model=Item)
def create_item(item: Item):
    return item

class UserIn(BaseModel):
    username: str
    password: str
    email: str
    full_name: Optional[str] = None

class UserOut(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

@app.post("/users/", response_model=UserOut)
def create_user(user: UserIn):
    return user  # Password will be filtered out
```

Benefits:
- Automatic data filtering (e.g., removing password fields)
- Response validation
- Response schema documentation
- Clear API contracts
"""

# ===== DEPENDENCY INJECTION =====
print("\n===== DEPENDENCY INJECTION =====")
"""
FastAPI has a powerful dependency injection system:

```python
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()

async def verify_token(x_token: str = Header(...)):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")
    return x_token

async def verify_key(x_key: str = Header(...)):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key

@app.get("/items/", dependencies=[Depends(verify_token), Depends(verify_key)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]
```

Use cases:
- Database connections
- Authentication and authorization
- Logging
- Request validation
- Common parameters
"""

# ===== SECURITY AND AUTHENTICATION =====
print("\n===== SECURITY AND AUTHENTICATION =====")
"""
FastAPI provides several security utilities:

1. OAuth2 with Password (and hashing), Bearer token:

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    email: str
    full_name: str = None
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": user.username, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

2. JWT Tokens:

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt

SECRET_KEY = "YOUR_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user
```

Security best practices:
- Use HTTPS in production
- Store passwords with secure hashing (bcrypt, Argon2)
- Use JWT with short expiration times
- Implement proper CORS policies
- Use rate limiting to prevent abuse
"""

# ===== ERROR HANDLING =====
print("\n===== ERROR HANDLING =====")
"""
FastAPI provides a clean way to handle errors:

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )

@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn_name": name}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}
```

Error handling features:
- Built-in HTTPException for standard HTTP errors
- Custom exception handlers
- Request validation errors
- Global exception handlers
"""

# ===== MIDDLEWARE =====
print("\n===== MIDDLEWARE =====")
"""
FastAPI supports ASGI middleware:

```python
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

Common middleware use cases:
- CORS (Cross-Origin Resource Sharing)
- Authentication
- Request timing
- Logging
- Compression
- Rate limiting
"""

# ===== DATABASE INTEGRATION =====
print("\n===== DATABASE INTEGRATION =====")
"""
FastAPI works well with any database through ORMs like SQLAlchemy:

```python
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ItemDB(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    owner_id = Column(Integer, index=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

@app.post("/items/")
def create_item(title: str, description: str, db: Session = Depends(get_db)):
    db_item = ItemDB(title=title, description=description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/{item_id}")
def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemDB).filter(ItemDB.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

Database best practices:
- Use connection pooling
- Use dependency injection for database sessions
- Close connections properly
- Use migrations for schema changes
- Use transactions for data consistency
"""

# ===== ASYNC SUPPORT =====
print("\n===== ASYNC SUPPORT =====")
"""
FastAPI has first-class support for async/await:

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # Simulate an async operation (e.g., database query)
    await asyncio.sleep(1)
    return {"item_id": item_id}
```

Benefits of async:
- Handle more concurrent requests
- Better performance for I/O-bound operations
- Non-blocking code execution
- Works well with async databases (e.g., asyncpg)
"""

# ===== TESTING =====
print("\n===== TESTING =====")
"""
FastAPI makes testing easy with TestClient:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
```

Testing features:
- TestClient based on requests
- Easy to test status codes, headers, and response bodies
- Works with pytest
- Can test dependencies with overrides
"""

# ===== SECURITY BEST PRACTICES =====
print("\n===== SECURITY BEST PRACTICES =====")
"""
When building APIs with FastAPI, follow these security best practices:

1. Input Validation
   - Use Pydantic models for request validation
   - Define strict types and constraints
   - Validate all user inputs

2. Authentication and Authorization
   - Use OAuth2 or JWT for authentication
   - Implement proper role-based access control
   - Use secure password hashing (bcrypt, Argon2)
   - Set short expiration times for tokens

3. HTTPS
   - Always use HTTPS in production
   - Configure proper SSL/TLS settings
   - Use HTTP Strict Transport Security (HSTS)

4. CORS (Cross-Origin Resource Sharing)
   - Restrict allowed origins in production
   - Only allow necessary HTTP methods
   - Control allowed headers and credentials

5. Rate Limiting
   - Implement rate limiting to prevent abuse
   - Use tools like slowapi or custom middleware

6. Security Headers
   - Set appropriate security headers:
     * X-Content-Type-Options: nosniff
     * X-Frame-Options: DENY
     * Content-Security-Policy
     * X-XSS-Protection: 1; mode=block

7. Dependency Management
   - Keep dependencies updated
   - Use tools like safety or snyk to check for vulnerabilities

8. Error Handling
   - Don't expose sensitive information in error messages
   - Log errors properly but securely
   - Return appropriate status codes

9. Database Security
   - Use parameterized queries to prevent SQL injection
   - Limit database user permissions
   - Encrypt sensitive data

10. Logging and Monitoring
    - Implement proper logging
    - Monitor for suspicious activities
    - Set up alerts for security events
"""

# ===== DEPLOYMENT =====
print("\n===== DEPLOYMENT =====")
"""
Deploying FastAPI applications:

1. Docker Deployment:
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Gunicorn with Uvicorn workers:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

3. Deployment platforms:
   - Heroku
   - AWS (EC2, ECS, Lambda)
   - Google Cloud Run
   - Azure App Service
   - Digital Ocean App Platform

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
FastAPI is a powerful framework for building high-performance APIs with Python.
Its key advantages include:

1. Speed: Both in terms of execution and development time
2. Type safety: Reduces bugs through Python type hints
3. Automatic documentation: OpenAPI and Swagger UI out of the box
4. Modern Python features: Async support, type hints, dependency injection
5. Security: Built-in security utilities and best practices

To build secure and efficient APIs with FastAPI:
1. Use the built-in validation features
2. Implement proper authentication and authorization
3. Follow security best practices
4. Use async features for better performance
5. Write tests for your endpoints
6. Deploy with proper configuration

FastAPI is an excellent choice for building modern, secure, and high-performance APIs.
""")

print("\n===== END OF FASTAPI BASICS =====")