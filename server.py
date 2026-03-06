from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
import re
import uuid
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
import bcrypt
# Fix for passlib/bcrypt 4.0+ compatibility issue
if not hasattr(bcrypt, '__about__'):
    class About:
        __version__ = bcrypt.__version__
    bcrypt.__about__ = About()
from passlib.context import CryptContext
import qrcode
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
import base64
from pywebpush import webpush, WebPushException
from py_vapid import Vapid
from bson import ObjectId

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
DB_AVAILABLE = True
client = None
db = None
try:
    client = AsyncIOMotorClient(mongo_url)
    # Attempt a quick server_info() to trigger DNS/connection issues at startup
    # This is synchronous but will raise quickly if DNS fails for SRV
    client.server_info()
    db = client[os.environ['DB_NAME']]
except Exception as e:
    DB_AVAILABLE = False
    logger.error('Failed to initialize MongoDB client at startup: %s', e)
    db = None

# JWT Configuration
SECRET_KEY = os.environ.get('JWT_SECRET', 'hotel-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# VAPID Configuration for Push Notifications
# Generate keys if not exists
VAPID_PRIVATE_KEY = os.environ.get('VAPID_PRIVATE_KEY', '')
VAPID_PUBLIC_KEY = os.environ.get('VAPID_PUBLIC_KEY', '')
VAPID_EMAIL = os.environ.get('VAPID_EMAIL', 'mailto:admin@lavilladelice.com')

# Generate VAPID keys if not provided
if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
    vapid_keys_file = ROOT_DIR / 'vapid_keys.json'
    if vapid_keys_file.exists():
        with open(vapid_keys_file, 'r') as f:
            keys = json.load(f)
            VAPID_PRIVATE_KEY = keys['private_key']
            VAPID_PUBLIC_KEY = keys['public_key']
    else:
        # Generate new keys using py_vapid
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        vapid = Vapid()
        vapid.generate_keys()
        VAPID_PRIVATE_KEY = vapid.private_pem().decode('utf-8')
        # Get uncompressed public key and encode to urlsafe base64
        raw_pub = vapid.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
        VAPID_PUBLIC_KEY = base64.urlsafe_b64encode(raw_pub).decode('utf-8').rstrip('=')
        # Save keys
        with open(vapid_keys_file, 'w') as f:
            json.dump({
                'private_key': VAPID_PRIVATE_KEY,
                'public_key': VAPID_PUBLIC_KEY
            }, f)
        print(f"Generated new VAPID keys")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Create the main app
app = FastAPI(title="Hotel Restaurant API")

# In-memory mapping of user_id -> list of websocket connections and metadata
# Each entry: {"ws": WebSocket, "endpoint": Optional[str]}
user_ws: Dict[str, List[dict]] = {}
# Parse allowed origins from environment for debugging / WebSocket checks
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get('CORS_ORIGINS', '*').split(',') if o.strip()]

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Allowed CORS origins: {ALLOWED_ORIGINS}")

# ==================== MODELS ====================

class UserRole:
    ADMIN = "admin"
    STAFF = "staff"
    GUEST = "guest"
    SERVEUR = "serveur"
    CUISINIER = "cuisinier"
    MAGAZINIER = "magazinier"

class UserBase(BaseModel):
    email: str
    name: str
    role: str = UserRole.GUEST

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class User(UserBase):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class Category(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # "food" or "drink"
    description: Optional[str] = ""
    image_url: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CategoryCreate(BaseModel):
    name: str
    type: str
    description: Optional[str] = ""
    image_url: Optional[str] = ""

class MenuItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    price: float
    category_id: str
    image_url: Optional[str] = ""
    available: bool = True
    stock_quantity: Optional[int] = None  # For drinks inventory
    track_stock: bool = False  # If true, track stock quantity
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MenuItemCreate(BaseModel):
    name: str
    description: str
    price: float
    category_id: str
    image_url: Optional[str] = ""
    available: bool = True
    stock_quantity: Optional[int] = None
    track_stock: bool = False

class Room(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    number: str
    type: str  # "single", "double", "suite"
    price_per_night: float
    status: str = "available"  # "available", "occupied", "maintenance"
    description: Optional[str] = ""
    image_url: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RoomCreate(BaseModel):
    number: str
    type: str
    price_per_night: float
    status: str = "available"
    description: Optional[str] = ""
    image_url: Optional[str] = ""

class OrderItem(BaseModel):
    menu_item_id: str
    name: str
    price: float
    quantity: int

class Order(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    items: List[OrderItem]
    total: float
    status: str = "pending"  # "pending", "preparing", "ready", "delivered", "cancelled"
    customer_name: Optional[str] = ""
    room_number: Optional[str] = ""
    table_number: Optional[str] = ""
    notes: Optional[str] = ""
    created_by: Optional[str] = ""  # Who created this order/invoice
    created_by_name: Optional[str] = ""  # Name of the creator
    deleted_by: Optional[str] = ""  # Who deleted (if deleted)
    deleted_by_name: Optional[str] = ""  # Name of who deleted
    deleted_at: Optional[datetime] = None  # When deleted
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class OrderCreate(BaseModel):
    items: List[OrderItem]
    customer_name: Optional[str] = ""
    room_number: Optional[str] = ""
    table_number: Optional[str] = ""
    notes: Optional[str] = ""

class OrderUpdate(BaseModel):
    status: str

# Room Booking/Stay Models
class RoomBooking(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    room_id: str
    room_number: str
    room_type: str
    guest_name: str
    guest_email: Optional[str] = ""
    guest_phone: Optional[str] = ""
    check_in: str  # Date string YYYY-MM-DD
    check_out: str  # Date string YYYY-MM-DD
    nights: int
    price_per_night: float
    total_price: float
    status: str = "confirmed"  # "confirmed", "checked_in", "checked_out", "cancelled"
    notes: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RoomBookingCreate(BaseModel):
    room_id: str
    guest_name: str
    guest_email: Optional[str] = ""
    guest_phone: Optional[str] = ""
    check_in: str
    check_out: str
    notes: Optional[str] = ""

class BookingExtension(BaseModel):
    check_out: str

class AddItemToBooking(BaseModel):
    menu_item_id: str
    name: str
    price: float
    quantity: int

# Stock Exit Model
class StockExit(BaseModel):
    quantity: int
    reason: str  # "casse", "perime", "vol", "consommation_interne", "autre"
    notes: Optional[str] = ""
    date: Optional[str] = None

# Direct Invoice Model
class DirectInvoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    invoice_number: str = ""
    customer_name: str
    customer_email: Optional[str] = ""
    customer_phone: Optional[str] = ""
    items: List[OrderItem] = []
    subtotal: float = 0
    tax: float = 0
    total: float = 0
    payment_method: str = "cash"  # cash, card, room_charge
    room_number: Optional[str] = ""
    notes: Optional[str] = ""
    # credit fields removed
    status: str = "paid"  # paid, pending, cancelled
    created_by: Optional[str] = ""  # Who created this invoice
    created_by_name: Optional[str] = ""  # Name of the creator
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DirectInvoiceCreate(BaseModel):
    customer_name: str
    customer_email: Optional[str] = ""
    customer_phone: Optional[str] = ""
    items: List[OrderItem]
    payment_method: str = "cash"
    room_number: Optional[str] = ""
    notes: Optional[str] = ""
    # credit fields removed

# Push Notification Models
class PushSubscriptionKeys(BaseModel):
    p256dh: str
    auth: str

class PushSubscription(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    endpoint: str
    keys: PushSubscriptionKeys
    user_agent: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PushSubscriptionCreate(BaseModel):
    endpoint: str
    keys: PushSubscriptionKeys
    user_agent: Optional[str] = ""


class UnsubscribeOthersRequest(BaseModel):
    exclude_endpoint: Optional[str] = None


# Credit Models
class Credit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    invoice_id: Optional[str] = None
    customer_name: str
    customer_phone: Optional[str] = ""
    customer_email: Optional[str] = ""
    amount_due: float = 0.0
    balance: float = 0.0
    status: str = "open"  # open, paid
    notes: Optional[str] = ""
    created_by: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    items: List[OrderItem] = Field(default_factory=list)


class CreditCreate(BaseModel):
    invoice_id: Optional[str] = None
    customer_name: str
    customer_phone: Optional[str] = ""
    customer_email: Optional[str] = ""
    amount: float = 0.0
    notes: Optional[str] = ""
    items: List[OrderItem] = Field(default_factory=list)

# Message Models
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    receiver_id: str
    content: str
    read: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MessageCreate(BaseModel):
    receiver_id: str
    content: str

# Audit Log Model
class AuditLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    user_name: str
    user_role: str
    action: str
    details: Optional[str] = ""
    metadata: Optional[Dict] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ==================== HELPER FUNCTIONS ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
    if user_doc is None:
        raise credentials_exception
    return User(**user_doc)

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_roles(*allowed_roles):
    """Dependency factory that ensures the current user has one of the allowed roles."""
    async def _dep(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: insufficient role"
            )
        return current_user
    return _dep

def serialize_datetime(doc: dict) -> dict:
    """Convert datetime objects to ISO strings for JSON serialization"""
    if 'created_at' in doc and isinstance(doc['created_at'], datetime):
        doc['created_at'] = doc['created_at'].isoformat()
    return doc

def deserialize_datetime(doc: dict) -> dict:
    """Convert ISO strings back to datetime objects"""
    if 'created_at' in doc and isinstance(doc['created_at'], str):
        doc['created_at'] = datetime.fromisoformat(doc['created_at'])
    return doc

async def log_action(user: User, action: str, details: str = "", metadata: dict = None, timestamp: datetime = None):
    """Log an administrative action to the database."""
    try:
        log = AuditLog(
            user_id=user.id,
            user_name=user.name,
            user_role=user.role,
            action=action,
            details=details,
            metadata=metadata or {},
            created_at=timestamp or datetime.now(timezone.utc)
        )
        doc = log.model_dump()
        doc = serialize_datetime(doc)
        await db.audit_logs.insert_one(doc)
    except Exception as e:
        logger.error(f"Failed to log action: {e}")

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    # Check if user exists
    existing = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        email=user_data.email,
        name=user_data.name,
        role=user_data.role
    )
    user_in_db = UserInDB(
        **user.model_dump(),
        hashed_password=get_password_hash(user_data.password)
    )
    
    doc = user_in_db.model_dump()
    doc = serialize_datetime(doc)
    await db.users.insert_one(doc)
    
    access_token = create_access_token(data={"sub": user.id})
    return Token(access_token=access_token, token_type="bearer", user=user)

@api_router.post("/auth/login", response_model=Token)
async def login(login_data: UserLogin):
    logger.info(f"⚡ TENTATIVE DE CONNEXION REÇUE pour : {login_data.email}")
    # Normalize email input (strip whitespace)
    email_input = login_data.email.strip()
    
    # Escape regex special characters to prevent errors
    safe_email = re.escape(email_input)
    
    # Find user case-insensitively (regex search)
    user_doc = await db.users.find_one(
        {"email": {"$regex": f"^{safe_email}$", "$options": "i"}}, 
        {"_id": 0}
    )
    
    if not user_doc:
        logger.warning(f"LOGIN DEBUG: User '{email_input}' not found in DB.")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password with debug log
    is_valid = verify_password(login_data.password, user_doc.get("hashed_password", ""))
    logger.info(f"LOGIN DEBUG: User found. Password valid? {is_valid}")

    if not is_valid:
        logger.warning(f"LOGIN DEBUG: Password mismatch for {email_input}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_doc = deserialize_datetime(user_doc)
    user = User(**user_doc)
    access_token = create_access_token(data={"sub": user.id})
    return Token(access_token=access_token, token_type="bearer", user=user)

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ==================== DEBUG ROUTES ====================
@api_router.get("/debug/force-reset")
async def force_reset_user():
    """Route de secours pour réinitialiser manuellement l'utilisateur sept@gmail.com"""
    if db is None:
        return {"error": "Database not connected"}
        
    email = "sept@gmail.com"
    password = "123456"
    hashed = get_password_hash(password)
    
    # 1. Supprimer l'ancien (pour être sûr à 100%)
    await db.users.delete_many({"email": {"$regex": f"^{email}$", "$options": "i"}})
    
    # 2. Recréer le nouveau
    user = UserInDB(
        id=str(uuid.uuid4()),
        email=email,
        name="Bertrand Sept",
        role=UserRole.SERVEUR,
        hashed_password=hashed
    )
    doc = serialize_datetime(user.model_dump())
    await db.users.insert_one(doc)
    
    return {"message": f"SUCCÈS : Utilisateur {email} recréé avec le mot de passe {password}. Réessaie de te connecter !"}

# ==================== USER MANAGEMENT (ADMIN ONLY) ====================

@api_router.get("/users", response_model=List[User])
async def get_users(admin: User = Depends(get_admin_user)):
    users = await db.users.find({}, {"_id": 0, "hashed_password": 0}).to_list(1000)
    return [deserialize_datetime(u) for u in users]

@api_router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: User = Depends(get_admin_user)):
    user_to_delete = await db.users.find_one({"id": user_id})
    result = await db.users.delete_one({"id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    name = user_to_delete.get("name", "Inconnu") if user_to_delete else "Inconnu"
    await log_action(admin, "Suppression utilisateur", f"Nom: {name} (ID: {user_id})")
    return {"message": "User deleted"}

@api_router.put("/users/{user_id}/role")
async def update_user_role(user_id: str, role: str, admin: User = Depends(get_admin_user)):
    if role not in [UserRole.ADMIN, UserRole.STAFF, UserRole.GUEST, UserRole.SERVEUR, UserRole.CUISINIER, UserRole.MAGAZINIER]:
        raise HTTPException(status_code=400, detail="Invalid role")
    user_to_update = await db.users.find_one({"id": user_id})
    result = await db.users.update_one({"id": user_id}, {"$set": {"role": role}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    name = user_to_update.get("name", "Inconnu") if user_to_update else "Inconnu"
    await log_action(admin, "Modification rôle utilisateur", f"Utilisateur: {name}, Nouveau rôle: {role}")
    return {"message": "Role updated"}

# ==================== CATEGORY ROUTES ====================

@api_router.get("/categories", response_model=List[Category])
async def get_categories():
    categories = await db.categories.find({}, {"_id": 0}).to_list(1000)
    return [deserialize_datetime(c) for c in categories]

@api_router.post("/categories", response_model=Category)
async def create_category(category_data: CategoryCreate, admin: User = Depends(get_admin_user)):
    category = Category(**category_data.model_dump())
    doc = category.model_dump()
    doc = serialize_datetime(doc)
    await db.categories.insert_one(doc)
    return category

@api_router.put("/categories/{category_id}", response_model=Category)
async def update_category(category_id: str, category_data: CategoryCreate, admin: User = Depends(get_admin_user)):
    update_data = category_data.model_dump()
    result = await db.categories.update_one({"id": category_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    updated = await db.categories.find_one({"id": category_id}, {"_id": 0})
    return Category(**deserialize_datetime(updated))

@api_router.delete("/categories/{category_id}")
async def delete_category(category_id: str, admin: User = Depends(get_admin_user)):
    result = await db.categories.delete_one({"id": category_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": "Category deleted"}

# ==================== MENU ITEM ROUTES ====================

@api_router.get("/menu", response_model=List[MenuItem])
async def get_menu_items():
    items = await db.menu_items.find({}, {"_id": 0}).to_list(1000)
    return [deserialize_datetime(i) for i in items]

@api_router.get("/menu/{item_id}", response_model=MenuItem)
async def get_menu_item(item_id: str):
    item = await db.menu_items.find_one({"id": item_id}, {"_id": 0})
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    return MenuItem(**deserialize_datetime(item))

@api_router.post("/menu", response_model=MenuItem)
async def create_menu_item(item_data: MenuItemCreate, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF))):
    item = MenuItem(**item_data.model_dump())
    doc = item.model_dump()
    doc = serialize_datetime(doc)
    await db.menu_items.insert_one(doc)
    
    stock_val = item.stock_quantity if item.stock_quantity is not None else 0
    await log_action(current_user, "Création produit", f"Nom: {item.name}, Prix: {item.price}", metadata={
        "product_id": item.id,
        "product_name": item.name,
        "stock_before": 0,
        "stock_after": stock_val,
        "quantity_change": stock_val
    })
    return item

@api_router.put("/menu/{item_id}", response_model=MenuItem)
async def update_menu_item(item_id: str, item_data: MenuItemCreate, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF))):
    # Fetch old item first to get stock before update
    old_item = await db.menu_items.find_one({"id": item_id})
    if not old_item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    
    old_stock = old_item.get("stock_quantity")
    if old_stock is None: old_stock = 0

    update_data = item_data.model_dump()
    result = await db.menu_items.update_one({"id": item_id}, {"$set": update_data})
    
    updated = await db.menu_items.find_one({"id": item_id}, {"_id": 0})
    new_stock = updated.get("stock_quantity")
    if new_stock is None: new_stock = 0

    await log_action(current_user, "Modification produit", f"Nom: {item_data.name} (ID: {item_id})", metadata={
        "product_id": item_id,
        "product_name": item_data.name,
        "stock_before": old_stock,
        "stock_after": new_stock,
        "quantity_change": new_stock - old_stock
    })
    return MenuItem(**deserialize_datetime(updated))

@api_router.delete("/menu/{item_id}")
async def delete_menu_item(item_id: str, admin: User = Depends(get_admin_user)):
    item = await db.menu_items.find_one({"id": item_id})
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
        
    stock = item.get("stock_quantity")
    if stock is None: stock = 0
    
    result = await db.menu_items.delete_one({"id": item_id})
    name = item.get("name", "Inconnu") if item else "Inconnu"
    await log_action(admin, "Suppression produit", f"Nom: {name}", metadata={
        "product_id": item_id,
        "product_name": name,
        "stock_before": stock,
        "stock_after": 0,
        "quantity_change": -stock
    })
    return {"message": "Menu item deleted"}

# Stock Management
@api_router.put("/menu/{item_id}/stock")
async def update_stock(item_id: str, quantity: int, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF))):
    """Update stock quantity for a menu item"""
    item = await db.menu_items.find_one({"id": item_id})
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
        
    old_stock = item.get("stock_quantity", 0) or 0
    
    result = await db.menu_items.update_one(
        {"id": item_id}, 
        {"$set": {"stock_quantity": quantity}}
    )
    await log_action(current_user, "Mise à jour stock", f"ID: {item_id}, Nouvelle quantité: {quantity}", metadata={
        "product_id": item_id,
        "product_name": item.get("name"),
        "stock_before": old_stock,
        "stock_after": quantity,
        "quantity_change": quantity - old_stock
    })
    return {"message": "Stock updated", "quantity": quantity}

@api_router.post("/menu/{item_id}/stock/add")
async def add_stock(item_id: str, quantity: int, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF))):
    """Add stock to a menu item"""
    item = await db.menu_items.find_one({"id": item_id}, {"_id": 0})
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    
    current_stock = item.get("stock_quantity", 0) or 0
    new_stock = current_stock + quantity
    
    await db.menu_items.update_one(
        {"id": item_id}, 
        {"$set": {"stock_quantity": new_stock, "track_stock": True}}
    )
    await log_action(current_user, "Ajout stock", f"Produit: {item.get('name')}, Ajout: {quantity}, Total: {new_stock}", metadata={
        "product_id": item_id,
        "product_name": item.get("name"),
        "stock_before": current_stock,
        "stock_after": new_stock,
        "quantity_change": quantity
    })
    return {"message": "Stock added", "new_quantity": new_stock}

@api_router.post("/menu/{item_id}/stock/remove")
async def remove_stock(
    item_id: str, 
    payload: StockExit, 
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.MAGAZINIER))
):
    """Remove stock from a menu item (for breakage, spoilage, etc.)"""
    item = await db.menu_items.find_one({"id": item_id}, {"_id": 0})
    if not item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    
    current_stock = item.get("stock_quantity", 0) or 0
    new_stock = max(0, current_stock - payload.quantity)
    
    await db.menu_items.update_one(
        {"id": item_id}, 
        {"$set": {"stock_quantity": new_stock, "track_stock": True}}
    )
    
    details = f"Produit: {item.get('name')}, Retrait: {payload.quantity}, Motif: {payload.reason}"
    if payload.notes:
        details += f" ({payload.notes})"

    # Handle custom date if provided
    action_date = None
    if payload.date:
        try:
            action_date = datetime.strptime(payload.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    await log_action(current_user, "Sortie stock", details, metadata={
        "product_id": item_id,
        "product_name": item.get("name"),
        "stock_before": current_stock,
        "stock_after": new_stock,
        "quantity_change": -payload.quantity,
        "reason": payload.reason,
        "notes": payload.notes
    }, timestamp=action_date)
    return {"message": "Stock removed", "new_quantity": new_stock}

# ==================== DIRECT INVOICES ====================

@api_router.get("/invoices")
async def get_all_invoices(current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))):
    """Get all direct invoices (admin/staff/serveur only)."""
    invoices = await db.direct_invoices.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [deserialize_datetime(i) for i in invoices]

@api_router.get("/invoices/history/pdf")
async def export_invoices_pdf(
    agent: Optional[str] = None,
    payment: Optional[str] = None,
    start: Optional[str] = None,  # YYYY-MM-DD
    end: Optional[str] = None,    # YYYY-MM-DD
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))
):
    """Export filtered invoices history to a compact PDF list."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    query: dict = {}
    if agent:
        query["$or"] = [{"created_by_name": agent}, {"created_by": agent}]
    if payment and payment != "all":
        query["payment_method"] = payment
    # Date filter (inclusive)
    if start or end:
        date_filter: dict = {}
        if start:
            try:
                s = datetime.fromisoformat(start).date()
                date_filter["$gte"] = datetime(s.year, s.month, s.day, 0, 0, 0, tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
        if end:
            try:
                e = datetime.fromisoformat(end).date()
                date_filter["$lte"] = datetime(e.year, e.month, e.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
        if date_filter:
            query["created_at"] = date_filter

    cursor = db.direct_invoices.find(query, {"_id": 0}).sort("created_at", -1).limit(2000)
    invoices = await cursor.to_list(length=2000)

    buffer = BytesIO()
    # Ticket-like width
    page_width = 300
    doc = SimpleDocTemplate(buffer, pagesize=(page_width, 900), rightMargin=10, leftMargin=10, topMargin=15, bottomMargin=15)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Normal'], fontSize=14, fontName='Helvetica-Bold', alignment=1, spaceAfter=8)
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, alignment=1, spaceAfter=2)

    elements.append(Paragraph("Historique des Factures", title_style))
    sub = []
    if agent: sub.append(f"Facturé par: {agent}")
    if payment and payment != "all": sub.append(f"Paiement: {payment}")
    if start or end: sub.append(f"Période: {start or '...'} → {end or '...'}")
    if sub:
        elements.append(Paragraph(" | ".join(sub), small))
    elements.append(Spacer(1, 4))

    data = [["N°", "Client", "Agent", "Date", "Pay.", "Total"]]
    for inv in invoices:
        num = inv.get("invoice_number", "")
        client = (inv.get("customer_name") or "")[:14]
        agent_name = inv.get("created_by_name") or inv.get("created_by") or ""
        date_str = inv.get("created_at")
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00')) if isinstance(date_str, str) else date_str
            date_fmt = dt.strftime("%d/%m")
        except Exception:
            date_fmt = ""
        pay = inv.get("payment_method", "")
        total = f"{float(inv.get('total', 0)):.2f}"
        data.append([num, client, agent_name[:10], date_fmt, pay[:6], total])

    table = Table(data, colWidths=[60, 70, 60, 40, 35, 35])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (3, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (5, 1), (5, -1), 'RIGHT'),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=invoices_history.pdf"}
    )

@api_router.post("/invoices/create-debug")
async def create_direct_invoice_debug(invoice_data: DirectInvoiceCreate, current_user: User = Depends(get_current_user)):
    """Debug-only endpoint: build invoice payload without persisting to DB.

    Useful to determine if serialization issues come from the prepared document
    or from database insertion side-effects.
    """
    subtotal = sum(item.price * item.quantity for item in invoice_data.items)
    total = subtotal

    count = await db.direct_invoices.count_documents({})
    invoice_number = f"FACT-{count + 1:06d}"

    items_with_stock = []
    for item in invoice_data.items:
        menu_item = await db.menu_items.find_one({"id": item.menu_item_id}, {"_id": 0})
        stock_before = None
        stock_after = None
        if menu_item and menu_item.get("track_stock"):
            stock_before = menu_item.get("stock_quantity", 0) or 0
            stock_after = max(0, stock_before - item.quantity)

        if hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = item.copy()
        else:
            try:
                item_dict = dict(item)
            except Exception:
                item_dict = {
                    'menu_item_id': getattr(item, 'menu_item_id', None),
                    'name': getattr(item, 'name', None),
                    'price': getattr(item, 'price', None),
                    'quantity': getattr(item, 'quantity', None)
                }

        item_dict["stock_before"] = stock_before
        item_dict["stock_after"] = stock_after
        item_dict["unit_price"] = item.price
        item_dict["total_price"] = round(item.price * item.quantity, 2)
        items_with_stock.append(item_dict)

    invoice = DirectInvoice(
        invoice_number=invoice_number,
        customer_name=invoice_data.customer_name,
        customer_email=invoice_data.customer_email,
        customer_phone=invoice_data.customer_phone,
        items=invoice_data.items,
        subtotal=subtotal,
        tax=0,
        total=total,
        payment_method=invoice_data.payment_method,
        room_number=invoice_data.room_number,
        notes=invoice_data.notes,
        created_by=current_user.email,
        created_by_name=current_user.name
    )

    doc = invoice.model_dump()
    doc['items'] = items_with_stock
    doc = serialize_datetime(doc)
    # Sanitize BSON types before returning
    return sanitize_bson(doc)

@api_router.post("/invoices/create")
async def create_direct_invoice(invoice_data: DirectInvoiceCreate, current_user: User = Depends(get_current_user)):
    """Create a direct invoice (for restaurant billing)"""
    logger.info('create_direct_invoice called by %s with %d items', getattr(current_user, 'email', 'anonymous'), len(getattr(invoice_data, 'items', [])))
    try:
        # Calculate totals
        subtotal = sum(item.price * item.quantity for item in invoice_data.items)
        total = subtotal  # Sans TVA

        # Generate invoice number
        count = await db.direct_invoices.count_documents({})
        invoice_number = f"FACT-{count + 1:06d}"

        # Build enriched items list including stock before/after and price totals
        items_with_stock = []
        for item in invoice_data.items:
            menu_item = await db.menu_items.find_one({"id": item.menu_item_id}, {"_id": 0})
            stock_before = None
            stock_after = None
            if menu_item and menu_item.get("track_stock"):
                stock_before = menu_item.get("stock_quantity", 0) or 0
                stock_after = max(0, stock_before - item.quantity)
                await db.menu_items.update_one(
                    {"id": item.menu_item_id},
                    {"$set": {"stock_quantity": stock_after}}
                )

            # Support both pydantic models and plain dicts
            if hasattr(item, 'model_dump'):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = item.copy()
            else:
                try:
                    item_dict = dict(item)
                except Exception:
                    item_dict = {
                        'menu_item_id': getattr(item, 'menu_item_id', None),
                        'name': getattr(item, 'name', None),
                        'price': getattr(item, 'price', None),
                        'quantity': getattr(item, 'quantity', None)
                    }
            item_dict["stock_before"] = stock_before
            item_dict["stock_after"] = stock_after
            item_dict["unit_price"] = item.price
            item_dict["total_price"] = round(item.price * item.quantity, 2)
            items_with_stock.append(item_dict)

        invoice = DirectInvoice(
            invoice_number=invoice_number,
            customer_name=invoice_data.customer_name,
            customer_email=invoice_data.customer_email,
            customer_phone=invoice_data.customer_phone,
            items=invoice_data.items,
            subtotal=subtotal,
            tax=0,
            total=total,
            payment_method=invoice_data.payment_method,
            room_number=invoice_data.room_number,
            notes=invoice_data.notes,
            created_by=current_user.email,
            created_by_name=current_user.name
        )

        doc = invoice.model_dump()
        # Store enriched items in DB (with stock_before/stock_after and price totals)
        doc['items'] = items_with_stock
        doc = serialize_datetime(doc)
        # Debug: log types inside doc to catch non-serializable values
        try:
            logger.info('Invoice doc keys/types: %s', {k: type(v).__name__ for k, v in doc.items()})
            for idx, it in enumerate(doc.get('items', [])):
                logger.info('Item %s types: %s', idx, {k: type(v).__name__ for k, v in (it.items() if isinstance(it, dict) else [])})
        except Exception:
            logger.exception('Failed to log doc types')

        await db.direct_invoices.insert_one(doc)

        # If this invoice was created on credit, add a credit record for tracking
        try:
            if (doc.get('payment_method') or '') == 'credit':
                credit_doc = {
                    'id': str(uuid.uuid4()),
                    'invoice_id': doc.get('id'),
                    'invoice_number': doc.get('invoice_number'),
                    'customer_name': doc.get('customer_name'),
                    'customer_phone': doc.get('customer_phone'),
                    'customer_email': doc.get('customer_email'),
                    'created_by_name': doc.get('created_by_name'),
                    'subtotal': doc.get('subtotal', doc.get('total', 0)),
                    'amount_due': doc.get('total', 0),
                    'balance': doc.get('total', 0),
                    'items': doc.get('items', []),
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'created_by': doc.get('created_by'),
                }
                await db.credits.insert_one(serialize_datetime(credit_doc))
        except Exception:
            logger.exception('Failed to create credit record for invoice %s', doc.get('id'))

        # Return stored document (with enriched items)
        stored = deserialize_datetime(doc)
        stored = sanitize_bson(stored)
        logger.info('Returning sanitized invoice: %s', repr(stored)[:1000])
        return stored
    except Exception as e:
        # Log full exception for local debugging
        logger.exception('Error creating direct invoice')
        print('EXCEPTION in create_direct_invoice:', e)
        import traceback
        traceback.print_exc()
        # In dev return error detail to client to aid debugging
        detail = str(e)
        raise HTTPException(status_code=500, detail=detail)

def sanitize_bson(obj):
    """Recursively convert BSON ObjectId to str so JSON serialization succeeds."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: sanitize_bson(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_bson(v) for v in obj]
    return obj

@api_router.get("/invoices/direct/{invoice_id}")
async def get_direct_invoice(invoice_id: str, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))):
    """Get a direct invoice by ID (admin/staff/serveur only)."""
    invoice = await db.direct_invoices.find_one({"id": invoice_id}, {"_id": 0})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return deserialize_datetime(invoice)


@api_router.post("/invoices/{invoice_id}/pay")
async def pay_invoice(invoice_id: str, current_user: User = Depends(get_current_user)):
    """Mark an invoice as paid."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    invoice = await db.direct_invoices.find_one({"id": invoice_id})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    # Update status
    update = {
        "$set": {"status": "paid", "paid_by": current_user.email, "paid_at": datetime.now(timezone.utc).isoformat()}
    }
    await db.direct_invoices.update_one({"id": invoice_id}, update)

    updated = await db.direct_invoices.find_one({"id": invoice_id}, {"_id": 0})
    # credit tracking removed
    return sanitize_bson(deserialize_datetime(updated))


@api_router.get("/products/{menu_item_id}/stock-history")
async def get_product_stock_history(
    menu_item_id: str,
    page: int = 1,
    per_page: int = 20,
    current_user: User = Depends(get_current_user)
):
    """Return stock evolution for a given product/menu item with pagination.

    - `page`: 1-based page number
    - `per_page`: items per page (capped at 100)

    Results are sorted from most recent to oldest (newest first).
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if page < 1:
        page = 1
    per_page = max(1, min(per_page, 100))

    query = {"items.menu_item_id": menu_item_id}
    total = await db.direct_invoices.count_documents(query)

    skip = (page - 1) * per_page
    cursor = db.direct_invoices.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(per_page)
    invoices = await cursor.to_list(length=per_page)

    events = []
    for inv in invoices:
        created = inv.get("created_at")
        inv_id = inv.get("id")
        inv_number = inv.get("invoice_number")
        customer = inv.get("customer_name")
        for item in inv.get("items", []):
            if item.get("menu_item_id") == menu_item_id:
                events.append({
                    "created_at": created,
                    "invoice_id": inv_id,
                    "invoice_number": inv_number,
                    "customer": customer,
                    "quantity": item.get("quantity"),
                    "stock_before": item.get("stock_before"),
                    "stock_after": item.get("stock_after"),
                    "unit_price": item.get("unit_price") or item.get("price"),
                    "total_price": item.get("total_price") or round((item.get("price", 0) * item.get("quantity", 0)), 2),
                })

    # Ensure events are sorted from most recent to oldest
    def _parse_dt(v):
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except Exception:
                return datetime.min
        return datetime.min

    events.sort(key=lambda e: _parse_dt(e.get("created_at", "")), reverse=True)

    return {
        "events": events,
        "page": page,
        "per_page": per_page,
        "total": total,
    }


@api_router.get("/products/history/logs")
async def get_products_history_logs(
    page: int = 1,
    per_page: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product_name: Optional[str] = None,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.MAGAZINIER, UserRole.SERVEUR))
):
    """Get audit logs specifically for product stock changes."""
    query = {
        "action": {"$in": ["Ajout stock", "Sortie stock", "Mise à jour stock", "Création produit", "Suppression produit", "Modification produit"]}
    }
    
    if start_date or end_date:
        date_filter = {}
        if start_date:
            try:
                s = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                date_filter["$gte"] = s.isoformat()
            except ValueError: pass
        if end_date:
            try:
                e = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                date_filter["$lte"] = e.isoformat()
            except ValueError: pass
        if date_filter:
            query["created_at"] = date_filter
            
    if product_name:
        query["metadata.product_name"] = {"$regex": product_name, "$options": "i"}

    total = await db.audit_logs.count_documents(query)
    skip = (page - 1) * per_page

    logs = await db.audit_logs.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(per_page).to_list(per_page)
    return {
        "logs": [deserialize_datetime(l) for l in logs],
        "total": total,
        "page": page,
        "per_page": per_page
    }

@api_router.get("/credits")
async def list_credits(status: Optional[str] = None, admin: User = Depends(get_admin_user)):
    """List all credits (admin only)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    query = {}
    if status:
        query["status"] = status
    credits = await db.credits.find(query, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [deserialize_datetime(c) for c in credits]


@api_router.post("/credits")
async def create_credit(payload: CreditCreate, current_user: User = Depends(get_current_user)):
    """Create a credit record (can be used by staff/admin)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Compute amount from items if not provided
    computed_total = 0.0
    try:
        computed_total = sum((i.price * i.quantity) for i in payload.items) if payload.items else 0.0
    except Exception:
        computed_total = 0.0
    final_amount = payload.amount or computed_total

    credit = Credit(
        invoice_id=payload.invoice_id,
        customer_name=payload.customer_name,
        customer_phone=payload.customer_phone,
        customer_email=payload.customer_email,
        amount_due=final_amount,
        balance=final_amount,
        notes=payload.notes,
        created_by=current_user.email,
        items=payload.items
    )
    doc = credit.model_dump()
    doc = serialize_datetime(doc)
    await db.credits.insert_one(doc)
    stored = deserialize_datetime(doc)
    return sanitize_bson(stored)


@api_router.post("/credits/{credit_id}/pay")
async def pay_credit(credit_id: str, current_user: User = Depends(get_current_user)):
    """Mark a credit as paid. Also mark linked invoice paid if present."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    credit = await db.credits.find_one({"id": credit_id})
    if not credit:
        raise HTTPException(status_code=404, detail="Credit not found")

    update = {
        "$set": {"status": "paid", "balance": 0, "paid_by": current_user.email, "paid_at": datetime.now(timezone.utc).isoformat()}
    }
    await db.credits.update_one({"id": credit_id}, update)

    # If linked to an invoice, mark invoice paid as well
    try:
        inv_id = credit.get('invoice_id')
        if inv_id:
            await db.direct_invoices.update_one({"id": inv_id}, {"$set": {"status": "paid", "paid_by": current_user.email, "paid_at": datetime.now(timezone.utc).isoformat()}})
    except Exception:
        logger.exception('Failed to update linked invoice for credit %s', credit_id)

    updated = await db.credits.find_one({"id": credit_id}, {"_id": 0})
    return sanitize_bson(deserialize_datetime(updated))

@api_router.get("/credits/{credit_id}/pdf")
async def generate_credit_pdf(credit_id: str, current_user: User = Depends(get_current_user)):
    credit = await db.credits.find_one({"id": credit_id}, {"_id": 0})
    if not credit:
        raise HTTPException(status_code=404, detail="Credit not found")

    buffer = BytesIO()
    page_width = 226
    doc = SimpleDocTemplate(buffer, pagesize=(page_width, 900), rightMargin=10, leftMargin=10, topMargin=15, bottomMargin=15)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ThermalTitle', parent=styles['Normal'], fontSize=14, fontName='Helvetica-Bold', alignment=1, spaceAfter=5)
    center_style = ParagraphStyle('ThermalCenter', parent=styles['Normal'], fontSize=9, alignment=1, spaceAfter=3)
    small_style = ParagraphStyle('ThermalSmall', parent=styles['Normal'], fontSize=8, alignment=1, spaceAfter=2)
    section_style = ParagraphStyle('ThermalSection', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', alignment=1, spaceBefore=5, spaceAfter=3)

    elements.append(Paragraph("LA VILLA DELICE", title_style))
    elements.append(Paragraph("Restaurant & Bar", center_style))
    elements.append(Paragraph("Q. Les volcans, Avenue Grevaillas", small_style))
    elements.append(Paragraph("Numero 076 Goma Nord Kivu", small_style))
    elements.append(Paragraph("Tél: 980629999", small_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("-" * 35, center_style))

    elements.append(Paragraph("CRÉDIT CLIENT", section_style))
    inv_number = credit.get("invoice_number") or credit.get("invoice_id", "")[:8].upper() if credit.get("invoice_id") else ""
    if inv_number:
        elements.append(Paragraph(f"N°: {inv_number}", small_style))
    created_at = credit.get("created_at", datetime.now(timezone.utc).isoformat())
    try:
        created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00')) if isinstance(created_at, str) else created_at
    except Exception:
        created_dt = datetime.now(timezone.utc)
    elements.append(Paragraph(f"Date: {created_dt.strftime('%d/%m/%Y %H:%M')}", small_style))
    if credit.get("customer_name"):
        elements.append(Paragraph(f"Client: {credit.get('customer_name')}", small_style))
    if credit.get("created_by_name") or credit.get("created_by"):
        elements.append(Paragraph(f"Facturé par: {credit.get('created_by_name') or credit.get('created_by')}", small_style))
    elements.append(Paragraph("-" * 35, center_style))

    items = credit.get("items", [])
    if items:
        data = [["Qté", "Article", "Total"]]
        subtotal = 0.0
        for item in items:
            qty = int(item.get("quantity", 0))
            name = str(item.get("name", ""))[:15]
            price = float(item.get("price", 0))
            line_total = qty * price
            subtotal += line_total
            data.append([str(qty), name, f"{line_total:.2f} $"])
        table = Table(data, colWidths=[25, 115, 50])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(Paragraph("-" * 35, center_style))
        elements.append(Paragraph(f"Sous-total: {subtotal:.2f} $", ParagraphStyle('SubTotal', parent=styles['Normal'], fontSize=9, alignment=2)))

    total = float(credit.get("amount_due", 0))
    elements.append(Paragraph("=" * 35, center_style))
    elements.append(Paragraph(f"<b>TOTAL: {total:.2f} $</b>", ParagraphStyle('GrandTotal', parent=styles['Normal'], fontSize=12, fontName='Helvetica-Bold', alignment=1)))
    elements.append(Paragraph("=" * 35, center_style))

    status = credit.get("status", "open")
    elements.append(Paragraph(f"Statut: {status.upper()}", small_style))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Merci de votre visite !", center_style))
    elements.append(Paragraph("La Villa Delice", small_style))

    doc.build(elements)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=credit_{credit_id[:8]}.pdf"}
    )
@api_router.get("/invoices/direct/{invoice_id}/pdf")
async def generate_direct_invoice_pdf(invoice_id: str, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))):
    """Generate thermal receipt PDF for a direct invoice (80mm width)"""
    invoice = await db.direct_invoices.find_one({"id": invoice_id}, {"_id": 0})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    # Thermal receipt size: 80mm width = 226 points, variable height
    receipt_width = 226
    receipt_height = 800  # Will be trimmed
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=(receipt_width, receipt_height),
        rightMargin=10, leftMargin=10, topMargin=15, bottomMargin=15
    )
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles for thermal receipt
    center_style = ParagraphStyle('Center', parent=styles['Normal'], alignment=1, fontSize=9)
    bold_center = ParagraphStyle('BoldCenter', parent=styles['Normal'], alignment=1, fontSize=10, fontName='Helvetica-Bold')
    title_style = ParagraphStyle('Title', parent=styles['Normal'], alignment=1, fontSize=14, fontName='Helvetica-Bold')
    small_style = ParagraphStyle('Small', parent=styles['Normal'], alignment=1, fontSize=8)
    
    # Header - La Villa Delice
    elements.append(Paragraph("LA VILLA DELICE", title_style))
    elements.append(Paragraph("Restaurant & Bar", center_style))
    elements.append(Paragraph("Q. Les volcans, Avenue Grevaillas", small_style))
    elements.append(Paragraph("Numero 076 Goma Nord Kivu", small_style))
    elements.append(Paragraph("Tél: 980629999", small_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Invoice info
    elements.append(Paragraph(f"<b>{invoice.get('invoice_number', '')}</b>", bold_center))
    elements.append(Paragraph(datetime.fromisoformat(invoice.get("created_at", datetime.now(timezone.utc).isoformat())).strftime("%d/%m/%Y %H:%M"), center_style))
    
    if invoice.get("customer_name"):
        elements.append(Paragraph(f"Client: {invoice.get('customer_name')}", center_style))
    if invoice.get("room_number"):
        elements.append(Paragraph(f"Chambre: {invoice.get('room_number')}", center_style))
    
    payment_labels = {"cash": "Espèces", "card": "Carte Bancaire", "room_charge": "Note Chambre"}
    elements.append(Paragraph(f"Paiement: {payment_labels.get(invoice.get('payment_method', 'cash'), 'Espèces')}", center_style))
    
    elements.append(Paragraph("-" * 35, center_style))
    
    # Items header
    elements.append(Paragraph("Qté  Article              Total", ParagraphStyle('Header', parent=styles['Normal'], fontSize=8, fontName='Courier-Bold')))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Items
    for item in invoice.get("items", []):
        qty = str(item.get("quantity", 0)).ljust(3)
        name = item.get("name", "")[:18].ljust(18)
        total = f"{item.get('price', 0) * item.get('quantity', 0):.2f} $"
        elements.append(Paragraph(f"{qty} {name} {total}", ParagraphStyle('Item', parent=styles['Normal'], fontSize=8, fontName='Courier')))
    
    elements.append(Paragraph("-" * 35, center_style))
    
    # Totals (sans TVA)
    subtotal = invoice.get('subtotal', 0)
    total = subtotal
    
    elements.append(Paragraph(f"Sous-total: {subtotal:.2f} $", ParagraphStyle('Subtotal', parent=styles['Normal'], fontSize=9, alignment=2)))
    elements.append(Paragraph("=" * 35, center_style))
    elements.append(Paragraph(f"<b>TOTAL: {total:.2f} $</b>", ParagraphStyle('Total', parent=styles['Normal'], fontSize=12, fontName='Helvetica-Bold', alignment=1)))
    elements.append(Paragraph("=" * 35, center_style))
    
    elements.append(Spacer(1, 5))
    
    # Who created the invoice
    if invoice.get("created_by_name"):
        elements.append(Paragraph(f"Facturé par: {invoice.get('created_by_name')}", small_style))
    
    elements.append(Spacer(1, 5))
    
    # Footer
    elements.append(Paragraph("Merci de votre visite !", center_style))
    elements.append(Paragraph("La Villa Delice", small_style))
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={invoice.get('invoice_number', 'facture')}.pdf"}
    )

@api_router.delete("/invoices/direct/{invoice_id}")
async def delete_direct_invoice(invoice_id: str, admin: User = Depends(get_admin_user)):
    inv = await db.direct_invoices.find_one({"id": invoice_id})
    result = await db.direct_invoices.delete_one({"id": invoice_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Invoice not found")
    num = inv.get("invoice_number", "Inconnu") if inv else "Inconnu"
    await log_action(admin, "Suppression facture", f"N°: {num}")
    return {"message": "Invoice deleted"}

# ==================== ROOM ROUTES ====================

@api_router.get("/rooms", response_model=List[Room])
async def get_rooms():
    rooms = await db.rooms.find({}, {"_id": 0}).to_list(1000)
    return [deserialize_datetime(r) for r in rooms]

@api_router.post("/rooms", response_model=Room)
async def create_room(room_data: RoomCreate, admin: User = Depends(get_admin_user)):
    room = Room(**room_data.model_dump())
    doc = room.model_dump()
    doc = serialize_datetime(doc)
    await db.rooms.insert_one(doc)
    return room

@api_router.put("/rooms/{room_id}", response_model=Room)
async def update_room(room_id: str, room_data: RoomCreate, admin: User = Depends(get_admin_user)):
    update_data = room_data.model_dump()
    result = await db.rooms.update_one({"id": room_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Room not found")
    updated = await db.rooms.find_one({"id": room_id}, {"_id": 0})
    await log_action(admin, "Modification chambre", f"Numéro: {room_data.number}")
    return Room(**deserialize_datetime(updated))

@api_router.delete("/rooms/{room_id}")
async def delete_room(room_id: str, admin: User = Depends(get_admin_user)):
    result = await db.rooms.delete_one({"id": room_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Room not found")
    await log_action(admin, "Suppression chambre", f"ID: {room_id}")
    return {"message": "Room deleted"}

# ==================== ROOM BOOKING ROUTES ====================

@api_router.get("/bookings", response_model=List[RoomBooking])
async def get_bookings(current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))):
    bookings = await db.bookings.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [deserialize_datetime(b) for b in bookings]

@api_router.post("/bookings", response_model=RoomBooking)
async def create_booking(booking_data: RoomBookingCreate, current_user: User = Depends(get_current_user)):
    # Get room info
    room = await db.rooms.find_one({"id": booking_data.room_id}, {"_id": 0})
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Calculate nights and total
    check_in_date = datetime.strptime(booking_data.check_in, "%Y-%m-%d")
    check_out_date = datetime.strptime(booking_data.check_out, "%Y-%m-%d")
    nights = (check_out_date - check_in_date).days
    
    if nights <= 0:
        raise HTTPException(status_code=400, detail="Check-out must be after check-in")
    
    total_price = nights * room.get("price_per_night", 0)
    
    booking = RoomBooking(
        room_id=booking_data.room_id,
        room_number=room.get("number", ""),
        room_type=room.get("type", ""),
        guest_name=booking_data.guest_name,
        guest_email=booking_data.guest_email,
        guest_phone=booking_data.guest_phone,
        check_in=booking_data.check_in,
        check_out=booking_data.check_out,
        nights=nights,
        price_per_night=room.get("price_per_night", 0),
        total_price=total_price,
        notes=booking_data.notes
    )
    
    doc = booking.model_dump()
    doc = serialize_datetime(doc)
    await db.bookings.insert_one(doc)
    
    # Update room status
    await db.rooms.update_one({"id": booking_data.room_id}, {"$set": {"status": "occupied"}})
    
    return booking

@api_router.put("/bookings/{booking_id}/status")
async def update_booking_status(booking_id: str, status: str, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))):
    valid_statuses = ["confirmed", "checked_in", "checked_out", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    await db.bookings.update_one({"id": booking_id}, {"$set": {"status": status}})
    
    # Update room status based on booking status
    if status in ["checked_out", "cancelled"]:
        await db.rooms.update_one({"id": booking["room_id"]}, {"$set": {"status": "available"}})
    elif status == "checked_in":
        await db.rooms.update_one({"id": booking["room_id"]}, {"$set": {"status": "occupied"}})
    
    return {"message": "Status updated"}

@api_router.put("/bookings/{booking_id}/extend")
async def extend_booking(booking_id: str, extension: BookingExtension, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))):
    """Extend a booking to a new checkout date"""
    booking = await db.bookings.find_one({"id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    check_in_date = datetime.strptime(booking["check_in"], "%Y-%m-%d")
    new_check_out_date = datetime.strptime(extension.check_out, "%Y-%m-%d")
    
    if new_check_out_date <= check_in_date:
        raise HTTPException(status_code=400, detail="New check-out date must be after check-in date")
        
    nights = (new_check_out_date - check_in_date).days
    total_price = nights * booking["price_per_night"]
    
    await db.bookings.update_one(
        {"id": booking_id},
        {"$set": {
            "check_out": extension.check_out,
            "nights": nights,
            "total_price": total_price
        }}
    )
    
    await log_action(current_user, "Prolongation réservation", f"Client: {booking.get('guest_name')}, Nouveau départ: {extension.check_out}")
    
    return {"message": "Booking extended", "nights": nights, "total_price": total_price}

@api_router.delete("/bookings/{booking_id}")
async def delete_booking(booking_id: str, admin: User = Depends(get_admin_user)):
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # Free up the room
    await db.rooms.update_one({"id": booking["room_id"]}, {"$set": {"status": "available"}})
    
    result = await db.bookings.delete_one({"id": booking_id})
    guest = booking.get("guest_name", "Inconnu") if booking else "Inconnu"
    await log_action(admin, "Suppression réservation", f"Client: {guest}")
    return {"message": "Booking deleted"}

# Add restaurant items to booking
@api_router.post("/bookings/{booking_id}/items")
async def add_item_to_booking(booking_id: str, item: AddItemToBooking, current_user: User = Depends(get_current_user)):
    """Add a restaurant item (food/drink) to a room booking bill"""
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # Get current items or initialize empty list
    current_items = booking.get("restaurant_items", [])
    
    # Check if item already exists, update quantity
    item_found = False
    for existing in current_items:
        if existing.get("menu_item_id") == item.menu_item_id:
            existing["quantity"] += item.quantity
            item_found = True
            break
    
    if not item_found:
        current_items.append(item.model_dump())
    
    # Calculate restaurant total
    restaurant_total = sum(i["price"] * i["quantity"] for i in current_items)
    
    await db.bookings.update_one(
        {"id": booking_id}, 
        {"$set": {"restaurant_items": current_items, "restaurant_total": restaurant_total}}
    )
    
    return {"message": "Item added", "restaurant_total": restaurant_total}

@api_router.delete("/bookings/{booking_id}/items/{menu_item_id}")
async def remove_item_from_booking(booking_id: str, menu_item_id: str, current_user: User = Depends(get_current_user)):
    """Remove a restaurant item from a room booking bill"""
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    current_items = booking.get("restaurant_items", [])
    current_items = [i for i in current_items if i.get("menu_item_id") != menu_item_id]
    
    # Recalculate restaurant total
    restaurant_total = sum(i["price"] * i["quantity"] for i in current_items)
    
    await db.bookings.update_one(
        {"id": booking_id}, 
        {"$set": {"restaurant_items": current_items, "restaurant_total": restaurant_total}}
    )
    
    return {"message": "Item removed", "restaurant_total": restaurant_total}

@api_router.get("/bookings/{booking_id}/items")
async def get_booking_items(booking_id: str):
    """Get all restaurant items for a booking"""
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    return {
        "items": booking.get("restaurant_items", []),
        "restaurant_total": booking.get("restaurant_total", 0)
    }

@api_router.get("/bookings/{booking_id}/invoice")
async def get_booking_invoice(booking_id: str):
    """Get invoice for a room booking including restaurant items"""
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    room_total = booking.get("total_price", 0)
    restaurant_items = booking.get("restaurant_items", [])
    restaurant_total = booking.get("restaurant_total", 0)
    subtotal = room_total + restaurant_total
    tax = subtotal * 0.1
    
    return {
        "invoice_number": f"HTL-{booking_id[:8].upper()}",
        "type": "room",
        "guest_name": booking.get("guest_name", ""),
        "guest_email": booking.get("guest_email", ""),
        "guest_phone": booking.get("guest_phone", ""),
        "room_number": booking.get("room_number", ""),
        "room_type": booking.get("room_type", ""),
        "check_in": booking.get("check_in", ""),
        "check_out": booking.get("check_out", ""),
        "nights": booking.get("nights", 0),
        "price_per_night": booking.get("price_per_night", 0),
        "room_total": room_total,
        "restaurant_items": restaurant_items,
        "restaurant_total": restaurant_total,
        "subtotal": subtotal,
        "tax": tax,
        "total": subtotal + tax,
        "status": booking.get("status", ""),
        "notes": booking.get("notes", ""),
        "created_at": booking.get("created_at", "")
    }

@api_router.get("/bookings/{booking_id}/invoice/pdf")
async def generate_booking_invoice_pdf(booking_id: str):
    """Generate PDF invoice for a room booking including restaurant items - Thermal format"""
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    buffer = BytesIO()
    # Format ticket thermique 80mm
    page_width = 226
    doc = SimpleDocTemplate(buffer, pagesize=(page_width, 900), rightMargin=10, leftMargin=10, topMargin=15, bottomMargin=15)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Styles ticket thermique
    title_style = ParagraphStyle('ThermalTitle', parent=styles['Normal'], fontSize=14, fontName='Helvetica-Bold', alignment=1, spaceAfter=5)
    center_style = ParagraphStyle('ThermalCenter', parent=styles['Normal'], fontSize=9, alignment=1, spaceAfter=3)
    small_style = ParagraphStyle('ThermalSmall', parent=styles['Normal'], fontSize=8, alignment=1, spaceAfter=2)
    section_style = ParagraphStyle('ThermalSection', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', alignment=1, spaceBefore=5, spaceAfter=3)
    
    # Header - La Villa Delice
    elements.append(Paragraph("LA VILLA DELICE", title_style))
    elements.append(Paragraph("Restaurant & Bar", center_style))
    elements.append(Paragraph("Q. Les volcans, Avenue Grevaillas", small_style))
    elements.append(Paragraph("Numero 076 Goma Nord Kivu", small_style))
    elements.append(Paragraph("Tél: 980629999", small_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Invoice info
    elements.append(Paragraph("FACTURE SÉJOUR", section_style))
    elements.append(Paragraph(f"N°: HTL-{booking_id[:8].upper()}", small_style))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", small_style))
    elements.append(Paragraph(f"Client: {booking.get('guest_name', '')}", small_style))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Room details
    elements.append(Paragraph("HÉBERGEMENT", section_style))
    room_type_labels = {"single": "Simple", "double": "Double", "suite": "Suite"}
    room_info = [
        [f"Chambre N° {booking.get('room_number', '')}", ""],
        [f"{room_type_labels.get(booking.get('room_type', ''), '')}", f"{booking.get('nights', 0)} nuit(s)"],
        [f"Du {booking.get('check_in', '')}", ""],
        [f"Au {booking.get('check_out', '')}", ""],
        [f"Tarif/nuit", f"{booking.get('price_per_night', 0):.2f} $"],
    ]
    room_table = Table(room_info, colWidths=[120, 70])
    room_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    elements.append(room_table)
    
    room_total = booking.get("total_price", 0)
    elements.append(Paragraph(f"Sous-total: {room_total:.2f} $", ParagraphStyle('SubTotal', parent=styles['Normal'], fontSize=9, alignment=2)))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Restaurant items
    restaurant_items = booking.get("restaurant_items", [])
    restaurant_total = booking.get("restaurant_total", 0)
    
    if restaurant_items:
        elements.append(Paragraph("RESTAURANT & BAR", section_style))
        
        resto_data = [["Qté", "Article", "Total"]]
        for item in restaurant_items:
            resto_data.append([
                str(item.get("quantity", 0)),
                item.get("name", "")[:15],
                f"{item.get('price', 0) * item.get('quantity', 0):.2f} $"
            ])
        
        resto_table = Table(resto_data, colWidths=[25, 115, 50])
        resto_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
        ]))
        elements.append(resto_table)
        elements.append(Paragraph(f"Sous-total: {restaurant_total:.2f} $", ParagraphStyle('SubTotal', parent=styles['Normal'], fontSize=9, alignment=2)))
        elements.append(Paragraph("-" * 35, center_style))
    
    # Grand Total
    subtotal = room_total + restaurant_total
    total = subtotal
    
    elements.append(Paragraph("=" * 35, center_style))
    total_data = [
        ["Hébergement", f"{room_total:.2f} $"],
        ["Restaurant", f"{restaurant_total:.2f} $"],
    ]
    total_table = Table(total_data, colWidths=[120, 70])
    total_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
    ]))
    elements.append(total_table)
    elements.append(Paragraph("=" * 35, center_style))
    elements.append(Paragraph(f"<b>TOTAL: {total:.2f} $</b>", ParagraphStyle('GrandTotal', parent=styles['Normal'], fontSize=12, fontName='Helvetica-Bold', alignment=1)))
    elements.append(Paragraph("=" * 35, center_style))
    
    # Footer
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Merci de votre séjour !", center_style))
    elements.append(Paragraph("La Villa Delice", small_style))
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=facture_sejour_{booking_id[:8]}.pdf"}
    )

# ==================== ORDER ROUTES ====================

@api_router.get("/orders", response_model=List[Order])
async def get_orders(current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR, UserRole.CUISINIER))):
    orders = await db.orders.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [deserialize_datetime(o) for o in orders]

@api_router.post("/orders", response_model=Order)
async def create_order(order_data: OrderCreate):
    total = sum(item.price * item.quantity for item in order_data.items)
    order = Order(
        items=order_data.items,
        total=total,
        customer_name=order_data.customer_name,
        room_number=order_data.room_number,
        table_number=order_data.table_number,
        notes=order_data.notes
    )
    doc = order.model_dump()
    doc['items'] = [item.model_dump() for item in order.items]
    doc = serialize_datetime(doc)
    await db.orders.insert_one(doc)
    
    # Send push notification for new order
    try:
        customer = order_data.customer_name or "Client"
        table = f" - Table {order_data.table_number}" if order_data.table_number else ""
        await send_push_notification(
            title="🔔 Nouvelle commande !",
            body=f"{customer}{table} - {total:.2f} $",
            url="/admin/orders"
        )
    except Exception as e:
        logger.error(f"Failed to send push notification: {e}")
    
    return order

@api_router.put("/orders/{order_id}/status")
async def update_order_status(order_id: str, update: OrderUpdate, current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.SERVEUR))):
    valid_statuses = ["pending", "preparing", "ready", "delivered", "cancelled"]
    if update.status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")
    result = await db.orders.update_one({"id": order_id}, {"$set": {"status": update.status}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Send push notification when order is ready
    if update.status == "ready":
        try:
            order = await db.orders.find_one({"id": order_id}, {"_id": 0})
            if order:
                customer = order.get("customer_name", "Client")
                table = f" - Table {order.get('table_number')}" if order.get('table_number') else ""
                await send_push_notification(
                    title="🍽️ Commande prête !",
                    body=f"{customer}{table} est prête à servir",
                    url="/admin/orders",
                    exclude_user_id=current_user.id  # Don't notify the cook who marked it ready
                )
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
    
    return {"message": "Status updated"}

@api_router.delete("/orders/{order_id}")
async def delete_order(order_id: str, admin: User = Depends(get_admin_user)):
    # Instead of hard delete, soft delete with who deleted
    order = await db.orders.find_one({"id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Log deletion
    deletion_log = {
        "id": str(uuid.uuid4()),
        "order_id": order_id,
        "order_data": order,
        "deleted_by": admin.email,
        "deleted_by_name": admin.name,
        "deleted_at": datetime.now(timezone.utc).isoformat()
    }
    await db.deletion_logs.insert_one(deletion_log)
    
    # Delete the order
    result = await db.orders.delete_one({"id": order_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": "Order deleted", "deleted_by": admin.name}

@api_router.get("/orders/deletions/history")
async def get_deletion_history(current_user: User = Depends(get_admin_user)):
    """Get history of all deleted orders (admin only)"""
    deletions = await db.deletion_logs.find({}, {"_id": 0}).sort("deleted_at", -1).to_list(100)
    return deletions

# ==================== PUSH NOTIFICATIONS ====================

async def send_push_notification(title: str, body: str, url: str = "/admin/orders", exclude_user_id: str = None):
    """Send push notification to all subscribed users"""
    query = {} if exclude_user_id is None else {"user_id": {"$ne": exclude_user_id}}
    subscriptions = await db.push_subscriptions.find(query, {"_id": 0}).to_list(1000)
    
    payload = json.dumps({
        "title": title,
        "body": body,
        "icon": "https://customer-assets.emergentagent.com/job_hotel-qr-menu-2/artifacts/99kuvteg_logo%201.png",
        "badge": "https://customer-assets.emergentagent.com/job_hotel-qr-menu-2/artifacts/99kuvteg_logo%201.png",
        "url": url,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    failed_subscriptions = []
    
    for sub in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": {
                        "p256dh": sub["keys"]["p256dh"],
                        "auth": sub["keys"]["auth"]
                    }
                },
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={
                    "sub": VAPID_EMAIL
                }
            )
            logger.info(f"Push sent to user {sub.get('user_id', 'unknown')}")
        except WebPushException as e:
            logger.error(f"Push failed: {e}")
            # If subscription is expired/invalid, mark for deletion
            if e.response and e.response.status_code in [404, 410]:
                failed_subscriptions.append(sub["id"])
    
    # Clean up invalid subscriptions
    if failed_subscriptions:
        await db.push_subscriptions.delete_many({"id": {"$in": failed_subscriptions}})
        logger.info(f"Cleaned up {len(failed_subscriptions)} invalid subscriptions")
    
    return len(subscriptions) - len(failed_subscriptions)

@api_router.get("/push/vapid-public-key")
async def get_vapid_public_key():
    """Get the VAPID public key for push subscription"""
    return {"publicKey": VAPID_PUBLIC_KEY}

@api_router.post("/push/subscribe")
async def subscribe_to_push(
    subscription: PushSubscriptionCreate,
    current_user: User = Depends(get_current_user)
):
    """Subscribe to push notifications"""
    # Check if this endpoint already exists for this user
    existing = await db.push_subscriptions.find_one({
        "user_id": current_user.id,
        "endpoint": subscription.endpoint
    })
    
    if existing:
        return {"message": "Already subscribed", "id": existing.get("id")}
    
    # Create new subscription
    sub = PushSubscription(
        user_id=current_user.id,
        endpoint=subscription.endpoint,
        keys=subscription.keys,
        user_agent=subscription.user_agent
    )
    
    doc = sub.model_dump()
    doc = serialize_datetime(doc)
    await db.push_subscriptions.insert_one(doc)
    
    logger.info(f"User {current_user.id} subscribed to push notifications")
    
    return {"message": "Subscribed successfully", "id": sub.id}

@api_router.delete("/push/unsubscribe")
async def unsubscribe_from_push(
    endpoint: str,
    current_user: User = Depends(get_current_user)
):
    """Unsubscribe from push notifications"""
    result = await db.push_subscriptions.delete_one({
        "user_id": current_user.id,
        "endpoint": endpoint
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    return {"message": "Unsubscribed successfully"}


@api_router.post("/push/unsubscribe/others")
async def unsubscribe_other_devices(req: UnsubscribeOthersRequest, current_user: User = Depends(get_current_user)):
    """Unsubscribe (delete) push subscriptions for the current user on all other devices.

    If `exclude_endpoint` is provided, that subscription is kept and all others are removed.
    This endpoint also sends a WebSocket 'mute' message to other connected devices and closes their socket.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    exclude = req.exclude_endpoint

    # Delete subscriptions in DB for this user except optionally the excluded endpoint
    if exclude:
        result = await db.push_subscriptions.delete_many({"user_id": current_user.id, "endpoint": {"$ne": exclude}})
    else:
        result = await db.push_subscriptions.delete_many({"user_id": current_user.id})

    deleted_count = result.deleted_count if hasattr(result, 'deleted_count') else 0

    # Notify other websocket connections for this user to mute sounds
    # Important: do NOT close or remove websocket connections here — clients should only stop playing sound.
    conns = user_ws.get(current_user.id, [])[:]
    notified = 0
    for entry in conns:
        try:
            ep = entry.get('endpoint')
            if exclude and ep == exclude:
                continue
            ws = entry.get('ws')
            # Send a small JSON command telling the client to mute/disable sounds
            try:
                await ws.send_text(json.dumps({"type": "mute", "reason": "muted_by_user"}))
            except Exception:
                # Best effort: ignore send failures (do not close connection)
                logger.debug('Failed to send mute message to websocket entry')
            notified += 1
        except Exception:
            logger.exception('Error notifying websocket')

    # Clean up empty mapping
    if current_user.id in user_ws and not user_ws[current_user.id]:
        del user_ws[current_user.id]

    return {"deleted": deleted_count, "notified": notified}


@api_router.post("/push/mute/others")
async def mute_other_devices(req: UnsubscribeOthersRequest, current_user: User = Depends(get_current_user)):
    """Send a 'mute' command to ALL connected devices (global mute)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    exclude = req.exclude_endpoint
    notified = 0
    
    # Broadcast to ALL connected users (Global Mute)
    all_connections = []
    for user_conns in user_ws.values():
        all_connections.extend(user_conns)

    for entry in all_connections:
        try:
            ep = entry.get('endpoint')
            if exclude and ep == exclude:
                continue
            ws = entry.get('ws')
            try:
                await ws.send_text(json.dumps({"type": "mute", "reason": "muted_by_user"}))
            except Exception:
                logger.debug('Failed to send mute message to websocket entry')
            notified += 1
        except Exception:
            logger.exception('Error notifying websocket')

    return {"notified": notified}

@api_router.post("/push/test")
async def test_push_notification(current_user: User = Depends(get_current_user)):
    """Send a test push notification to the current user"""
    subscriptions = await db.push_subscriptions.find(
        {"user_id": current_user.id}, {"_id": 0}
    ).to_list(10)
    
    if not subscriptions:
        raise HTTPException(status_code=404, detail="No push subscription found. Please enable notifications.")
    
    payload = json.dumps({
        "title": "🔔 Test Notification",
        "body": "Les notifications push fonctionnent correctement !",
        "icon": "https://customer-assets.emergentagent.com/job_hotel-qr-menu-2/artifacts/99kuvteg_logo%201.png",
        "url": "/admin/orders",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    success_count = 0
    for sub in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": {
                        "p256dh": sub["keys"]["p256dh"],
                        "auth": sub["keys"]["auth"]
                    }
                },
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={
                    "sub": VAPID_EMAIL
                }
            )
            success_count += 1
        except WebPushException as e:
            logger.error(f"Test push failed: {e}")
    
    return {"message": f"Test notification sent to {success_count} device(s)"}

# ==================== QR CODE ROUTES ====================

@api_router.get("/qrcode")
async def generate_qr_code(url: str):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")

@api_router.get("/qrcode/base64")
async def generate_qr_code_base64(url: str):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    base64_image = base64.b64encode(buffer.getvalue()).decode()
    return {"image": f"data:image/png;base64,{base64_image}"}

# ==================== REPORTS ROUTES ====================

@api_router.get("/reports/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    total_orders = await db.orders.count_documents({})
    pending_orders = await db.orders.count_documents({"status": "pending"})
    total_menu_items = await db.menu_items.count_documents({})
    total_rooms = await db.rooms.count_documents({})
    available_rooms = await db.rooms.count_documents({"status": "available"})
    
    # Calculate revenue
    pipeline = [
        {"$match": {"status": {"$in": ["delivered", "ready"]}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]
    revenue_result = await db.orders.aggregate(pipeline).to_list(1)
    total_revenue = revenue_result[0]["total"] if revenue_result else 0
    
    # Today's orders
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_orders = await db.orders.count_documents({
        "created_at": {"$gte": today.isoformat()}
    })
    
    return {
        "total_orders": total_orders,
        "pending_orders": pending_orders,
        "today_orders": today_orders,
        "total_menu_items": total_menu_items,
        "total_rooms": total_rooms,
        "available_rooms": available_rooms,
        "total_revenue": total_revenue
    }

@api_router.get("/reports/stock-exits")
async def get_stock_exits(
    page: int = 1,
    per_page: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.MAGAZINIER))
):
    """Get list of stock exits (losses)"""
    query = {"action": "Sortie stock"}
    
    if start_date or end_date:
        date_filter = {}
        if start_date:
            try:
                s = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                date_filter["$gte"] = s.isoformat()
            except ValueError: pass
        if end_date:
            try:
                e = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                date_filter["$lte"] = e.isoformat()
            except ValueError: pass
        if date_filter:
            query["created_at"] = date_filter

    total = await db.audit_logs.count_documents(query)
    skip = (page - 1) * per_page
    logs = await db.audit_logs.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(per_page).to_list(per_page)
    
    return {
        "logs": [deserialize_datetime(l) for l in logs],
        "total": total,
        "page": page,
        "per_page": per_page
    }

@api_router.get("/reports/stock-exits/pdf")
async def generate_stock_exits_pdf(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.STAFF, UserRole.MAGAZINIER))
):
    """Générer un rapport PDF des sorties de stock (Pertes, Casse, etc.)"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    query = {"action": "Sortie stock"}
    
    # Filtre par date
    if start_date or end_date:
        date_filter = {}
        if start_date:
            try:
                s = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                date_filter["$gte"] = s.isoformat()
            except ValueError: pass
        if end_date:
            try:
                e = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                date_filter["$lte"] = e.isoformat()
            except ValueError: pass
        if date_filter:
            query["created_at"] = date_filter

    logs = await db.audit_logs.find(query).sort("created_at", -1).to_list(1000)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1, spaceAfter=20)
    
    elements.append(Paragraph("Rapport des Pertes / Sorties de Stock", title_style))
    elements.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 20))

    data = [["Date", "Produit", "Qté", "Motif", "Auteur"]]
    
    for log in logs:
        # Formatage de la date
        created_at = log.get("created_at")
        date_str = ""
        if isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                date_str = dt.strftime("%d/%m/%Y %H:%M")
            except:
                date_str = created_at[:10]
        
        meta = log.get("metadata", {})
        product_name = meta.get("product_name", "Inconnu")
        qty = str(abs(meta.get("quantity_change", 0)))
        reason = meta.get("reason", "Autre")
        user = log.get("user_name", "Inconnu")
        
        data.append([date_str, product_name, qty, reason, user])

    table = Table(data, colWidths=[3.5*cm, 6*cm, 2*cm, 4*cm, 3.5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=rapport_pertes.pdf"}
    )

@api_router.get("/reports/daily")
async def get_daily_report(date: str = None, current_user: User = Depends(get_current_user)):
    """Get daily report for a specific date (format: YYYY-MM-DD)"""
    if date:
        target_date = datetime.fromisoformat(date)
    else:
        target_date = datetime.now(timezone.utc)
    
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Get orders for the day
    orders = await db.orders.find({
        "created_at": {
            "$gte": start_of_day.isoformat(),
            "$lte": end_of_day.isoformat()
        }
    }, {"_id": 0}).to_list(1000)
    
    total_orders = len(orders)
    total_revenue = sum(o.get("total", 0) for o in orders)
    
    # Count by status
    status_counts = {}
    for order in orders:
        status = order.get("status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Top selling items
    item_sales = {}
    for order in orders:
        for item in order.get("items", []):
            name = item.get("name", "Unknown")
            qty = item.get("quantity", 0)
            item_sales[name] = item_sales.get(name, 0) + qty
    
    top_items = sorted(item_sales.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "total_orders": total_orders,
        "total_revenue": total_revenue,
        "status_breakdown": status_counts,
        "top_selling_items": [{"name": name, "quantity": qty} for name, qty in top_items],
        "orders": orders
    }

@api_router.get("/reports/monthly")
async def get_monthly_report(year: int = None, month: int = None, current_user: User = Depends(get_current_user)):
    """Get monthly report for a specific month"""
    now = datetime.now(timezone.utc)
    year = year or now.year
    month = month or now.month
    
    start_of_month = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end_of_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end_of_month = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    
    # Get orders for the month
    orders = await db.orders.find({
        "created_at": {
            "$gte": start_of_month.isoformat(),
            "$lte": end_of_month.isoformat()
        }
    }, {"_id": 0}).to_list(10000)
    
    total_orders = len(orders)
    total_revenue = sum(o.get("total", 0) for o in orders)
    
    # Daily breakdown
    daily_breakdown = {}
    for order in orders:
        order_date = order.get("created_at", "")[:10]
        if order_date not in daily_breakdown:
            daily_breakdown[order_date] = {"orders": 0, "revenue": 0}
        daily_breakdown[order_date]["orders"] += 1
        daily_breakdown[order_date]["revenue"] += order.get("total", 0)
    
    # Status counts
    status_counts = {}
    for order in orders:
        status = order.get("status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Top selling items
    item_sales = {}
    for order in orders:
        for item in order.get("items", []):
            name = item.get("name", "Unknown")
            qty = item.get("quantity", 0)
            revenue = item.get("price", 0) * qty
            if name not in item_sales:
                item_sales[name] = {"quantity": 0, "revenue": 0}
            item_sales[name]["quantity"] += qty
            item_sales[name]["revenue"] += revenue
    
    top_items = sorted(item_sales.items(), key=lambda x: x[1]["revenue"], reverse=True)[:10]
    
    return {
        "year": year,
        "month": month,
        "period": f"{year}-{month:02d}",
        "total_orders": total_orders,
        "total_revenue": total_revenue,
        "average_order_value": total_revenue / total_orders if total_orders > 0 else 0,
        "status_breakdown": status_counts,
        "daily_breakdown": daily_breakdown,
        "top_selling_items": [{"name": name, **data} for name, data in top_items]
    }

@api_router.get("/invoices/{order_id}")
async def get_invoice(order_id: str):
    """Generate invoice for a specific order"""
    order = await db.orders.find_one({"id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return {
        "invoice_number": f"INV-{order_id[:8].upper()}",
        "date": order.get("created_at", ""),
        "customer_name": order.get("customer_name", "Client"),
        "room_number": order.get("room_number", ""),
        "table_number": order.get("table_number", ""),
        "items": order.get("items", []),
        "subtotal": order.get("total", 0),
        "tax": 0,
        "total": order.get("total", 0),
        "status": order.get("status", "pending"),
        "notes": order.get("notes", "")
    }

@api_router.get("/invoices/{order_id}/pdf")
async def generate_invoice_pdf(order_id: str):
    """Generate PDF invoice for a specific order - Thermal format"""
    order = await db.orders.find_one({"id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    buffer = BytesIO()
    # Format ticket thermique 80mm
    page_width = 226
    doc = SimpleDocTemplate(buffer, pagesize=(page_width, 600), rightMargin=10, leftMargin=10, topMargin=15, bottomMargin=15)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Styles ticket thermique
    title_style = ParagraphStyle('ThermalTitle', parent=styles['Normal'], fontSize=14, fontName='Helvetica-Bold', alignment=1, spaceAfter=5)
    center_style = ParagraphStyle('ThermalCenter', parent=styles['Normal'], fontSize=9, alignment=1, spaceAfter=3)
    small_style = ParagraphStyle('ThermalSmall', parent=styles['Normal'], fontSize=8, alignment=1, spaceAfter=2)
    section_style = ParagraphStyle('ThermalSection', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', alignment=1, spaceBefore=5, spaceAfter=3)
    
    # Header - La Villa Delice
    elements.append(Paragraph("LA VILLA DELICE", title_style))
    elements.append(Paragraph("Restaurant & Bar", center_style))
    elements.append(Paragraph("Q. Les volcans, Avenue Grevaillas", small_style))
    elements.append(Paragraph("Numero 076 Goma Nord Kivu", small_style))
    elements.append(Paragraph("Tél: 980629999", small_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Invoice info
    order_date = order.get("created_at", datetime.now(timezone.utc).isoformat())
    if isinstance(order_date, str):
        order_date = datetime.fromisoformat(order_date.replace('Z', '+00:00'))
    
    elements.append(Paragraph(f"Facture N°: INV-{order_id[:8].upper()}", small_style))
    elements.append(Paragraph(f"Date: {order_date.strftime('%d/%m/%Y %H:%M')}", small_style))
    if order.get("customer_name"):
        elements.append(Paragraph(f"Client: {order.get('customer_name')}", small_style))
    if order.get("table_number"):
        elements.append(Paragraph(f"Table: {order.get('table_number')}", small_style))
    if order.get("room_number"):
        elements.append(Paragraph(f"Chambre: {order.get('room_number')}", small_style))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Items header
    elements.append(Paragraph("Qté  Article              Total", ParagraphStyle('Header', parent=styles['Normal'], fontSize=8, fontName='Courier-Bold')))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Items
    for item in order.get("items", []):
        qty = str(item.get("quantity", 0)).ljust(3)
        name = item.get("name", "")[:18].ljust(18)
        total = f"{item.get('price', 0) * item.get('quantity', 0):.2f} $"
        elements.append(Paragraph(f"{qty} {name} {total}", ParagraphStyle('Item', parent=styles['Normal'], fontSize=8, fontName='Courier')))
    
    elements.append(Paragraph("-" * 35, center_style))
    
    # Total
    elements.append(Paragraph(f"Sous-total: {order.get('total', 0):.2f} $", ParagraphStyle('SubTotal', parent=styles['Normal'], fontSize=9, alignment=2)))
    elements.append(Paragraph("=" * 35, center_style))
    elements.append(Paragraph(f"<b>TOTAL: {order.get('total', 0):.2f} $</b>", ParagraphStyle('GrandTotal', parent=styles['Normal'], fontSize=12, fontName='Helvetica-Bold', alignment=1)))
    elements.append(Paragraph("=" * 35, center_style))
    
    # Footer
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Merci de votre visite !", center_style))
    elements.append(Paragraph("La Villa Delice", small_style))
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=facture_{order_id[:8]}.pdf"}
    )

@api_router.get("/reports/pdf")
async def generate_report_pdf(period: str = "all", date: str = None, current_user: User = Depends(get_current_user)):
    buffer = BytesIO()
    # Format ticket thermique 80mm (environ 226 points = 80mm)
    page_width = 226
    doc = SimpleDocTemplate(buffer, pagesize=(page_width, 800), rightMargin=10, leftMargin=10, topMargin=15, bottomMargin=15)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Styles ticket thermique
    title_style = ParagraphStyle(
        'ThermalTitle',
        parent=styles['Normal'],
        fontSize=14,
        fontName='Helvetica-Bold',
        alignment=1,
        spaceAfter=5
    )
    center_style = ParagraphStyle(
        'ThermalCenter',
        parent=styles['Normal'],
        fontSize=9,
        alignment=1,
        spaceAfter=3
    )
    small_style = ParagraphStyle(
        'ThermalSmall',
        parent=styles['Normal'],
        fontSize=8,
        alignment=1,
        spaceAfter=2
    )
    section_style = ParagraphStyle(
        'ThermalSection',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        alignment=1,
        spaceBefore=10,
        spaceAfter=5
    )
    
    # Header - La Villa Delice
    elements.append(Paragraph("LA VILLA DELICE", title_style))
    elements.append(Paragraph("Restaurant & Bar", center_style))
    elements.append(Paragraph("Q. Les volcans, Avenue Grevaillas", small_style))
    elements.append(Paragraph("Numero 076 Goma Nord Kivu", small_style))
    elements.append(Paragraph("Tél: 980629999", small_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Title
    elements.append(Paragraph("RAPPORT", section_style))
    elements.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", small_style))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Stats
    stats = await get_stats(current_user)
    elements.append(Paragraph("STATISTIQUES", section_style))
    
    stats_data = [
        ["Commandes totales", str(stats["total_orders"])],
        ["Commandes aujourd'hui", str(stats["today_orders"])],
        ["En attente", str(stats["pending_orders"])],
        ["Articles menu", str(stats["total_menu_items"])],
        ["Chambres totales", str(stats["total_rooms"])],
        ["Chambres dispo", str(stats["available_rooms"])],
        ["Revenu total", f"{stats['total_revenue']:.2f} $"]
    ]
    
    stats_table = Table(stats_data, colWidths=[100, 80])
    stats_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("-" * 35, center_style))
    
    # Recent orders
    elements.append(Paragraph("COMMANDES RÉCENTES", section_style))
    
    orders = await db.orders.find().sort("created_at", -1).limit(10).to_list(10)
    
    if orders:
        for order in orders:
            order_date = order.get('created_at', datetime.now())
            if isinstance(order_date, str):
                order_date = datetime.fromisoformat(order_date.replace('Z', '+00:00'))
            date_str = order_date.strftime('%d/%m %H:%M')
            total = f"{order.get('total', 0):.2f} $"
            status = order.get('status', 'pending')[:3].upper()
            
            elements.append(Paragraph(f"{date_str} | {status} | {total}", small_style))
    else:
        elements.append(Paragraph("Aucune commande", small_style))
    
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("=" * 35, center_style))
    
    # Footer
    elements.append(Paragraph("Merci !", center_style))
    elements.append(Paragraph("La Villa Delice", small_style))
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=rapport_villa_delice.pdf"}
    )

# ==================== MESSAGING ROUTES ====================

@api_router.get("/messages/users")
async def get_chat_users(current_user: User = Depends(get_current_user)):
    """Get list of users with their last message for the chat sidebar"""
    # Get all users (excluding password)
    users = await db.users.find({}, {"_id": 0, "hashed_password": 0}).to_list(1000)
    
    # Aggregation to get last message for each conversation involving current_user
    pipeline = [
        {
            "$match": {
                "$or": [{"sender_id": current_user.id}, {"receiver_id": current_user.id}]
            }
        },
        {
            "$sort": {"created_at": -1}
        },
        {
            "$group": {
                "_id": {
                    "$cond": [
                        {"$eq": ["$sender_id", current_user.id]},
                        "$receiver_id",
                        "$sender_id"
                    ]
                },
                "last_message": {"$first": "$$ROOT"}
            }
        }
    ]
    
    conversations = await db.messages.aggregate(pipeline).to_list(1000)
    conv_map = {c["_id"]: c["last_message"] for c in conversations}

    # Aggregation to get unread counts from each user
    unread_pipeline = [
        {
            "$match": {"receiver_id": current_user.id, "read": False}
        },
        {
            "$group": {
                "_id": "$sender_id",
                "count": {"$sum": 1}
            }
        }
    ]
    unread_counts = await db.messages.aggregate(unread_pipeline).to_list(1000)
    unread_map = {c["_id"]: c["count"] for c in unread_counts}
    
    result = []
    for u in users:
        if u["id"] == current_user.id:
            continue
        
        u_dict = deserialize_datetime(u)
        last_msg = conv_map.get(u["id"])
        
        if last_msg:
            u_dict["last_message"] = last_msg.get("content")
            u_dict["last_message_time"] = last_msg.get("created_at")
        else:
            u_dict["last_message"] = ""
            u_dict["last_message_time"] = ""
        
        u_dict["unread_count"] = unread_map.get(u["id"], 0)
            
        result.append(u_dict)
        
    # Sort users: those with recent messages first, then by name
    result.sort(key=lambda x: (x.get("last_message_time") or "", x.get("name")), reverse=True)
    return result

@api_router.get("/messages/{other_user_id}", response_model=List[Message])
async def get_messages(other_user_id: str, current_user: User = Depends(get_current_user)):
    """Get conversation history with a specific user"""
    messages = await db.messages.find({
        "$or": [
            {"sender_id": current_user.id, "receiver_id": other_user_id},
            {"sender_id": other_user_id, "receiver_id": current_user.id}
        ]
    }, {"_id": 0}).sort("created_at", 1).to_list(1000)
    return [deserialize_datetime(m) for m in messages]

@api_router.post("/messages", response_model=Message)
async def send_message(msg_data: MessageCreate, current_user: User = Depends(get_current_user)):
    """Send a message to another user"""
    msg = Message(
        sender_id=current_user.id,
        receiver_id=msg_data.receiver_id,
        content=msg_data.content
    )
    doc = msg.model_dump()
    doc = serialize_datetime(doc)
    await db.messages.insert_one(doc)
    
    # Real-time notification via WebSocket if receiver is connected
    receiver_conns = user_ws.get(msg_data.receiver_id, [])
    for entry in receiver_conns:
        try:
            ws = entry.get("ws")
            # Send simple event payload
            message_payload = deserialize_datetime(doc)
            message_payload['sender_name'] = current_user.name # Add sender name here
            await ws.send_text(json.dumps({
                "type": "new_message",
                "message": message_payload
            }))
        except Exception:
            pass
            
    return deserialize_datetime(doc)

@api_router.post("/messages/mark-as-read")
async def mark_messages_as_read(payload: Dict[str, str], current_user: User = Depends(get_current_user)):
    """Mark messages from a specific sender as read."""
    other_user_id = payload.get("other_user_id")
    if not other_user_id:
        raise HTTPException(status_code=400, detail="other_user_id is required")

    await db.messages.update_many(
        {"receiver_id": current_user.id, "sender_id": other_user_id, "read": False},
        {"$set": {"read": True}}
    )
    # Return the new unread count
    return await get_unread_message_count(current_user)

@api_router.get("/messages/unread-count")
async def get_unread_message_count(current_user: User = Depends(get_current_user)):
    """Get the count of unread messages for the current user."""
    count = await db.messages.count_documents({
        "receiver_id": current_user.id,
        "read": False
    })
    return {"count": count}

# ==================== MESSAGING ROUTES ====================

@api_router.get("/messages/users")
async def get_chat_users(current_user: User = Depends(get_current_user)):
    """Get list of users with their last message for the chat sidebar"""
    # Get all users (excluding password)
    users = await db.users.find({}, {"_id": 0, "hashed_password": 0}).to_list(1000)
    
    # Aggregation to get last message for each conversation involving current_user
    pipeline = [
        {
            "$match": {
                "$or": [{"sender_id": current_user.id}, {"receiver_id": current_user.id}]
            }
        },
        {
            "$sort": {"created_at": -1}
        },
        {
            "$group": {
                "_id": {
                    "$cond": [
                        {"$eq": ["$sender_id", current_user.id]},
                        "$receiver_id",
                        "$sender_id"
                    ]
                },
                "last_message": {"$first": "$$ROOT"}
            }
        }
    ]
    
    conversations = await db.messages.aggregate(pipeline).to_list(1000)
    conv_map = {c["_id"]: c["last_message"] for c in conversations}
    
    result = []
    for u in users:
        if u["id"] == current_user.id:
            continue
        
        u_dict = deserialize_datetime(u)
        last_msg = conv_map.get(u["id"])
        
        if last_msg:
            u_dict["last_message"] = last_msg.get("content")
            u_dict["last_message_time"] = last_msg.get("created_at")
        else:
            u_dict["last_message"] = ""
            u_dict["last_message_time"] = ""
            
        result.append(u_dict)
        
    # Sort users: those with recent messages first, then by name
    result.sort(key=lambda x: (x.get("last_message_time") or "", x.get("name")), reverse=True)
    return result

@api_router.get("/messages/{other_user_id}", response_model=List[Message])
async def get_messages(other_user_id: str, current_user: User = Depends(get_current_user)):
    """Get conversation history with a specific user"""
    messages = await db.messages.find({
        "$or": [
            {"sender_id": current_user.id, "receiver_id": other_user_id},
            {"sender_id": other_user_id, "receiver_id": current_user.id}
        ]
    }, {"_id": 0}).sort("created_at", 1).to_list(1000)
    return [deserialize_datetime(m) for m in messages]

@api_router.post("/messages", response_model=Message)
async def send_message(msg_data: MessageCreate, current_user: User = Depends(get_current_user)):
    """Send a message to another user"""
    msg = Message(
        sender_id=current_user.id,
        receiver_id=msg_data.receiver_id,
        content=msg_data.content
    )
    doc = msg.model_dump()
    doc = serialize_datetime(doc)
    await db.messages.insert_one(doc)
    
    # Real-time notification via WebSocket if receiver is connected
    receiver_conns = user_ws.get(msg_data.receiver_id, [])
    for entry in receiver_conns:
        try:
            ws = entry.get("ws")
            # Send simple event payload
            await ws.send_text(json.dumps({
                "type": "new_message",
                "message": deserialize_datetime(doc)
            }))
        except Exception:
            pass
            
    return deserialize_datetime(doc)

@api_router.get("/messages/unread-count")
async def get_unread_message_count(current_user: User = Depends(get_current_user)):
    """Get the count of unread messages for the current user."""
    count = await db.messages.count_documents({
        "receiver_id": current_user.id,
        "read": False
    })
    return {"count": count}

# ==================== AUDIT LOGS ROUTES ====================

@api_router.get("/audit-logs")
async def get_audit_logs(
    page: int = 1,
    per_page: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_admin_user)
):
    """Get audit logs (admin only)."""
    query = {}
    if start_date or end_date:
        date_filter = {}
        if start_date:
            try:
                s = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                date_filter["$gte"] = s.isoformat()
            except ValueError:
                pass
        if end_date:
            try:
                e = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                date_filter["$lte"] = e.isoformat()
            except ValueError:
                pass
        if date_filter:
            query["created_at"] = date_filter

    total = await db.audit_logs.count_documents(query)
    skip = (page - 1) * per_page
    
    logs = await db.audit_logs.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(per_page).to_list(per_page)
    return {
        "logs": [deserialize_datetime(l) for l in logs],
        "total": total,
        "page": page,
        "per_page": per_page
    }

# ==================== SEED DATA ====================

@api_router.post("/seed")
async def seed_data():
    """Seed initial data for demo purposes. This is now idempotent for key users and data."""

    # --- Upsert admin user ---
    admin_email = "admin@hotel.com"
    admin_user = await db.users.find_one({"email": admin_email})
    if not admin_user:
        admin = UserInDB(
            id=str(uuid.uuid4()),
            email=admin_email,
            name="Administrateur",
            role=UserRole.ADMIN,
            hashed_password=get_password_hash("admin123")
        )
        admin_doc = admin.model_dump()
        admin_doc = serialize_datetime(admin_doc)
        await db.users.insert_one(admin_doc)

    # --- Force Reset 'sept' user to ensure password is correct ---
    sept_email = "sept@gmail.com"
    
    # Check if user exists (case insensitive)
    existing_sept = await db.users.find_one({"email": {"$regex": f"^{sept_email}$", "$options": "i"}})
    
    if existing_sept:
        # Update existing user password and role
        await db.users.update_one(
            {"_id": existing_sept["_id"]},
            {"$set": {"hashed_password": get_password_hash("123456"), "role": UserRole.SERVEUR, "name": "Bertrand Sept"}}
        )
        logger.info(f"User {sept_email} updated with reset password '123456'")
    else:
        # Create fresh user
        sept = UserInDB(
            id=str(uuid.uuid4()),
            email=sept_email,
            name="Bertrand Sept",
            role=UserRole.SERVEUR,
            hashed_password=get_password_hash("123456")
        )
        sept_doc = sept.model_dump()
        sept_doc = serialize_datetime(sept_doc)
        await db.users.insert_one(sept_doc)
        logger.info(f"User {sept_email} created with password '123456'")

    # --- Seed other data only if it's missing ---
    existing_categories = await db.categories.count_documents({})
    if existing_categories == 0:
        # Create categories
        categories_data = [
            {"name": "Entrées", "type": "food", "description": "Nos délicieuses entrées"},
            {"name": "Plats Principaux", "type": "food", "description": "Nos plats signatures"},
            {"name": "Desserts", "type": "food", "description": "Finissez en beauté"},
            {"name": "Cocktails", "type": "drink", "description": "Nos créations originales"},
            {"name": "Vins", "type": "drink", "description": "Sélection de vins fins"},
            {"name": "Boissons Chaudes", "type": "drink", "description": "Café, thé et plus"},
        ]
        
        category_ids = {}
        for cat_data in categories_data:
            cat = Category(**cat_data)
            doc = cat.model_dump()
            doc = serialize_datetime(doc)
            await db.categories.insert_one(doc)
            category_ids[cat_data["name"]] = cat.id
        
        # Create menu items
        menu_items_data = [
            {"name": "Soupe du Jour", "description": "Préparation fraîche quotidienne", "price": 8.50, "category_id": category_ids["Entrées"], "image_url": "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400"},
            {"name": "Salade César", "description": "Laitue romaine, parmesan, croûtons", "price": 12.00, "category_id": category_ids["Entrées"], "image_url": "https://images.unsplash.com/photo-1546793665-c74683f339c1?w=400"},
            {"name": "Steak Frites", "description": "Entrecôte grillée, frites maison", "price": 28.00, "category_id": category_ids["Plats Principaux"], "image_url": "https://images.unsplash.com/photo-1600891964092-4316c288032e?w=400"},
            {"name": "Saumon Grillé", "description": "Filet de saumon, légumes de saison", "price": 25.00, "category_id": category_ids["Plats Principaux"], "image_url": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400"},
            {"name": "Risotto aux Champignons", "description": "Riz arborio, champignons variés", "price": 22.00, "category_id": category_ids["Plats Principaux"], "image_url": "https://images.unsplash.com/photo-1476124369491-e7addf5db371?w=400"},
            {"name": "Tiramisu", "description": "Recette italienne traditionnelle", "price": 9.00, "category_id": category_ids["Desserts"], "image_url": "https://images.unsplash.com/photo-1571877227200-a0d98ea607e9?w=400"},
            {"name": "Crème Brûlée", "description": "Vanille de Madagascar", "price": 8.50, "category_id": category_ids["Desserts"], "image_url": "https://images.unsplash.com/photo-1470124182917-cc6e71b22ecc?w=400"},
            {"name": "Mojito", "description": "Rhum, menthe fraîche, citron vert", "price": 12.00, "category_id": category_ids["Cocktails"], "image_url": "https://images.unsplash.com/photo-1551538827-9c037cb4f32a?w=400"},
            {"name": "Martini Espresso", "description": "Vodka, café, liqueur de café", "price": 14.00, "category_id": category_ids["Cocktails"], "image_url": "https://images.unsplash.com/photo-1545438102-799c3991ffef?w=400"},
            {"name": "Bordeaux Rouge", "description": "Saint-Émilion Grand Cru", "price": 45.00, "category_id": category_ids["Vins"], "image_url": "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=400"},
            {"name": "Café Expresso", "description": "Torréfaction artisanale", "price": 3.50, "category_id": category_ids["Boissons Chaudes"], "image_url": "https://images.unsplash.com/photo-1510707577719-ae7c14805e3a?w=400"},
        ]
        
        for item_data in menu_items_data:
            item = MenuItem(**item_data)
            doc = item.model_dump()
            doc = serialize_datetime(doc)
            await db.menu_items.insert_one(doc)
        
        # Create rooms
        rooms_data = [
            {"number": "101", "type": "single", "price_per_night": 120.00, "description": "Chambre simple avec vue jardin", "image_url": "https://images.unsplash.com/photo-1631049307264-da0ec9d70304?w=400"},
            {"number": "102", "type": "double", "price_per_night": 180.00, "description": "Chambre double standard", "image_url": "https://images.unsplash.com/photo-1590490360182-c33d57733427?w=400"},
            {"number": "201", "type": "suite", "price_per_night": 350.00, "description": "Suite de luxe avec terrasse", "image_url": "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=400"},
            {"number": "202", "type": "double", "price_per_night": 200.00, "description": "Chambre double vue mer", "image_url": "https://images.unsplash.com/photo-1566665797739-1674de7a421a?w=400"},
        ]
        
        for room_data in rooms_data:
            room = Room(**room_data)
            doc = room.model_dump()
            doc = serialize_datetime(doc)
            await db.rooms.insert_one(doc)

    return {"message": "Data seeding check complete. Core users are present and passwords are set."}

# Root endpoint
@api_router.get("/")
async def root():
    return {"message": "Hotel Restaurant API", "version": "1.0.0"}

# Include the router in the main app
app.include_router(api_router)

# Get origins from env var. Default to an empty string.
raw_origins = os.environ.get('CORS_ORIGINS', '')
allowed_origins = [o.strip() for o in raw_origins.split(',') if o.strip()]

# For local development, always allow localhost:3000
if "http://localhost:3000" not in allowed_origins:
    allowed_origins.append("http://localhost:3000")

# Add production frontend domain
if "https://lavilladelice.netlify.app" not in allowed_origins:
    allowed_origins.append("https://lavilladelice.netlify.app")

# Log the actual origins being used for debugging
logger.info(f"Final CORS Allowed Origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Simple WebSocket endpoint that enforces Origin checks based on CORS_ORIGINS.
    Clients may connect without auth for development; adjust token validation as needed.
    """
    origin = websocket.headers.get('origin')
    # If ALLOWED_ORIGINS does not contain '*' and origin not in list, reject
    if origin and '*' not in ALLOWED_ORIGINS and origin not in ALLOWED_ORIGINS:
        logger.warning(f"WebSocket connection rejected from origin: {origin}")
        await websocket.close(code=1008)
        return
    # Try to authenticate the socket using Authorization header (Bearer <token>)
    user_id = None
    try:
        auth_header = websocket.headers.get('authorization')
        if auth_header and auth_header.lower().startswith('bearer '):
            token = auth_header.split(' ', 1)[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get('sub')
    except Exception:
        user_id = None

    # Fallback: allow token via query param for browser-based WS clients
    if not user_id:
        try:
            token = websocket.query_params.get('token')
            if token:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user_id = payload.get('sub')
        except Exception:
            user_id = None

    await websocket.accept()
    logger.info(f"WebSocket connection accepted from origin: {origin} user_id={user_id}")

    # Register this websocket for the user (optional endpoint query param)
    try:
        endpoint = websocket.query_params.get('endpoint')
        if user_id:
            entry = {"ws": websocket, "endpoint": endpoint}
            user_ws.setdefault(user_id, []).append(entry)
    except Exception:
        logger.exception('Failed to register websocket')
    try:
        while True:
            msg = await websocket.receive_text()
            # Echo or handle messages as needed; keep simple ping/pong for now
            if msg.lower() in ("ping", "ping\n"):
                await websocket.send_text("pong")
            else:
                await websocket.send_text(f"received: {msg}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        # Remove from user_ws mappings if present
        try:
            if user_id and user_id in user_ws:
                # Remove any entries referencing this websocket object
                user_ws[user_id] = [e for e in user_ws[user_id] if e.get('ws') is not websocket]
                if not user_ws[user_id]:
                    del user_ws[user_id]
        except Exception:
            logger.exception('Failed to cleanup websocket mapping')

@app.on_event("startup")
async def startup_event():
    if DB_AVAILABLE:
        try:
            await seed_data()
            logger.info("Automatic startup seeding executed successfully.")
        except Exception as e:
            logger.error(f"Error during startup seeding: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
