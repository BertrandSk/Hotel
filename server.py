from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
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
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
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

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

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

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class UserRole:
    ADMIN = "admin"
    STAFF = "staff"
    GUEST = "guest"
    SERVEUR = "serveur"
    CUISINIER = "cuisinier"

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

class AddItemToBooking(BaseModel):
    menu_item_id: str
    name: str
    price: float
    quantity: int

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
    user_doc = await db.users.find_one({"email": login_data.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(login_data.password, user_doc.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_doc = deserialize_datetime(user_doc)
    user = User(**user_doc)
    access_token = create_access_token(data={"sub": user.id})
    return Token(access_token=access_token, token_type="bearer", user=user)

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ==================== USER MANAGEMENT (ADMIN ONLY) ====================

@api_router.get("/users", response_model=List[User])
async def get_users(admin: User = Depends(get_admin_user)):
    users = await db.users.find({}, {"_id": 0, "hashed_password": 0}).to_list(1000)
    return [deserialize_datetime(u) for u in users]

@api_router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: User = Depends(get_admin_user)):
    result = await db.users.delete_one({"id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted"}

@api_router.put("/users/{user_id}/role")
async def update_user_role(user_id: str, role: str, admin: User = Depends(get_admin_user)):
    if role not in [UserRole.ADMIN, UserRole.STAFF, UserRole.GUEST]:
        raise HTTPException(status_code=400, detail="Invalid role")
    result = await db.users.update_one({"id": user_id}, {"$set": {"role": role}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
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
async def create_menu_item(item_data: MenuItemCreate, admin: User = Depends(get_admin_user)):
    item = MenuItem(**item_data.model_dump())
    doc = item.model_dump()
    doc = serialize_datetime(doc)
    await db.menu_items.insert_one(doc)
    return item

@api_router.put("/menu/{item_id}", response_model=MenuItem)
async def update_menu_item(item_id: str, item_data: MenuItemCreate, admin: User = Depends(get_admin_user)):
    update_data = item_data.model_dump()
    result = await db.menu_items.update_one({"id": item_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Menu item not found")
    updated = await db.menu_items.find_one({"id": item_id}, {"_id": 0})
    return MenuItem(**deserialize_datetime(updated))

@api_router.delete("/menu/{item_id}")
async def delete_menu_item(item_id: str, admin: User = Depends(get_admin_user)):
    result = await db.menu_items.delete_one({"id": item_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Menu item not found")
    return {"message": "Menu item deleted"}

# Stock Management
@api_router.put("/menu/{item_id}/stock")
async def update_stock(item_id: str, quantity: int, admin: User = Depends(get_admin_user)):
    """Update stock quantity for a menu item"""
    result = await db.menu_items.update_one(
        {"id": item_id}, 
        {"$set": {"stock_quantity": quantity}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Menu item not found")
    return {"message": "Stock updated", "quantity": quantity}

@api_router.post("/menu/{item_id}/stock/add")
async def add_stock(item_id: str, quantity: int, admin: User = Depends(get_admin_user)):
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
    return {"message": "Stock added", "new_quantity": new_stock}

# ==================== DIRECT INVOICES ====================

@api_router.get("/invoices")
async def get_all_invoices(current_user: User = Depends(get_current_user)):
    """Get all direct invoices"""
    invoices = await db.direct_invoices.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [deserialize_datetime(i) for i in invoices]

@api_router.post("/invoices/create")
async def create_direct_invoice(invoice_data: DirectInvoiceCreate, current_user: User = Depends(get_current_user)):
    """Create a direct invoice (for restaurant billing)"""
    # Calculate totals
    subtotal = sum(item.price * item.quantity for item in invoice_data.items)
    total = subtotal  # Sans TVA
    
    # Generate invoice number
    count = await db.direct_invoices.count_documents({})
    invoice_number = f"FACT-{count + 1:06d}"
    
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
    
    # Deduct stock for tracked items
    for item in invoice_data.items:
        menu_item = await db.menu_items.find_one({"id": item.menu_item_id}, {"_id": 0})
        if menu_item and menu_item.get("track_stock"):
            current_stock = menu_item.get("stock_quantity", 0) or 0
            new_stock = max(0, current_stock - item.quantity)
            await db.menu_items.update_one(
                {"id": item.menu_item_id},
                {"$set": {"stock_quantity": new_stock}}
            )
    
    doc = invoice.model_dump()
    doc['items'] = [item.model_dump() for item in invoice.items]
    doc = serialize_datetime(doc)
    await db.direct_invoices.insert_one(doc)
    
    return invoice

@api_router.get("/invoices/direct/{invoice_id}")
async def get_direct_invoice(invoice_id: str):
    """Get a direct invoice by ID"""
    invoice = await db.direct_invoices.find_one({"id": invoice_id}, {"_id": 0})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return deserialize_datetime(invoice)

@api_router.get("/invoices/direct/{invoice_id}/pdf")
async def generate_direct_invoice_pdf(invoice_id: str):
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
    result = await db.direct_invoices.delete_one({"id": invoice_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Invoice not found")
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
    return Room(**deserialize_datetime(updated))

@api_router.delete("/rooms/{room_id}")
async def delete_room(room_id: str, admin: User = Depends(get_admin_user)):
    result = await db.rooms.delete_one({"id": room_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"message": "Room deleted"}

# ==================== ROOM BOOKING ROUTES ====================

@api_router.get("/bookings", response_model=List[RoomBooking])
async def get_bookings(current_user: User = Depends(get_current_user)):
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
async def update_booking_status(booking_id: str, status: str, current_user: User = Depends(get_current_user)):
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

@api_router.delete("/bookings/{booking_id}")
async def delete_booking(booking_id: str, admin: User = Depends(get_admin_user)):
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # Free up the room
    await db.rooms.update_one({"id": booking["room_id"]}, {"$set": {"status": "available"}})
    
    result = await db.bookings.delete_one({"id": booking_id})
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
async def get_orders(current_user: User = Depends(get_current_user)):
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
            body=f"{customer}{table} - {total:.2f} €",
            url="/admin/orders"
        )
    except Exception as e:
        logger.error(f"Failed to send push notification: {e}")
    
    return order

@api_router.put("/orders/{order_id}/status")
async def update_order_status(order_id: str, update: OrderUpdate, current_user: User = Depends(get_current_user)):
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
async def delete_order(order_id: str, current_user: User = Depends(get_current_user)):
    # Instead of hard delete, soft delete with who deleted
    order = await db.orders.find_one({"id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Log deletion
    deletion_log = {
        "id": str(uuid.uuid4()),
        "order_id": order_id,
        "order_data": order,
        "deleted_by": current_user.email,
        "deleted_by_name": current_user.name,
        "deleted_at": datetime.now(timezone.utc).isoformat()
    }
    await db.deletion_logs.insert_one(deletion_log)
    
    # Delete the order
    result = await db.orders.delete_one({"id": order_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": "Order deleted", "deleted_by": current_user.name}

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

# ==================== SEED DATA ====================

@api_router.post("/seed")
async def seed_data():
    """Seed initial data for demo purposes"""
    # Check if data already exists
    existing_users = await db.users.count_documents({})
    if existing_users > 0:
        return {"message": "Data already seeded"}
    
    # Create admin user
    admin = UserInDB(
        id=str(uuid.uuid4()),
        email="admin@hotel.com",
        name="Administrateur",
        role=UserRole.ADMIN,
        hashed_password=get_password_hash("admin123")
    )
    admin_doc = admin.model_dump()
    admin_doc = serialize_datetime(admin_doc)
    await db.users.insert_one(admin_doc)
    
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
    
    return {"message": "Data seeded successfully", "admin_email": "admin@hotel.com", "admin_password": "admin123"}

# Root endpoint
@api_router.get("/")
async def root():
    return {"message": "Hotel Restaurant API", "version": "1.0.0"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
