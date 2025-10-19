from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
from polygon import RESTClient
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Polygon client
polygon_client = RESTClient(os.environ['POLYGON_API_KEY'])

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class Stock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    name: str
    current_price: Optional[float] = None
    change_percent: Optional[float] = None

class StockAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    candlestick_pattern: Optional[str] = None
    ai_analysis: Optional[str] = None
    potential_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PortfolioStock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    name: str
    shares: float
    purchase_price: float
    purchase_date: datetime
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PortfolioStockCreate(BaseModel):
    symbol: str
    name: str
    shares: float
    purchase_price: float
    purchase_date: datetime

# Helper functions
def calculate_ema(prices: List[float], period: int) -> float:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return round(ema, 2)

def analyze_candlestick(candles: List[Dict]) -> str:
    """Analyze candlestick patterns"""
    if len(candles) < 3:
        return "Insufficient data"
    
    last = candles[-1]
    prev = candles[-2]
    
    open_price = last['o']
    close = last['c']
    high = last['h']
    low = last['l']
    
    body = abs(close - open_price)
    range_size = high - low
    
    patterns = []
    
    # Bullish patterns
    if close > open_price:
        if body / range_size > 0.7:
            patterns.append("Vela Alcista Fuerte")
        elif (high - close) / range_size < 0.1:
            patterns.append("Marubozu Alcista")
    
    # Bearish patterns
    elif open_price > close:
        if body / range_size > 0.7:
            patterns.append("Vela Bajista Fuerte")
        elif (close - low) / range_size < 0.1:
            patterns.append("Marubozu Bajista")
    
    # Doji
    if body / range_size < 0.1:
        patterns.append("Doji - Indecisión")
    
    # Hammer
    if close > open_price and (close - low) > 2 * body and (high - close) < body:
        patterns.append("Martillo Alcista")
    
    return ", ".join(patterns) if patterns else "Patrón Neutral"

async def get_ai_analysis(symbol: str, analysis_data: Dict) -> str:
    """Get AI-powered analysis using OpenAI"""
    try:
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"stock_{symbol}_{datetime.now().timestamp()}",
            system_message="Eres un analista financiero experto. Proporciona análisis concisos y profesionales."
        ).with_model("openai", "gpt-4o-mini")
        
        prompt = f"""
Analiza la siguiente acción: {symbol}

Datos técnicos:
- Beta: {analysis_data.get('beta', 'N/A')}
- Dividend Yield: {analysis_data.get('dividend_yield', 'N/A')}%
- EMA 20: ${analysis_data.get('ema_20', 'N/A')}
- EMA 50: ${analysis_data.get('ema_50', 'N/A')}
- Patrón de velas: {analysis_data.get('candlestick_pattern', 'N/A')}
- Precio actual: ${analysis_data.get('current_price', 'N/A')}

Proporciona un análisis breve (máximo 150 palabras) sobre:
1. Potencial de inversión
2. Riesgos principales
3. Recomendación (Comprar/Mantener/Vender)
"""
        
        message = UserMessage(text=prompt)
        response = await chat.send_message(message)
        return response
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return "Análisis IA no disponible temporalmente"

# Routes
@api_router.get("/")
async def root():
    return {"message": "Stock Analyzer API"}

@api_router.get("/stocks/search")
async def search_stocks(query: str):
    """Search for stocks by symbol or name"""
    try:
        # Search for ticker
        results = polygon_client.get_ticker_details(query.upper())
        
        stocks = []
        if results:
            # Get current price
            try:
                aggs = list(polygon_client.get_aggs(
                    ticker=results.ticker,
                    multiplier=1,
                    timespan="day",
                    from_=(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                    to=datetime.now().strftime('%Y-%m-%d')
                ))
                
                current_price = aggs[-1].close if aggs else None
                change_percent = None
                if len(aggs) >= 2:
                    change_percent = round(((aggs[-1].close - aggs[-2].close) / aggs[-2].close) * 100, 2)
                
                stocks.append({
                    "symbol": results.ticker,
                    "name": results.name,
                    "current_price": current_price,
                    "change_percent": change_percent
                })
            except Exception as e:
                logger.error(f"Error getting price for {results.ticker}: {e}")
                stocks.append({
                    "symbol": results.ticker,
                    "name": results.name,
                    "current_price": None,
                    "change_percent": None
                })
        
        return stocks
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stocks/{symbol}/analysis")
async def analyze_stock(symbol: str):
    """Get comprehensive analysis for a stock"""
    try:
        symbol = symbol.upper()
        
        # Get ticker details for beta and dividend
        details = polygon_client.get_ticker_details(symbol)
        
        # Get historical data for EMA calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        
        aggs = list(polygon_client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        ))
        
        if not aggs:
            raise HTTPException(status_code=404, detail="No data available for this symbol")
        
        # Extract closing prices
        prices = [bar.close for bar in aggs]
        
        # Calculate EMAs
        ema_20 = calculate_ema(prices, 20)
        ema_50 = calculate_ema(prices, 50)
        
        # Analyze candlestick patterns (last 10 candles)
        candles = [
            {'o': bar.open, 'h': bar.high, 'l': bar.low, 'c': bar.close}
            for bar in aggs[-10:]
        ]
        candlestick_pattern = analyze_candlestick(candles)
        
        # Prepare analysis data
        analysis_data = {
            'beta': getattr(details, 'beta', None),
            'dividend_yield': getattr(details, 'dividend_yield', None),
            'ema_20': ema_20,
            'ema_50': ema_50,
            'candlestick_pattern': candlestick_pattern,
            'current_price': prices[-1] if prices else None
        }
        
        # Get AI analysis
        ai_analysis = await get_ai_analysis(symbol, analysis_data)
        
        # Calculate potential score (simple algorithm)
        potential_score = 50  # Base score
        if analysis_data['beta'] and analysis_data['beta'] < 1.2:
            potential_score += 10
        if analysis_data['dividend_yield'] and analysis_data['dividend_yield'] > 2:
            potential_score += 15
        if ema_20 and ema_50 and ema_20 > ema_50:
            potential_score += 15
        if "Alcista" in candlestick_pattern:
            potential_score += 10
        
        potential_score = min(100, potential_score)
        
        return {
            'symbol': symbol,
            'name': details.name,
            'beta': analysis_data['beta'],
            'dividend_yield': analysis_data['dividend_yield'],
            'ema_20': ema_20,
            'ema_50': ema_50,
            'candlestick_pattern': candlestick_pattern,
            'ai_analysis': ai_analysis,
            'potential_score': potential_score,
            'current_price': analysis_data['current_price'],
            'historical_prices': prices[-30:],  # Last 30 days
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio management
@api_router.post("/portfolio", response_model=PortfolioStock)
async def add_to_portfolio(stock: PortfolioStockCreate):
    """Add stock to portfolio"""
    portfolio_obj = PortfolioStock(**stock.model_dump())
    doc = portfolio_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    doc['purchase_date'] = doc['purchase_date'].isoformat()
    
    await db.portfolio.insert_one(doc)
    return portfolio_obj

@api_router.get("/portfolio", response_model=List[PortfolioStock])
async def get_portfolio():
    """Get all portfolio stocks"""
    stocks = await db.portfolio.find({}, {"_id": 0}).to_list(1000)
    
    for stock in stocks:
        if isinstance(stock['timestamp'], str):
            stock['timestamp'] = datetime.fromisoformat(stock['timestamp'])
        if isinstance(stock['purchase_date'], str):
            stock['purchase_date'] = datetime.fromisoformat(stock['purchase_date'])
    
    return stocks

@api_router.delete("/portfolio/{stock_id}")
async def remove_from_portfolio(stock_id: str):
    """Remove stock from portfolio"""
    result = await db.portfolio.delete_one({"id": stock_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Stock not found in portfolio")
    return {"message": "Stock removed from portfolio"}

@api_router.get("/portfolio/summary")
async def get_portfolio_summary():
    """Get portfolio summary with current values"""
    stocks = await db.portfolio.find({}, {"_id": 0}).to_list(1000)
    
    total_invested = 0
    total_current_value = 0
    portfolio_details = []
    
    for stock in stocks:
        symbol = stock['symbol']
        shares = stock['shares']
        purchase_price = stock['purchase_price']
        
        # Get current price
        try:
            aggs = list(polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d')
            ))
            current_price = aggs[-1].close if aggs else purchase_price
        except:
            current_price = purchase_price
        
        invested = shares * purchase_price
        current_value = shares * current_price
        profit_loss = current_value - invested
        profit_loss_percent = (profit_loss / invested) * 100 if invested > 0 else 0
        
        total_invested += invested
        total_current_value += current_value
        
        portfolio_details.append({
            'id': stock['id'],
            'symbol': symbol,
            'name': stock['name'],
            'shares': shares,
            'purchase_price': purchase_price,
            'current_price': current_price,
            'invested': round(invested, 2),
            'current_value': round(current_value, 2),
            'profit_loss': round(profit_loss, 2),
            'profit_loss_percent': round(profit_loss_percent, 2)
        })
    
    return {
        'total_invested': round(total_invested, 2),
        'total_current_value': round(total_current_value, 2),
        'total_profit_loss': round(total_current_value - total_invested, 2),
        'total_profit_loss_percent': round(((total_current_value - total_invested) / total_invested * 100) if total_invested > 0 else 0, 2),
        'stocks': portfolio_details
    }

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