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
import os
from typing import List
from pydantic import BaseModel
from fastapi import HTTPException
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()  # lee OPENAI_API_KEY del entorno

class UserMessage(BaseModel):
    role: str  # "user" | "system" | "assistant"
    content: str

def llm_chat(messages: List[UserMessage]) -> str:
    try:
        # Convertimos a formato OpenAI
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=payload)
        return resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")
import asyncio
from bs4 import BeautifulSoup
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (coincide con docker-compose)
mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/emergent")
client = AsyncIOMotorClient(mongo_uri)
db_name = os.getenv("DB_NAME", "emergent")
db = client[db_name]

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
async def scrape_finviz_data(symbol: str) -> Dict:
    """Scrape additional stock data from FINVIZ"""
    import requests
    
    try:
        url = f"https://finviz.com/quote.ashx?t={symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table with fundamental data
        table = soup.find('table', class_='snapshot-table2')
        if not table:
            return {}
        
        data = {}
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all('td')
            for i in range(0, len(cells), 2):
                if i + 1 < len(cells):
                    key = cells[i].get_text(strip=True)
                    value = cells[i + 1].get_text(strip=True)
                    data[key] = value
        
        # Parse specific values
        def parse_number(s):
            try:
                s = s.replace(',', '').replace('%', '').replace('B', '').replace('M', '')
                s = re.sub(r'[^0-9.\-]', '', s.split()[0] if ' ' else s)
                return float(s) if s and s not in ['-', ''] else None
            except:
                return None
        
        parsed_data = {
            'pe_ratio': parse_number(data.get('P/E', '')),
            'forward_pe': parse_number(data.get('Forward P/E', '')),
            'peg_ratio': parse_number(data.get('PEG', '')),
            'ps_ratio': parse_number(data.get('P/S', '')),
            'pb_ratio': parse_number(data.get('P/B', '')),
            'beta': parse_number(data.get('Beta', '')),
            'dividend_yield': parse_number(data.get('Dividend %', '')),
            'rsi': parse_number(data.get('RSI (14)', '')),
            'sma20': parse_number(data.get('SMA20', '')),
            'sma50': parse_number(data.get('SMA50', '')),
            'sma200': parse_number(data.get('SMA200', '')),
            'volatility': data.get('Volatility', ''),
            'target_price': parse_number(data.get('Target Price', '')),
            'recommendation': parse_number(data.get('Recom', '')),
            'roe': parse_number(data.get('ROE', '')),
            'roa': parse_number(data.get('ROA', '')),
            'profit_margin': parse_number(data.get('Profit Margin', '')),
            'debt_equity': parse_number(data.get('Debt/Eq', '')),
            'eps_ttm': parse_number(data.get('EPS (ttm)', '')),
            'market_cap': data.get('Market Cap', ''),
            'perf_week': parse_number(data.get('Perf Week', '')),
            'perf_month': parse_number(data.get('Perf Month', '')),
            'perf_quarter': parse_number(data.get('Perf Quarter', '')),
            'perf_year': parse_number(data.get('Perf Year', '')),
        }
        
        return parsed_data
    except Exception as e:
        logger.error(f"FINVIZ scraping error for {symbol}: {e}")
        return {}

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

async def get_ai_analysis(symbol: str, analysis_data: Dict) -> tuple[str, str]:
    """Get AI-powered analysis using OpenAI - returns (analysis, recommendation)"""
    try:
        system_msg = UserMessage(
            role="system",
            content="Eres un analista financiero experto. Proporciona análisis concisos y profesionales."
        )

        prompt = f"""
Analiza la siguiente acción: {symbol}

Datos técnicos:
- Beta: {analysis_data.get('beta', 'N/A')}
- Dividend Yield: {analysis_data.get('dividend_yield', 'N/A')}%
- P/E Ratio: {analysis_data.get('pe_ratio', 'N/A')}
- PEG Ratio: {analysis_data.get('peg_ratio', 'N/A')}
- ROE: {analysis_data.get('roe', 'N/A')}%
- Profit Margin: {analysis_data.get('profit_margin', 'N/A')}%
- EMA 20: ${analysis_data.get('ema_20', 'N/A')}
- EMA 50: ${analysis_data.get('ema_50', 'N/A')}
- RSI: {analysis_data.get('rsi', 'N/A')}
- Patrón de velas: {analysis_data.get('candlestick_pattern', 'N/A')}
- Precio actual: ${analysis_data.get('current_price', 'N/A')}
- Precio objetivo: ${analysis_data.get('target_price', 'N/A')}

IMPORTANTE: Tu respuesta debe tener exactamente este formato:

RECOMENDACIÓN: [Escribe solo una de estas tres palabras: COMPRAR, VENDER o MANTENER]

ANÁLISIS:
[Proporciona un análisis breve de máximo 150 palabras sobre:
1. Potencial de inversión
2. Riesgos principales
3. Justificación de tu recomendación]
"""
        
        user_msg = UserMessage(role="user", content=prompt)

        response_text = llm_chat([system_msg, user_msg]).strip()

        # Parsear RECOMENDACIÓN y ANÁLISIS
        recommendation = "MANTENER"
        analysis = response_text

        for line in response_text.splitlines():
            if line.upper().startswith("RECOMENDACIÓN:") or line.upper().startswith("RECOMENDACION:"):
                rec_text = line.split(":", 1)[-1].strip().upper()
                if "COMPRAR" in rec_text:
                    recommendation = "COMPRAR"
                elif "VENDER" in rec_text:
                    recommendation = "VENDER"
                else:
                    recommendation = "MANTENER"
                break

        if "ANÁLISIS:" in response_text or "ANALISIS:" in response_text:
            splitter = "ANÁLISIS:" if "ANÁLISIS:" in response_text else "ANALISIS:"
            analysis = response_text.split(splitter, 1)[-1].strip()

        return analysis, recommendation

    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return "Análisis IA no disponible temporalmente", "MANTENER"

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
        
        # Get data from FINVIZ
        finviz_data = await scrape_finviz_data(symbol)
        
        # Get historical data for EMA calculation and candlestick analysis
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
        
        # Prepare analysis data with FINVIZ data
        analysis_data = {
            'beta': finviz_data.get('beta'),
            'dividend_yield': finviz_data.get('dividend_yield'),
            'pe_ratio': finviz_data.get('pe_ratio'),
            'peg_ratio': finviz_data.get('peg_ratio'),
            'pb_ratio': finviz_data.get('pb_ratio'),
            'ps_ratio': finviz_data.get('ps_ratio'),
            'rsi': finviz_data.get('rsi'),
            'roe': finviz_data.get('roe'),
            'roa': finviz_data.get('roa'),
            'profit_margin': finviz_data.get('profit_margin'),
            'debt_equity': finviz_data.get('debt_equity'),
            'target_price': finviz_data.get('target_price'),
            'market_cap': finviz_data.get('market_cap'),
            'ema_20': ema_20,
            'ema_50': ema_50,
            'candlestick_pattern': candlestick_pattern,
            'current_price': prices[-1] if prices else None,
            'perf_week': finviz_data.get('perf_week'),
            'perf_month': finviz_data.get('perf_month'),
            'perf_quarter': finviz_data.get('perf_quarter'),
            'perf_year': finviz_data.get('perf_year'),
        }
        
        # Get AI analysis with recommendation
        ai_analysis, recommendation = await get_ai_analysis(symbol, analysis_data)
        
        # Calculate potential score (improved algorithm)
        potential_score = 50  # Base score
        
        # Beta check (lower is better for stability)
        if analysis_data['beta']:
            if analysis_data['beta'] < 1:
                potential_score += 10
            elif analysis_data['beta'] > 1.5:
                potential_score -= 10
        
        # Dividend yield (higher is better)
        if analysis_data['dividend_yield']:
            if analysis_data['dividend_yield'] > 3:
                potential_score += 15
            elif analysis_data['dividend_yield'] > 1:
                potential_score += 8
        
        # EMA trend
        if ema_20 and ema_50 and ema_20 > ema_50:
            potential_score += 15
        elif ema_20 and ema_50 and ema_20 < ema_50:
            potential_score -= 10
        
        # Candlestick pattern
        if "Alcista" in candlestick_pattern:
            potential_score += 10
        elif "Bajista" in candlestick_pattern:
            potential_score -= 10
        
        # P/E ratio (reasonable range)
        if analysis_data['pe_ratio']:
            if 15 <= analysis_data['pe_ratio'] <= 25:
                potential_score += 10
            elif analysis_data['pe_ratio'] > 40:
                potential_score -= 10
        
        # RSI (overbought/oversold)
        if analysis_data['rsi']:
            if 30 <= analysis_data['rsi'] <= 70:
                potential_score += 5
            elif analysis_data['rsi'] > 70:
                potential_score -= 10
            elif analysis_data['rsi'] < 30:
                potential_score += 10  # Oversold can be buying opportunity
        
        potential_score = max(0, min(100, potential_score))
        
        # Get ticker details for name
        details = polygon_client.get_ticker_details(symbol)
        
        return {
            'symbol': symbol,
            'name': details.name if details else symbol,
            **analysis_data,
            'ai_analysis': ai_analysis,
            'recommendation': recommendation,
            'potential_score': potential_score,
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