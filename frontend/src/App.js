import { useState, useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { Search, TrendingUp, Wallet, BarChart3, LineChart, DollarSign, Activity, AlertCircle, TrendingDown, Minus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [portfolio, setPortfolio] = useState([]);
  const [portfolioSummary, setPortfolioSummary] = useState(null);

  const searchStocks = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.get(`${API}/stocks/search?query=${searchQuery}`);
      setSearchResults(response.data);
      if (response.data.length === 0) {
        toast.error("No se encontraron resultados");
      }
    } catch (error) {
      console.error("Search error:", error);
      toast.error("Error al buscar acciones");
    } finally {
      setLoading(false);
    }
  };

  const loadPortfolio = async () => {
    try {
      const response = await axios.get(`${API}/portfolio/summary`);
      setPortfolioSummary(response.data);
      setPortfolio(response.data.stocks);
    } catch (error) {
      console.error("Portfolio load error:", error);
    }
  };

  const removeFromPortfolio = async (stockId) => {
    try {
      await axios.delete(`${API}/portfolio/${stockId}`);
      toast.success("Acción eliminada de la cartera");
      loadPortfolio();
    } catch (error) {
      console.error("Remove error:", error);
      toast.error("Error al eliminar acción");
    }
  };

  useEffect(() => {
    loadPortfolio();
  }, []);

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl font-bold text-white mb-2" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
            Analizador de Acciones
          </h1>
          <p className="text-white/80 text-lg" style={{ fontFamily: 'Inter, sans-serif' }}>
            Análisis técnico avanzado con IA para tus inversiones
          </p>
        </div>

        {/* Search Section */}
        <Card className="mb-8 backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="search-section">
          <CardHeader>
            <CardTitle className="flex items-center gap-2" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
              <Search className="w-5 h-5" />
              Buscar Acciones
            </CardTitle>
            <CardDescription>Ingresa el símbolo de la acción (ej: AAPL, TSLA, MSFT)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-3">
              <Input
                data-testid="search-input"
                placeholder="Símbolo de acción..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && searchStocks()}
                className="text-lg"
              />
              <Button 
                data-testid="search-button"
                onClick={searchStocks} 
                disabled={loading}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              >
                {loading ? "Buscando..." : "Buscar"}
              </Button>
            </div>

            {searchResults.length > 0 && (
              <div className="mt-6 grid gap-3" data-testid="search-results">
                {searchResults.map((stock) => (
                  <div
                    key={stock.symbol}
                    data-testid={`stock-result-${stock.symbol}`}
                    className="p-4 rounded-lg bg-gradient-to-r from-slate-50 to-slate-100 hover:from-slate-100 hover:to-slate-200 cursor-pointer transition-all flex justify-between items-center"
                    onClick={() => navigate(`/stock/${stock.symbol}`)}
                  >
                    <div>
                      <h3 className="text-xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                        {stock.symbol}
                      </h3>
                      <p className="text-sm text-gray-600">{stock.name}</p>
                    </div>
                    <div className="text-right">
                      {stock.current_price && (
                        <p className="text-2xl font-bold">${stock.current_price.toFixed(2)}</p>
                      )}
                      {stock.change_percent !== null && (
                        <Badge variant={stock.change_percent >= 0 ? "default" : "destructive"}>
                          {stock.change_percent >= 0 ? "+" : ""}{stock.change_percent}%
                        </Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Portfolio Section */}
        <Card className="backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="portfolio-section">
          <CardHeader>
            <CardTitle className="flex items-center gap-2" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
              <Wallet className="w-5 h-5" />
              Mi Cartera
            </CardTitle>
          </CardHeader>
          <CardContent>
            {portfolioSummary && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6" data-testid="portfolio-summary">
                <div className="p-4 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 text-white">
                  <p className="text-sm opacity-90">Invertido</p>
                  <p className="text-2xl font-bold">${portfolioSummary.total_invested.toLocaleString()}</p>
                </div>
                <div className="p-4 rounded-lg bg-gradient-to-br from-purple-500 to-purple-600 text-white">
                  <p className="text-sm opacity-90">Valor Actual</p>
                  <p className="text-2xl font-bold">${portfolioSummary.total_current_value.toLocaleString()}</p>
                </div>
                <div className={`p-4 rounded-lg text-white ${portfolioSummary.total_profit_loss >= 0 ? 'bg-gradient-to-br from-green-500 to-green-600' : 'bg-gradient-to-br from-red-500 to-red-600'}`}>
                  <p className="text-sm opacity-90">Ganancia/Pérdida</p>
                  <p className="text-2xl font-bold">
                    {portfolioSummary.total_profit_loss >= 0 ? "+" : ""}${portfolioSummary.total_profit_loss.toLocaleString()}
                  </p>
                </div>
                <div className={`p-4 rounded-lg text-white ${portfolioSummary.total_profit_loss_percent >= 0 ? 'bg-gradient-to-br from-emerald-500 to-emerald-600' : 'bg-gradient-to-br from-rose-500 to-rose-600'}`}>
                  <p className="text-sm opacity-90">Rendimiento</p>
                  <p className="text-2xl font-bold">
                    {portfolioSummary.total_profit_loss_percent >= 0 ? "+" : ""}{portfolioSummary.total_profit_loss_percent}%
                  </p>
                </div>
              </div>
            )}

            {portfolio.length > 0 ? (
              <div className="space-y-3" data-testid="portfolio-stocks">
                {portfolio.map((stock) => (
                  <div
                    key={stock.id}
                    data-testid={`portfolio-stock-${stock.symbol}`}
                    className="p-4 rounded-lg bg-gradient-to-r from-slate-50 to-slate-100 hover:shadow-md transition-shadow"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 
                            className="text-xl font-bold cursor-pointer hover:text-blue-600"
                            style={{ fontFamily: 'Space Grotesk, sans-serif' }}
                            onClick={() => navigate(`/stock/${stock.symbol}`)}
                          >
                            {stock.symbol}
                          </h3>
                          <Badge variant={stock.profit_loss >= 0 ? "default" : "destructive"}>
                            {stock.profit_loss >= 0 ? "+" : ""}${stock.profit_loss.toFixed(2)} ({stock.profit_loss_percent}%)
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">{stock.name}</p>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                          <div>
                            <p className="text-gray-500">Acciones</p>
                            <p className="font-semibold">{stock.shares}</p>
                          </div>
                          <div>
                            <p className="text-gray-500">Precio Compra</p>
                            <p className="font-semibold">${stock.purchase_price.toFixed(2)}</p>
                          </div>
                          <div>
                            <p className="text-gray-500">Precio Actual</p>
                            <p className="font-semibold">${stock.current_price.toFixed(2)}</p>
                          </div>
                          <div>
                            <p className="text-gray-500">Valor Total</p>
                            <p className="font-semibold">${stock.current_value.toFixed(2)}</p>
                          </div>
                        </div>
                      </div>
                      <Button
                        data-testid={`remove-stock-${stock.symbol}`}
                        variant="destructive"
                        size="sm"
                        onClick={() => removeFromPortfolio(stock.id)}
                      >
                        Eliminar
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500" data-testid="empty-portfolio">
                <Wallet className="w-16 h-16 mx-auto mb-4 opacity-30" />
                <p>Tu cartera está vacía. Busca acciones y agrégalas para comenzar.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

const StockDetail = () => {
  const navigate = useNavigate();
  const [symbol, setSymbol] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [addToPortfolioOpen, setAddToPortfolioOpen] = useState(false);
  const [portfolioForm, setPortfolioForm] = useState({
    shares: "",
    purchase_price: "",
    purchase_date: new Date().toISOString().split('T')[0]
  });

  useEffect(() => {
    const path = window.location.pathname;
    const symbolFromPath = path.split('/').pop();
    setSymbol(symbolFromPath);
    loadAnalysis(symbolFromPath);
  }, []);

  const loadAnalysis = async (sym) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/stocks/${sym}/analysis`);
      setAnalysis(response.data);
      setPortfolioForm(prev => ({
        ...prev,
        purchase_price: response.data.current_price?.toString() || ""
      }));
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Error al cargar análisis");
    } finally {
      setLoading(false);
    }
  };

  const addToPortfolio = async () => {
    try {
      await axios.post(`${API}/portfolio`, {
        symbol: symbol,
        name: analysis.name,
        shares: parseFloat(portfolioForm.shares),
        purchase_price: parseFloat(portfolioForm.purchase_price),
        purchase_date: new Date(portfolioForm.purchase_date).toISOString()
      });
      toast.success("Acción agregada a la cartera");
      setAddToPortfolioOpen(false);
      setPortfolioForm({
        shares: "",
        purchase_price: analysis.current_price?.toString() || "",
        purchase_date: new Date().toISOString().split('T')[0]
      });
    } catch (error) {
      console.error("Add to portfolio error:", error);
      toast.error("Error al agregar a la cartera");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <div className="text-white text-2xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
          Analizando {symbol}...
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <div className="text-white text-2xl">Error al cargar datos</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <Button 
              data-testid="back-button"
              variant="ghost" 
              onClick={() => navigate('/')} 
              className="text-white hover:bg-white/20 mb-4"
            >
              ← Volver
            </Button>
            <h1 className="text-5xl font-bold text-white mb-2" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
              {symbol}
            </h1>
            <p className="text-white/80 text-lg">{analysis.name}</p>
          </div>
          <Dialog open={addToPortfolioOpen} onOpenChange={setAddToPortfolioOpen}>
            <DialogTrigger asChild>
              <Button 
                data-testid="add-portfolio-button"
                size="lg"
                className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white"
              >
                <Wallet className="w-4 h-4 mr-2" />
                Agregar a Cartera
              </Button>
            </DialogTrigger>
            <DialogContent data-testid="add-portfolio-dialog">
              <DialogHeader>
                <DialogTitle>Agregar {symbol} a Cartera</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label>Número de Acciones</Label>
                  <Input
                    data-testid="shares-input"
                    type="number"
                    step="0.01"
                    value={portfolioForm.shares}
                    onChange={(e) => setPortfolioForm({...portfolioForm, shares: e.target.value})}
                    placeholder="Ej: 10"
                  />
                </div>
                <div>
                  <Label>Precio de Compra</Label>
                  <Input
                    data-testid="price-input"
                    type="number"
                    step="0.01"
                    value={portfolioForm.purchase_price}
                    onChange={(e) => setPortfolioForm({...portfolioForm, purchase_price: e.target.value})}
                    placeholder="Ej: 150.50"
                  />
                </div>
                <div>
                  <Label>Fecha de Compra</Label>
                  <Input
                    data-testid="date-input"
                    type="date"
                    value={portfolioForm.purchase_date}
                    onChange={(e) => setPortfolioForm({...portfolioForm, purchase_date: e.target.value})}
                  />
                </div>
                <Button 
                  data-testid="confirm-add-button"
                  onClick={addToPortfolio} 
                  className="w-full"
                  disabled={!portfolioForm.shares || !portfolioForm.purchase_price}
                >
                  Agregar
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Price & Score */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <Card className="backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="price-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <DollarSign className="w-5 h-5" />
                Precio Actual
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-5xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                ${analysis.current_price?.toFixed(2)}
              </p>
            </CardContent>
          </Card>

          <Card className="backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="score-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Puntuación de Potencial
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <div className="text-5xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                  {analysis.potential_score}
                </div>
                <div className="flex-1">
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div 
                      className="h-4 rounded-full bg-gradient-to-r from-green-500 to-emerald-600"
                      style={{ width: `${analysis.potential_score}%` }}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Technical Indicators */}
        <Card className="mb-8 backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="indicators-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Indicadores Técnicos
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-500 mb-1">Beta</p>
                <p className="text-3xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                  {analysis.beta?.toFixed(2) || "N/A"}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Dividend Yield</p>
                <p className="text-3xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                  {analysis.dividend_yield?.toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">EMA 20</p>
                <p className="text-3xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                  ${analysis.ema_20?.toFixed(2) || "N/A"}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">EMA 50</p>
                <p className="text-3xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                  ${analysis.ema_50?.toFixed(2) || "N/A"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Candlestick & AI Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="candlestick-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Análisis de Velas Japonesas
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="p-6 rounded-lg bg-gradient-to-br from-blue-50 to-purple-50">
                <p className="text-xl font-semibold" style={{ fontFamily: 'Inter, sans-serif' }}>
                  {analysis.candlestick_pattern}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="ai-analysis-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LineChart className="w-5 h-5" />
                Análisis con IA
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="p-6 rounded-lg bg-gradient-to-br from-purple-50 to-pink-50">
                <p className="text-base leading-relaxed" style={{ fontFamily: 'Inter, sans-serif' }}>
                  {analysis.ai_analysis}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Price Chart */}
        {analysis.historical_prices && analysis.historical_prices.length > 0 && (
          <Card className="mt-8 backdrop-blur-md bg-white/95 shadow-2xl border-0" data-testid="chart-card">
            <CardHeader>
              <CardTitle>Gráfico de Precios (Últimos 30 Días)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-end justify-between gap-1">
                {analysis.historical_prices.map((price, idx) => {
                  const maxPrice = Math.max(...analysis.historical_prices);
                  const minPrice = Math.min(...analysis.historical_prices);
                  const height = ((price - minPrice) / (maxPrice - minPrice)) * 100;
                  
                  return (
                    <div
                      key={idx}
                      className="flex-1 bg-gradient-to-t from-blue-500 to-purple-500 rounded-t transition-all hover:opacity-80"
                      style={{ height: `${height}%` }}
                      title={`$${price.toFixed(2)}`}
                    />
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <Toaster position="top-right" richColors />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/stock/:symbol" element={<StockDetail />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;