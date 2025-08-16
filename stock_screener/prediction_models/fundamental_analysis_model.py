"""
Fundamental Analysis Price Prediction Model
==========================================

Uses fundamental financial data to determine intrinsic value through multiple valuation methods.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import yfinance as yf
from .base_model import BasePredictionModel


class FundamentalAnalysisModel(BasePredictionModel):
    """
    Fundamental analysis-based price prediction using:
    - P/E ratio valuation
    - P/B ratio valuation  
    - DCF (Discounted Cash Flow) analysis
    - Graham Formula
    - Revenue/Earnings growth analysis
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs):
        """
        Initialize Fundamental Analysis Model.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            **kwargs: Additional parameters
                - fundamental_data: Pre-fetched fundamental data (optional)
                - prediction_days: Number of days to predict (default: 30)
        """
        super().__init__(symbol, historical_data, **kwargs)
        self.prediction_days = kwargs.get('prediction_days', 30)
        self.fundamental_data = kwargs.get('fundamental_data')
        
        # Fetch fundamental data if not provided
        if self.fundamental_data is None:
            self.fundamental_data = self._fetch_fundamental_data()
    
    def predict(self) -> Dict[str, Any]:
        """
        Generate fundamental analysis-based price prediction.
        
        Returns:
            Dict with predicted_price, confidence, methods_used, etc.
        """
        try:
            if not self.fundamental_data:
                return {
                    "predicted_price": self.get_current_price(),
                    "confidence": 0.1,
                    "method": "Fundamental Analysis",
                    "error": "No fundamental data available"
                }
            
            predictions = []
            methods_used = []
            individual_predictions = []
            
            # 1. P/E based valuation
            pe_prediction = self._pe_based_valuation()
            if pe_prediction:
                predictions.append(pe_prediction['price'])
                methods_used.append(pe_prediction['method'])
                individual_predictions.append(pe_prediction)
            
            # 2. P/B based valuation
            pb_prediction = self._pb_based_valuation()
            if pb_prediction:
                predictions.append(pb_prediction['price'])
                methods_used.append(pb_prediction['method'])
                individual_predictions.append(pb_prediction)
            
            # 3. DCF analysis
            dcf_prediction = self._dcf_analysis()
            if dcf_prediction:
                predictions.append(dcf_prediction['price'])
                methods_used.append(dcf_prediction['method'])
                individual_predictions.append(dcf_prediction)
            
            # 4. Graham Formula
            graham_prediction = self._graham_formula()
            if graham_prediction:
                predictions.append(graham_prediction['price'])
                methods_used.append(graham_prediction['method'])
                individual_predictions.append(graham_prediction)
            
            # 5. Revenue Growth based valuation
            revenue_prediction = self._revenue_growth_valuation()
            if revenue_prediction:
                predictions.append(revenue_prediction['price'])
                methods_used.append(revenue_prediction['method'])
                individual_predictions.append(revenue_prediction)
            
            if predictions:
                # Calculate weighted average (give more weight to DCF and P/E)
                avg_prediction = self._calculate_weighted_average(individual_predictions)
                confidence = self.get_confidence()
                
                return {
                    "predicted_price": round(avg_prediction, 2),
                    "confidence": round(confidence, 2),
                    "methods_used": methods_used,
                    "individual_predictions": [round(p, 2) for p in predictions],
                    "method": "Fundamental Analysis",
                    "prediction_date": pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days),
                    "additional_metrics": {
                        "pe_ratio": self.fundamental_data.get('trailingPE'),
                        "pb_ratio": self.fundamental_data.get('priceToBook'),
                        "market_cap": self.fundamental_data.get('marketCap'),
                        "revenue_growth": self.fundamental_data.get('revenueGrowth'),
                        "profit_margins": self.fundamental_data.get('profitMargins')
                    }
                }
            else:
                return {
                    "predicted_price": self.get_current_price(),
                    "confidence": 0.1,
                    "method": "Fundamental Analysis",
                    "error": "Insufficient fundamental data for valuation"
                }
                
        except Exception as e:
            self.logger.error(f"Fundamental analysis prediction failed: {e}")
            return {
                "predicted_price": self.get_current_price(),
                "confidence": 0.1,
                "method": "Fundamental Analysis",
                "error": f"Prediction failed: {e}"
            }
    
    def get_confidence(self) -> float:
        """
        Calculate confidence based on data availability and quality.
        
        Returns:
            Confidence score between 0.1 and 0.85
        """
        try:
            if not self.fundamental_data:
                return 0.1
            
            confidence_factors = []
            
            # Data availability factor
            key_metrics = ['trailingPE', 'priceToBook', 'freeCashflow', 
                          'trailingEps', 'revenueGrowth', 'profitMargins']
            available_metrics = sum(1 for metric in key_metrics 
                                  if self.fundamental_data.get(metric) is not None)
            data_availability = available_metrics / len(key_metrics)
            confidence_factors.append(data_availability * 0.4)
            
            # Financial health factor
            profit_margins = self.fundamental_data.get('profitMargins', 0)
            if profit_margins:
                margin_score = min(1.0, max(0.0, profit_margins * 10))  # 10% margin = 1.0 score
                confidence_factors.append(margin_score * 0.3)
            
            # Market cap stability (larger companies = more reliable)
            market_cap = self.fundamental_data.get('marketCap', 0)
            if market_cap:
                # Log scale for market cap confidence
                cap_confidence = min(1.0, np.log10(market_cap / 1e9) / 2)  # $1B = 0.5, $100B = 1.0
                confidence_factors.append(max(0.1, cap_confidence) * 0.2)
            
            # PE ratio reasonableness
            pe_ratio = self.fundamental_data.get('trailingPE')
            if pe_ratio and pe_ratio > 0:
                pe_confidence = 1.0 if 5 <= pe_ratio <= 30 else 0.5
                confidence_factors.append(pe_confidence * 0.1)
            
            overall_confidence = sum(confidence_factors) if confidence_factors else 0.1
            return max(0.1, min(0.85, overall_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.1
    
    def _fetch_fundamental_data(self) -> Optional[Dict]:
        """Fetch fundamental data using yfinance."""
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            if info and 'trailingPE' in info:
                return info
            else:
                self.logger.warning(f"Limited fundamental data for {self.symbol}")
                return info if info else {}
                
        except Exception as e:
            self.logger.error(f"Error fetching fundamental data: {e}")
            return {}
    
    def _pe_based_valuation(self) -> Optional[Dict]:
        """Calculate price using P/E ratio method."""
        try:
            pe_ratio = self.fundamental_data.get('trailingPE')
            eps = self.fundamental_data.get('trailingEps')
            sector = self.fundamental_data.get('sector', '')
            
            if not pe_ratio or not eps or pe_ratio <= 0:
                return None
            
            # Use actual P/E ratio with adjustments for extreme values
            if pe_ratio > 50:  # Adjust only for extremely high P/E
                if sector in ['Technology', 'Healthcare']:
                    adjusted_pe = min(pe_ratio, 25)
                else:
                    adjusted_pe = min(pe_ratio, 20)
                price = eps * adjusted_pe
                method = "P/E Valuation (Adjusted)"
            else:
                # Use actual P/E ratio - market's real valuation
                price = eps * pe_ratio
                method = "P/E Valuation (Actual)"
            
            return {
                'price': price,
                'method': method,
                'weight': 0.3,  # 30% weight in ensemble
                'pe_used': pe_ratio,
                'eps': eps
            }
            
        except Exception as e:
            self.logger.error(f"P/E valuation error: {e}")
            return None
    
    def _pb_based_valuation(self) -> Optional[Dict]:
        """Calculate price using P/B ratio method."""
        try:
            pb_ratio = self.fundamental_data.get('priceToBook')
            book_value = self.fundamental_data.get('bookValue')
            
            if not pb_ratio or not book_value or pb_ratio <= 0:
                return None
            
            # Use actual P/B ratio with adjustments for extreme values
            if pb_ratio > 10:  # Adjust only for extremely high P/B
                adjusted_pb = min(pb_ratio, 5)
                price = book_value * adjusted_pb
                method = "P/B Valuation (Adjusted)"
            else:
                # Use actual P/B ratio - market's real assessment
                price = book_value * pb_ratio
                method = "P/B Valuation (Actual)"
            
            return {
                'price': price,
                'method': method,
                'weight': 0.2,  # 20% weight in ensemble
                'pb_used': pb_ratio,
                'book_value': book_value
            }
            
        except Exception as e:
            self.logger.error(f"P/B valuation error: {e}")
            return None
    
    def _dcf_analysis(self) -> Optional[Dict]:
        """Calculate price using DCF analysis."""
        try:
            free_cash_flow = self.fundamental_data.get('freeCashflow')
            shares_outstanding = self.fundamental_data.get('sharesOutstanding')
            market_cap = self.fundamental_data.get('marketCap')
            revenue_growth = self.fundamental_data.get('revenueGrowth', 0)
            
            if not free_cash_flow or not shares_outstanding or shares_outstanding <= 0:
                return None
            
            # Calculate FCF multiple based on current market valuation
            if market_cap and market_cap > 0:
                current_fcf_yield = free_cash_flow / market_cap
                if current_fcf_yield > 0:
                    dcf_multiple = 1 / current_fcf_yield
                    # Cap the multiple at reasonable bounds
                    dcf_multiple = max(5, min(dcf_multiple, 25))
                else:
                    dcf_multiple = self._get_growth_based_multiple(revenue_growth)
            else:
                dcf_multiple = self._get_growth_based_multiple(revenue_growth)
            
            price = (free_cash_flow * dcf_multiple) / shares_outstanding
            
            return {
                'price': price,
                'method': "DCF Analysis",
                'weight': 0.35,  # 35% weight in ensemble (highest weight)
                'fcf_multiple': dcf_multiple,
                'free_cash_flow': free_cash_flow
            }
            
        except Exception as e:
            self.logger.error(f"DCF analysis error: {e}")
            return None
    
    def _graham_formula(self) -> Optional[Dict]:
        """Calculate price using Benjamin Graham's formula with robust data validation."""
        try:
            eps = self.fundamental_data.get('trailingEps')
            growth_rate = self.fundamental_data.get('earningsQuarterlyGrowth', 0)
            current_price = self.get_current_price()
            
            if not eps or eps <= 0 or not current_price:
                return None
            
            # Robust growth rate validation and capping
            if growth_rate is None or abs(growth_rate) > 10:  # More than 1000% is likely data error
                # Fallback to industry/market average growth (assume 5-15%)
                growth_rate = 0.10  # 10% default
                self.logger.warning(f"Using default growth rate for {self.symbol} due to extreme data: {growth_rate}")
            
            # Convert to percentage and cap to reasonable ranges
            growth_rate_pct = max(-20, min(30, growth_rate * 100))  # Cap between -20% and +30%
            
            # Graham Formula: EPS * (8.5 + 2 * growth_rate)
            # Additional safeguard: ensure multiplier doesn't exceed 3x current P/E
            current_pe = current_price / eps if eps > 0 else 20
            max_reasonable_multiplier = min(50, current_pe * 3)  # No more than 3x current P/E or 50
            
            multiplier = max(5, min(max_reasonable_multiplier, 8.5 + 2 * growth_rate_pct))
            price = eps * multiplier
            
            # Final sanity check: don't predict more than 2x current price
            if price > 2 * current_price:
                price = min(price, current_price * 1.5)  # Cap at 50% increase
                self.logger.warning(f"Capped Graham formula prediction for {self.symbol} to avoid extreme valuation")
            
            return {
                'price': price,
                'method': "Graham Formula",
                'weight': 0.1,  # 10% weight in ensemble
                'growth_rate_used': growth_rate_pct,
                'eps': eps,
                'multiplier': multiplier,
                'capped': price < eps * multiplier  # Track if we had to cap the result
            }
            
        except Exception as e:
            self.logger.error(f"Graham formula error: {e}")
            return None
    
    def _revenue_growth_valuation(self) -> Optional[Dict]:
        """Calculate price based on revenue growth and margins with robust validation."""
        try:
            revenue_growth = self.fundamental_data.get('revenueGrowth')
            profit_margins = self.fundamental_data.get('profitMargins')
            market_cap = self.fundamental_data.get('marketCap')
            shares_outstanding = self.fundamental_data.get('sharesOutstanding')
            current_price = self.get_current_price()
            
            if not all([revenue_growth, profit_margins, market_cap, shares_outstanding, current_price]):
                return None
            
            # Validate and cap revenue growth to reasonable ranges
            if abs(revenue_growth) > 2:  # More than 200% is likely data error
                revenue_growth = 0.05  # Default to 5% growth
                self.logger.warning(f"Using default revenue growth for {self.symbol} due to extreme data")
            
            # Validate profit margins
            if abs(profit_margins) > 0.5:  # More than 50% margin is suspicious for most companies
                profit_margins = max(0.01, min(0.3, profit_margins))  # Cap between 1-30%
            
            # Conservative factor calculations
            growth_factor = max(0.8, min(1.5, 1 + revenue_growth))  # Much more conservative range
            margin_factor = max(0.8, min(1.3, 1 + profit_margins))   # Much more conservative range
            
            # Conservative base multiple
            revenue_multiple = growth_factor * margin_factor  # Remove the *2 multiplier
            
            # Apply the multiple to current price
            price = current_price * revenue_multiple
            
            # Final safety check: cap at 50% increase from current price
            if price > current_price * 1.5:
                price = current_price * 1.3  # Cap at 30% increase instead
                self.logger.warning(f"Capped revenue growth valuation for {self.symbol}")
            
            return {
                'price': price,
                'method': "Revenue Growth Valuation",
                'weight': 0.05,  # 5% weight in ensemble
                'growth_factor': growth_factor,
                'margin_factor': margin_factor,
                'multiple_applied': revenue_multiple
            }
            
        except Exception as e:
            self.logger.error(f"Revenue growth valuation error: {e}")
            return None
            return None
    
    def _get_growth_based_multiple(self, revenue_growth: float) -> float:
        """Get DCF multiple based on growth rate."""
        if revenue_growth and revenue_growth > 0.15:  # High growth (>15%)
            return 20
        elif revenue_growth and revenue_growth > 0.08:  # Moderate growth (>8%)
            return 15
        else:  # Low growth
            return 10
    
    def _calculate_weighted_average(self, predictions: List[Dict]) -> float:
        """Calculate weighted average of predictions."""
        if not predictions:
            return self.get_current_price()
        
        weighted_sum = sum(pred['price'] * pred['weight'] for pred in predictions)
        total_weight = sum(pred['weight'] for pred in predictions)
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean([pred['price'] for pred in predictions])


# Standalone function for direct usage
def predict_price_fundamental(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Standalone function to predict price using fundamental analysis.
    
    Args:
        symbol: Stock symbol
        days: Prediction period in days
        
    Returns:
        Prediction results dictionary
    """
    try:
        import yfinance as yf
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        historical_data = ticker.history(period="1y")
        
        if historical_data.empty:
            return {"error": f"No data found for {symbol}"}
        
        # Create and run model
        model = FundamentalAnalysisModel(symbol, historical_data, prediction_days=days)
        return model.predict()
        
    except Exception as e:
        return {"error": f"Fundamental analysis failed: {e}"}
