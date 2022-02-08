import streamlit as st
import streamlit.components.v1 as components

def page_settings():
    components.html(
        """
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
        <div id="technical-analysis"></div>
        <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/symbols/AAPL/" rel="noopener" target="_blank"><span class="blue-text">AAPL Chart</span></a> by TradingView</div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget(
        {
        "container_id": "technical-analysis",

        "width": 1000,
        "height": 610,
        "symbol": "AAPL",
        "interval": "D",
        "timezone": "exchange",
        "theme": "light",
        "style": "1",
        "toolbar_bg": "#f1f3f6",
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "save_image": false,
        "studies": [
            "ROC@tv-basicstudies",
            "StochasticRSI@tv-basicstudies",
            "MASimple@tv-basicstudies"
        ],
        "show_popup_button": true,
        "popup_width": "1000",
        "popup_height": "650",
        "locale": "in"
        }
        );
        </script>
        </div>
        <!-- TradingView Widget END -->
        """,
        height=1000, width=1050
    )

def app():
    page_settings()
