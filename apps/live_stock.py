import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen

def page_settings():
        components.html(
        """
        <input type="text" placeholder="Search symbols" name="search" id="pairs" OnChange="LoadCharts();">
        
        <br>
        <br>
        <!-- TradingView Widget BEGIN -->
        
        <div class="tradingview-widget-container">
        <div id="technical-analysis"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        LoadCharts()

        function LoadCharts(){
        let symbol = document.getElementById("pairs").value;        
        new TradingView.widget(
        {
        "container_id": "technical-analysis",

        "width": 1000,
        "height": 610,
        "symbol": ""+symbol,
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
        }
        </script>
        </div>
        
        <!-- TradingView Widget END -->
        """
        ,
        height=700, width=1050
        )
        st.title('View fundamentals of stock')
        user_input = st.text_input("ENTER STOCK SYMBOL",value="AAPL")
        information = []
        information.append(yf.Ticker(user_input).info)
        fundamentals = ["marketCap","fullTimeEmployees","priceToBook","floatShares","currentRatio","totalDebt","totalCash","volume","fiftyTwoWeekHigh","fiftyTwoWeekLow","dividendRate","dividendYield","profitMargins","grossMargins","grossProfits"]
        df = pd.DataFrame(information)
        df = df.set_index('symbol')
        df = df.append(pd.Series(), ignore_index=True)
        st.write(df[df.columns[df.columns.isin(fundamentals)]])

        #news scraper
        st.title("Latest News column")
        url = ("http://finviz.com/quote.ashx?t=" + user_input.lower())
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")

        def get_news():
                try:
                        # Find news table
                        news = pd.read_html(str(html), attrs={'class': 'fullview-news-outer'})[0]
                        links = []
                        for a in html.find_all('a', class_="tab-link-news"):
                                links.append(a['href'])

                        # Clean up news dataframe
                        news.columns = ['Date', 'News Headline']
                        news['Article Link'] = links
                        news = news.set_index('Date')
                        return news

                except Exception as e:
                        return e
        st.write(get_news())
def app():
    page_settings()
