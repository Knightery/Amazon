import requests
from urllib.parse import urlencode
from datetime import datetime, timedelta
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
from typing import List, Dict
import concurrent.futures

class GDELTScraper:
    def __init__(self):
        self.translator = Translator()
        self.analyzer = SentimentIntensityAnalyzer()
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.setup_google_sheets()
        
    def setup_google_sheets(self):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "C:\\Users\\nyter\\Desktop\\Scraper\\googleauth.json", scope
        )
        client = gspread.authorize(creds)
        self.worksheet = client.open("Model").worksheet("EgyptSentiment")
        
    def fetch_articles(self, start_datetime: datetime, end_datetime: datetime, 
                      keyword: str = "economy", country_code: str = "turkey") -> List[Dict]:
        """Fetch articles for a given time period with retries"""
        params = {
            "query": f'"{keyword}" AND sourcecountry:{country_code}',
            "mode": "artlist",
            "maxrecords": 200,  # Increased to 200 articles per request
            "format": "json",
            "STARTDATETIME": start_datetime.strftime("%Y%m%d%H%M%S"),
            "ENDDATETIME": end_datetime.strftime("%Y%m%d%H%M%S"),
        }
        
        for attempt in range(3):  # Retry logic
            try:
                response = requests.get(
                    f"{self.base_url}?{urlencode(params)}", 
                    headers=self.headers, 
                    timeout=30
                )
                response.raise_for_status()
                return response.json().get("articles", [])
            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt == 2:
                    print(f"Failed to fetch articles after 3 attempts: {e}")
                    return []
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                
    def translate_batch(self, titles: List[str], batch_size: int = 50) -> List[str]:
        """Translate titles in batches with error handling"""
        translated_titles = []
        
        # Create batches of 50 titles
        batches = [titles[i:i + batch_size] for i in range(0, len(titles), batch_size)]
        
        for batch in batches:
            combined_text = "|".join(batch)
            for attempt in range(3):
                try:
                    translated = self.translator.translate(combined_text, dest='en').text
                    translated_batch = translated.split("|")
                    translated_titles.extend(translated_batch)
                    time.sleep(1)  # Rate limiting
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"Translation failed for batch: {e}")
                        translated_titles.extend(batch)  # Use original titles as fallback
                    time.sleep(5 * (attempt + 1))
                    
        return translated_titles
    
    def analyze_sentiments(self, texts: List[str]) -> List[float]:
        """Analyze sentiments for a batch of texts using parallel processing"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            sentiments = list(executor.map(
                lambda text: self.analyzer.polarity_scores(text)['compound'],
                texts
            ))
        return [s for s in sentiments if s != 0]  # Filter out zero sentiments
        
    def process_month(self, start_date: datetime) -> tuple:
        """Process one month of articles"""
        end_date = (start_date + timedelta(days=32)).replace(day=1)  # Next month
        
        print(f"Processing month: {start_date.strftime('%Y-%m')}")
        articles = self.fetch_articles(start_date, end_date)
        
        if not articles:
            return start_date, None
            
        # Extract all titles
        titles = [article.get('title', '') for article in articles if article.get('title')]
        
        # Translate in batches of 50
        translated_titles = self.translate_batch(titles, batch_size=50)
        
        # Analyze sentiments in parallel
        sentiments = self.analyze_sentiments(translated_titles)
        
        if sentiments:
            return start_date, sum(sentiments) / len(sentiments)
        return start_date, None
        
    def run(self, start_date: datetime = datetime(2020, 1, 1)):
        """Main execution loop with bulk worksheet updates"""
        current_date = datetime.now()
        updates = []  # Collect all updates
        row = 2
        
        while start_date < current_date:
            date, sentiment = self.process_month(start_date)
            
            if sentiment is not None:
                updates.append({
                    'range': f'A{row}:B{row}',
                    'values': [[date.strftime("%Y-%m-%d"), sentiment]]
                })
                row += 1
            
            # Bulk update every 10 months or at the end
            if len(updates) >= 10 or start_date + timedelta(days=32) >= current_date:
                self.worksheet.batch_update(updates)
                updates = []
                
            start_date = (start_date + timedelta(days=32)).replace(day=1)  # Next month
            
if __name__ == "__main__":
    scraper = GDELTScraper()
    scraper.run()