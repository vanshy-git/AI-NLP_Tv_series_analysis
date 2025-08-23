import scrapy
from bs4 import BeautifulSoup
import re

class MultiStrategyJutsuSpider(scrapy.Spider):
    name = 'narutospider'
    allowed_domains = ['naruto.fandom.com']
    
    # Try multiple starting URLs
    start_urls = [
        'https://naruto.fandom.com/wiki/Category:Jutsu',
        'https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu',
        'https://naruto.fandom.com/wiki/List_of_Jutsu'
    ]

    def parse(self, response):
        """Parse different types of jutsu listing pages"""
        
        self.logger.info(f"Processing page: {response.url}")
        self.logger.info(f"Page title: {response.css('title::text').get()}")
        
        jutsu_links = []
        
        # Strategy 1: Category page (most reliable)
        if 'Category:Jutsu' in response.url:
            jutsu_links = self.parse_category_page(response)
        
        # Strategy 2: Special browse data page
        elif 'Special:BrowseData' in response.url:
            jutsu_links = self.parse_browse_data_page(response)
        
        # Strategy 3: List page
        elif 'List_of_Jutsu' in response.url:
            jutsu_links = self.parse_list_page(response)
        
        # Follow valid jutsu links
        for href in jutsu_links:
            if href and self._is_valid_jutsu_link(href):
                yield scrapy.Request(
                    response.urljoin(href),
                    callback=self.parse_jutsu,
                    meta={'jutsu_url': href}
                )
        
        # Handle pagination for category pages
        if 'Category:Jutsu' in response.url:
            next_links = response.css('a:contains("next")::attr(href)').getall()
            for next_link in next_links:
                yield response.follow(next_link, self.parse)
    
    def parse_category_page(self, response):
        """Parse a category page like Category:Jutsu"""
        self.logger.info("Parsing category page")
        
        # Category pages have links in .mw-category-group
        category_links = response.css('.mw-category-group li a::attr(href)').getall()
        if category_links:
            self.logger.info(f"Found {len(category_links)} links in category groups")
            return category_links
        
        # Fallback: any links in the content area
        content_links = response.css('#mw-pages a::attr(href)').getall()
        if content_links:
            self.logger.info(f"Found {len(content_links)} links in pages section")
            return content_links
        
        # Last resort: all links
        all_links = response.css('a::attr(href)').getall()
        wiki_links = [link for link in all_links if link and link.startswith('/wiki/')]
        self.logger.info(f"Found {len(wiki_links)} wiki links total")
        return wiki_links
    
    def parse_browse_data_page(self, response):
        """Parse Special:BrowseData page"""
        self.logger.info("Parsing browse data page")
        
        jutsu_links = []
        
        # Try multiple selectors for browse data
        selectors = [
            '.smw-columnlist-container a::attr(href)',
            '.smw-browse-container a::attr(href)',
            '.smw-datatable a::attr(href)',
            'table.wikitable a::attr(href)',
            '.mw-parser-output a::attr(href)'
        ]
        
        for selector in selectors:
            links = response.css(selector).getall()
            if links:
                jutsu_links.extend(links)
                self.logger.info(f"Found {len(links)} links with selector: {selector}")
                break
        
        return jutsu_links
    
    def parse_list_page(self, response):
        """Parse a list page"""
        self.logger.info("Parsing list page")
        
        # List pages typically have links in lists
        list_links = response.css('ul li a::attr(href)').getall()
        if list_links:
            self.logger.info(f"Found {len(list_links)} links in lists")
            return list_links
        
        # Or in tables
        table_links = response.css('table a::attr(href)').getall()
        if table_links:
            self.logger.info(f"Found {len(table_links)} links in tables")
            return table_links
        
        return []
    
    def _is_valid_jutsu_link(self, href):
        """Check if a link is likely a jutsu page"""
        if not href or not href.startswith('/wiki/'):
            return False
        
        # Exclude system pages
        excluded_prefixes = [
            '/wiki/Special:', '/wiki/Category:', '/wiki/File:', 
            '/wiki/Template:', '/wiki/Help:', '/wiki/User:', '/wiki/Talk:'
        ]
        
        for prefix in excluded_prefixes:
            if href.startswith(prefix):
                return False
        
        # Exclude common non-jutsu pages
        excluded_terms = [
            'BrowseData', 'List_of', 'disambiguation', 'Main_Page'
        ]
        
        for term in excluded_terms:
            if term in href:
                return False
        
        return True

    def parse_jutsu(self, response):
        """Parse individual jutsu page"""
        jutsu_name = response.css("span.mw-page-title-main::text").get()
        if not jutsu_name:
            jutsu_name = response.css("h1.firstHeading::text").get(default="").strip()
        else:
            jutsu_name = jutsu_name.strip()
        
        if not jutsu_name:
            self.logger.warning(f"Could not extract jutsu name from {response.url}")
            return

        # Extract main content
        main_content = response.css("div.mw-parser-output").get()
        if not main_content:
            self.logger.warning(f"No main content found for jutsu: {jutsu_name}")
            return

        soup = BeautifulSoup(main_content, 'html.parser')

        # Extract jutsu info
        jutsu_type = ""
        jutsu_rank = ""
        jutsu_class = ""
        users = []

        # Look for infobox
        aside = soup.find('aside', class_='portable-infobox')
        if aside:
            data_cells = aside.find_all('div', class_='pi-data')
            
            for cell in data_cells:
                label = cell.find('h3', class_='pi-data-label')
                value_div = cell.find('div', class_='pi-data-value')
                
                if not label or not value_div:
                    continue
                    
                label_text = label.get_text(strip=True).lower()
                
                if 'classification' in label_text:
                    jutsu_type = value_div.get_text(strip=True)
                elif 'rank' in label_text:
                    jutsu_rank = value_div.get_text(strip=True)
                elif 'class' in label_text:
                    jutsu_class = value_div.get_text(strip=True)
                elif 'user' in label_text:
                    user_links = value_div.find_all('a')
                    users = [link.get_text(strip=True) for link in user_links if link.get_text(strip=True)]
            
            aside.decompose()

        # Clean up unwanted elements
        for element in soup.find_all(['script', 'style', 'noscript', 'nav']):
            element.decompose()

        for edit_link in soup.find_all('span', class_='mw-editsection'):
            edit_link.decompose()

        # Get clean description
        jutsu_description = soup.get_text(separator="\n", strip=True)
        
        if jutsu_description:
            sections_to_remove = ['Trivia', 'References', 'See Also', 'Navigation']
            for section in sections_to_remove:
                if section in jutsu_description:
                    jutsu_description = jutsu_description.split(section)[0].strip()
            
            jutsu_description = re.sub(r'\n\s*\n', '\n\n', jutsu_description)
            jutsu_description = jutsu_description.strip()

        self.logger.info(f"Successfully scraped: {jutsu_name}")

        yield {
            'jutsu_name': jutsu_name,
            'jutsu_type': jutsu_type,
            'jutsu_rank': jutsu_rank,
            'jutsu_class': jutsu_class,
            'users': users,
            'jutsu_description': jutsu_description,
            'source_url': response.url
        }