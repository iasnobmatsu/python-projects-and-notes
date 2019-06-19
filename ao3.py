import requests
from bs4 import BeautifulSoup



#get a full fic


class Fiction:
    def __init__(self, url):
        self.url=url
        self.fic=None
        self.fic_title=None
        self.fic_author=None
        self.chapter_titles=None
        self.summary=None
        self.article_content=None
        

    def makeSoup(self):    
        response=requests.get(self.url)
        response.encoding=response.apparent_encoding
        self.fic=BeautifulSoup(response.text, "html.parser")
    
    def find_title(self):
        self.fic_title=self.fic.find("h2", class_="title heading").contents[0].strip()#string

    def find_author(self):
        self.fic_author=self.fic.find("a", rel="author").contents[0].strip()#string
        
    def find_summary(self):
        summary_container=self.fic.find("div", class_="summary module")
        self.summary=summary_container.find_all("p")#list of strings for summary pars
        for i in range(len(self.summary)):
            self.summary[i]=summary[i].contents[0]

    
    def find_chapters(self):
        chapter_titles_container=self.fic.find_all("div", class_="chapter preface group")
        self.chapter_titles=[]#list of strings for chapter titles
        for div in chapter_titles_container:
            temp=div.find("a").contents[0]
            self.chapter_titles.append(temp) 
       
    
    def find_content(self):
        article_content_container=self.fic.find_all("div", class_="userstuff module")#all article divs
        self.article_content=[]

        for i in range(len(article_content_container)):
            temp=article_content_container[i].find_all("p")
            for j in range(len(temp)):
                if (len(temp[j].contents)==0):
                    temp[j]=''
                else:
                    temp[j]=temp[j].contents[0]
            self.article_content.append(temp)

    def make_article(self):
        self.makeSoup()
        self.find_title()
        self.find_author()
        self.find_summary()
        self.find_chapters()
        self.find_content()
    
def main():
    url="https://archiveofourown.org/works/91885?view_full_work=true"
    fic=Fiction(url)
    fic.make_article()
    
    print(fic.fic_title)
    print(fic.article_content[0])
    
main()
