import requests
from bs4 import BeautifulSoup
import bs4

def parse(url):
    #soup
    response=requests.get(url)
    response.encoding=response.apparent_encoding
    fic=BeautifulSoup(response.text, "html.parser")
    return fic


def saveFic(fic, url):
    
    #title,author
    fic_title=fic.find("h2", class_="title heading").contents[0].strip()#string
    fic_author=fic.find("a", rel="author").contents[0].strip()#string
    
        
    #chapters
    chapter_titles_container=fic.find_all("h3",class_="title")                                                          
    chapter_titles=[]#list of strings for chapter titles
    for div in chapter_titles_container:
        temp=div.get_text().strip()
        chapter_titles.append(temp)
    chapter_titles
    

    
    #content
#     article_content_container=fic.find_all("div", class_="userstuff module")#all article divs
#     if (article_content_container==[] or article_content_container==None):
#         article_content_container=fic.find_all("div", class_="userstuff")
#     article_content=[]
#     for i in range(len(article_content_container)):
#         temp=article_content_container[i].find_all("p")
#         for j in range(len(temp)):
#             if (len(temp[j].contents)==0):
#                 temp[j]=''
#             else:
#                 holder=''
#                 for k in range(len(temp[j].contents)):
#                     if ((not isinstance(temp[j].contents[k],bs4.element.NavigableString) and len(temp[j].contents[k])>0)):
#                         if isinstance(temp[j].contents[k].contents[0],bs4.element.NavigableString):
#                             temp[j].contents[k]=temp[j].contents[k].contents[0]
#                         else:
#                             temp[j].contents[k]=str(temp[j].contents[k])
#                     elif not isinstance(temp[j].contents[k],bs4.element.NavigableString):
#                         temp[j].contents[k]=''
#                     holder+=temp[j].contents[k]
#                 temp[j]=holder

#         article_content.append(temp)

    article_content_container=fic.find_all("div", class_="userstuff module")#all article divs
    if (article_content_container==[] or article_content_container==None):
        article_content_container=fic.find_all("div", class_="userstuff")
    article_content=[]

    for div in article_content_container:
        temp=div.find_all("p")
        for j in range(len(temp)):
            if (len(temp[j].contents)==0):
                temp[j]=''
            else:
                temp[j]=temp[j].get_text().strip()
        article_content.append(temp)


        
    #save file
    path=fic_title.replace(" ","")+".txt"
    f=open(path,"w")
    bookt="\n"+fic_title+"\nby "+fic_author+"\n"+url+"\n\n\n"
    f.write(bookt)
 
    meta=getInfo(fic)
    for k,v in meta.items():
        s=k+" : "+v+"\n"
        f.write(s)
     
    f.write("\n\n\n")
    for i in range(len(article_content)):
        if not chapter_titles==[]:
            f.write(chapter_titles[i])
        f.write("\n")
        for j in article_content[i]:
            f.write("\t")
            f.write(j)
            f.write("\n")
        f.write("\n\n\n")
    f.close()


def getInfo(fic):
    metagroup=fic.find("dl", class_="work meta group")
    
    tags_container=metagroup.find_all("dt")
    tags=[]
    for dt in tags_container:
        temp=dt.get_text().strip()
        tags.append(temp)

    description_container=metagroup.find_all("dd")
    description=[]
    for des in description_container:
        temp=des.get_text().strip()
        description.append(temp)
    
    metadata={}
    for i in range(len(tags)):
        metadata[tags[i]]=description[i]
    return metadata

def main():
#     url="https://archiveofourown.org/works/919551/chapters/1785529"
    url= input("url:")
    soup=parse(url)
    saveFic(soup, url)

main()
    