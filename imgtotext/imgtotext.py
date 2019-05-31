import argparse
from PIL import Image


parser=argparse.ArgumentParser()

parser.add_argument('input',help="source image file in jpg or png")
parser.add_argument('--outputtxt',help="output txt file path")
parser.add_argument('--outputhtml',help="output html file path")
parser.add_argument('--width',type=int,default=50,help="output file width,default=50")
parser.add_argument('--height',type=int,default=50,help="output file hright,default=50") 

args=parser.parse_args()
input_image=args.input
w=args.width
h=args.height
outputtxt=args.outputtxt
outputhtml=args.outputhtml

def charOut(im,j,i):
    comp=im.getpixel((j,i))#tuple of size 3 means have a solid pixel, tuple of 4 means opacity smaller than 1
    
    if len(comp)==4 and comp[3]==0:
        return ' '
    r=comp[0]
    g=comp[1]
    b=comp[2]
    chars=list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")#last char is space to make white empty

    grayscale = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    return chars[int(grayscale/(255/len(chars)))]

def imgtxt(image,w,h):

    im = Image.open(image)
    im = im.resize((w,h))

    txt = ""

    for i in range(h):
        for j in range(w):
            txt += charOut(im,j,i)
        txt += '\n'

    if outputtxt:
        f=open(outputtxt,"w")
        f.write(txt)
    else:
        f=open("output.txt","w")
        f.write(txt)



def htmlOut(im,j,i,chara):
#     html element
    comp=im.getpixel((j,i))
    elementcolor=""
    if len(comp)==4 and comp[3]==0:
        elementcolor="color:rgb(255,255,255)"
    else:
        r=comp[0]
        g=comp[1]
        b=comp[2]
        elementcolor="color: rgb({},{},{})".format(r,g,b)
    htmltxt=chara
    htmlelement="<p style=\"margin:0; display:inline-block;{}\">{}</p>".format(elementcolor,htmltxt)
    return htmlelement



    
def imghtml(image,w,h):
    im = Image.open(image)
    im = im.resize((w,h))
    
    txt = "<div class=\"txtimg\" style=\"font-family:monospace;font-size:8px\">"

    for i in range(h):
        txt+="<div class=\"line\">"
        for j in range(w):
            txt += htmlOut(im,j,i,"0")
        txt+="</div>"
        
    txt+="</div>"
    
    if outputhtml:
        f=open(outputhtml,"w")
        f.write(txt)
    else:
        f=open("output.html","w")
        f.write(txt)

    
if __name__ == '__main__':
    imgtxt(input_image,w,h)
    imghtml(input_image,w,h)