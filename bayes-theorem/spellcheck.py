import re
import collections

#P(c|w)=P(w|c)*P(c)/P(w)
#c-correct, w-wrong

#text to lower case and eliminate special chars
def transformText(text):
    transformed_text=re.findall('[a-z]+', text.lower())
    return transformed_text

#in case typing a word not present in text file, make P(w) at least 1
#make a dict that collects word frequencies
def makeWordBank(text):
    word_bank=collections.defaultdict(lambda:1)
    for i in text:
        word_bank[i]+=1
    return word_bank 

frequencies=makeWordBank(transformText(open("textbank.txt").read()))

def singleError(word):
    #deletion, insertion, subsititution, transposition
    word_parts=[]
    for i in range(len(word)+1):
        word_parts.append((word[:i],word[i:]))
    #deletion
    dels=[left+right[1:] for left, right in word_parts if len(right)>0]
    #insertions
    letters="abcdefghijklmnopqrstuvwxyz"
    ins=[left+letter+right for letter in letters for left, right in word_parts]
    #subsitution
    subs=[left+letter+right[1:] for letter in letters for left, right in word_parts if len(right)>0]
    #transposition
    trans=[left+right[1]+right[0]+right[2:] for left, right in word_parts if len(right)>1]
    single_error_words=set(ins+dels+subs+trans)
    return single_error_words

# print(singleError("speling"))

def inWordBank(word):
    return set(i for i in word if i in frequencies)
# print(inWordBank(singleError("appl")))

#P(w|c)
def spellingCorrectProb(word):
    single_edits=inWordBank(singleError(word))
    double_edits=set()
    for i in single_edits:
        temp=singleError(i)
        double_edits=inWordBank(double_edits.union(temp))
    return inWordBank({word}) or inWordBank(single_edits) or inWordBank(double_edits)

# print(spellingCorrectProb("app"))

#P(c)
def wordProb(word):
    return frequencies[word]/sum(frequencies.values())

def spellingCorrection(word):
    potential_correct_freq_list=[frequencies[word]/sum(frequencies.values()) for word in spellingCorrectProb(word)]
    potential_correct_freq_dict={frequencies[word]/sum(frequencies.values()):word for word in spellingCorrectProb(word)}
    best_match=max(potential_correct_freq_list)
    return potential_correct_freq_dict[best_match]

def main():
    print("input 0 to exit")
    word=input("input a word with incorrect spelling:")
    while word!="0":
        print(spellingCorrection(word))
        word=input("input a word with incorrect spelling:")
    
    

if __name__=="__main__":
    main()
