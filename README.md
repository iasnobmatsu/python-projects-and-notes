# bayes-spelling-correction
A simple English spelling checker based on bayes's theorem, inspired by [Peter Norvig's work on spelling correction](http://www.norvig.com/spell-correct.html). 

The spelling corrector has a reference of word frequencies from three books (Alice in Wonderland, Sherlock Holmes, Heart of Darkness), all found from [Project Gutenberg](http://www.gutenberg.org/). Since it only uses three books for word reference, the spelling checker does not have a very high accuracy for words in other fields. But this could be improved by changing textbank.txt. The purpose of this program is simply to implement bayes theorem so I don't think I will spend too much time on finding an appropriate word bank.

### Bayes Theorem for spelling correction
For `c` represention correct word and `w` representing wrong spelling, word correction means to find `P(c|w)` meaning to find the most probable `c` when user typed `w`.

By Bayes Theorem, `P(c|w)=P(w|c)*P(c)/P(w)`.`P(c|w)` is probability of typing `w` but meaning `c`. `P(w|c)` is probability of wanting to typre `c` but typed `w`. It is easier to find out.

### python implementation
Use a word bank created from three books and put all words into a dictionary with their frequencies. 
```#text to lower case and eliminate special chars
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
```

Find `P(w|c)` which is how likely a wrong spelling would occur.
```
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

def inWordBank(word):
    return set(i for i in word if i in frequencies)

#P(w|c)
def spellingCorrectProb(word):
    single_edits=inWordBank(singleError(word))
    double_edits=set()
    for i in single_edits:
        temp=singleError(i)
        double_edits=inWordBank(double_edits.union(temp))
    return inWordBank({word}) or inWordBank(single_edits) or inWordBank(double_edits)
```
Find P(c) which is how likely a word is to occur in the word bank.
```
#P(c)
def wordProb(word):
    return frequencies[word]/sum(frequencies.values())
```
    
Put all parts together to return a possible correct word.
```def spellingCorrection(word):
    potential_correct_freq_list=[frequencies[word]/sum(frequencies.values()) for word in spellingCorrectProb(word)]
    potential_correct_freq_dict={frequencies[word]/sum(frequencies.values()):word for word in spellingCorrectProb(word)}
    best_match=max(potential_correct_freq_list)
    return potential_correct_freq_dict[best_match]
```
