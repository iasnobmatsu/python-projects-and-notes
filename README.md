# bayes-spelling-correction
A simple English spelling checker based on bayes's theorem, inspired by [Peter Norvig's work on spelling correction](http://www.norvig.com/spell-correct.html). 

The spelling corrector has a reference of word frequencies from three books (Alice in Wonderland, Sherlock Holmes, Heart of Darkness), all found from [Project Gutenberg](http://www.gutenberg.org/). Since it only uses three books for word reference, the spelling checker does not have a very high accuracy for words in other fields. But this could be improved by changing textbank.txt. The purpose of this program is simply to implement bayes theorem so I don't think I will spend too much time on finding an appropriate word bank.

###Bayes Theorem for spelling correction
For `c` represention correct word and `w` representing wrong spelling, word correction means to find `P(c|w)` meaning to find the most probable `c` when user typed `w`.

By Bayes Theorem, `P(c|w)=P(w|c)*P(c)/P(w)`.

