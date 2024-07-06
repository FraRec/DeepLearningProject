# DeepLearningProject
Progetto per Corso di DeepLearning

# Introduzione
La Super Resolution è uno degli ambiti più studiati nel contesto del Deep Learning.\
Ogni anno escono tantissimi Articoli Scientifici i quali studiano nuove tecniche e architetture, 
che hanno come obiettivo l'aumento delle performance e il miglioramento dei risultati dell'anno precedente.

Oggi si è arrivati a costruire delle Architetture molto complesse che nascono dalla consapevolezza 
di ciò che ha funzionato nelle architetture passate e di ciò che invece è stato scartato nel tempo.
Compiendo un analisi riferita all'evoluzione delle tecniche pubblicate durante gli anni, si possono 
trovare determinate scelte ricorrenti che contribuiscono all'ottenimento di migliori performance.

Alcune di queste scelte sono:
*   L'uso della Perceptual Loss
*   Residual Learning
*   Usare differenti tipi di ReLU
*   Sub-Pixel CNN
*   Dense Block
*   etc...

Lo studio fatto in questo progetto ha come obiettivo quello di applicare alcune  di queste migliorie (o tutte)
 ad architetture "anziane" per capire se e' possibile ottenere un migliorarmento sicuro delle performance.

Le Architetture che verranno affrontate saranno:
* SRCNN (Super-Resolution Convolutional Neural Networks)
* VDSR (Very Deep Super-Resolution)
* EDRN (Enhanced Deep Residual Networks)

L'iter con cui sarà portata avanti questa analisi sfrutterà l'implementazione di queste architetture ispirata a
quelle fatte da altre persone per poi cercare di migliorarla effettuando cambiamenti specifici e tenendo 
traccia dei risultati.
