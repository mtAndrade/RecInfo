# -*- coding: utf-8 -*-
import operator
import numpy

documents = ['O peã e o caval são pec de xadrez. O caval é o melhor do jog.',
             'A jog envolv a torr, o peã e o rei.',
             'O peã lac o boi.',
             'Caval de rodei!',
             'Polic o jog no xadrez.']
stopwords = ['a', 'o', 'e', 'é', 'de', 'do', 'no', 'são']
separators = [' ', ',', '.', '!', '?']
query = 'xadrez peã caval torr'
alpha = 1
beta = 0.75
gama = 0.15


class PageRank:
    def __init__(self,threshhold = 0.0001, beta=0.8, number_pages = 4):
        self.number_pages = number_pages
        self.T = 1/number_pages        
        self.struct = {
            "A":{
                "linksTo":["B", "C"],
                "linkedBy":["C"],
                "rank":self.T 
            },
            "B":{
                "linksTo":["C"],
                "linkedBy":["A"],
                "rank":self.T
            },
            "C":{
                "linksTo":["A"],
                "linkedBy":["A", "B", "D"],
                "rank":self.T
            },
            "D":{
                "linksTo":["C"],
                "linkedBy":[],
                "rank":self.T
            }
        }
        
        self.threshhold = threshhold
        self.beta = beta

    def rank(self): 
        notEnd = True
        oldRank = {
            "A": self.struct["A"]["rank"],
            "B": self.struct["B"]["rank"],
            "C": self.struct["C"]["rank"],
            "D": self.struct["D"]["rank"],
        }
        while notEnd:
            for page in self.struct.keys():
                randomPath = (1 - self.beta)/self.T   

                inRank = 0
                for linked in self.struct[page]["linkedBy"]:
                    inRank += self.struct[linked]["rank"] / len( self.struct[linked]["linksTo"] )

                newRank = randomPath +  self.beta*inRank
                self.struct[page]["rank"] = newRank
            
            count = 0
            for page in oldRank.keys():                            
                # print(page, oldRank[page],self.struct[page]["rank"])
                if numpy.absolute(oldRank[page] - self.struct[page]["rank"]) <= self.threshhold:
                    count +=1
                oldRank[page] = self.struct[page]["rank"]  

            if count == self.number_pages:
                notEnd = False

        print( self.struct )
            

class HITS:
    def __init__(self,threshhold = 0.0001, beta=0.8, number_pages = 4, k=650):
        self.number_pages = number_pages
        self.k = k
        self.T = 1        
        self.struct = {
            "A":{
                "linksTo":["B", "C"],
                "linkedBy":["C"],
                "hub":self.T,
                "authority":self.T 
            },
            "B":{
                "linksTo":["C"],
                "linkedBy":["A"],
                "hub":self.T,
                "authority":self.T
            },
            "C":{
                "linksTo":["A"],
                "linkedBy":["A", "B", "D"],
                "hub":self.T,
                "authority":self.T
            },
            "D":{
                "linksTo":["C"],
                "linkedBy":[],
                "hub":self.T,
                "authority":self.T
            }
        }

    def update_auth_score(self):
        for page in self.struct.keys():
            self.struct[page]["authority"] = sum( self.struct[linkedBy]["hub"] for linkedBy in self.struct[page]["linkedBy"])
               
    def update_hub_score(self):
        for page in self.struct.keys():
            self.struct[page]["hub"] = sum( self.struct[linkTo]["authority"] for linkTo in self.struct[page]["linksTo"])

    def normalize(self):
        authority_values = []
        hub_values = []

        for page in self.struct.keys():
            authority_values.append(self.struct[page]["authority"])
            hub_values.append(self.struct[page]["hub"])
        
        auth_norm = numpy.linalg.norm(authority_values)
        hub_norm = numpy.linalg.norm(hub_values)

        for page in self.struct.keys():
            self.struct[page]["authority"] = self.struct[page]["authority"]/ auth_norm
            self.struct[page]["hub"] = self.struct[page]["hub"]/ hub_norm

    def rank(self):
        for _ in range(self.k):
            self.update_auth_score()
            self.update_hub_score()
            self.normalize()
        print(self.struct)


class Corpus:
    def __init__(self, seps=' ', stop=[]):
        self._seps = seps
        self._stop = stop
        self._docs = []
        self._terms = []
        self._freqs = []
        self._ns = []
        

    def _preprocess(self, doc):
        doc = doc.lower()
        for sep in self._seps:
            doc = doc.replace(sep, ' ')
        doc = doc.split()
        return [t for t in doc if t not in self._stop]

    def _add_term(self, term):
        if term in self._terms:
            return
        self._terms.append(term)
        self._ns.append(0)
        for freq in self._freqs:
            freq.append(0)

    def _frequency(self, doc):
        return [doc.count(t) for t in self._terms]

    def _tf(self, freq):
        return [1 + numpy.log2(f) if f != 0 else 0 for f in freq]

    def _idf(self):
        N = len(self._docs)
        return [numpy.log2(N/n) for n in self._ns]

    def _weight(self, freq):
        tf = self._tf(freq)
        idf = self._idf()
        return [t*i for (t, i) in zip(tf, idf)]

    def _betha(self, freq, doc, K = 1, b=0.75):
        tam = len(doc)
        avg = sum([len(t.split()) for t in self._docs])/len(self._docs)        
        return ((K + 1) * freq) / (K * ((1 - b) + b * (tam / avg)) + freq)
    
    def _rocchio_weight(self, freq):
        tf = self._tf(freq)
        idf = self._idf()
        return [t*i for (t, i) in zip(tf, idf)]

    def _query_weight(self, freq):
        return self._weight(freq)

    def add_document(self, doc):
        self._docs.append(doc)
        doc = self._preprocess(doc)
        for term in doc:
            self._add_term(term)
        freq = self._frequency(doc)
        self._freqs.append(freq)
        for i, f in enumerate(freq):
            self._ns[i] += f > 0

    def query_weight(self, q):
        q = self._preprocess(q)
        freq = self._frequency(q)
        return self._query_weight(freq)
    
    def query_rocchio_weight(self, q):
        q = self._preprocess(q)
        freq = self._frequency(q)
        return self._rocchio_weight(freq)

    def weight_matrix(self):
        return [self._weight(f) for f in self._freqs]

    def vector_similarity(self, docs, query):
        return numpy.dot(docs, query) / (numpy.linalg.norm(docs) * numpy.linalg.norm(query))

    def find_indexes(self,s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def boolean(self, query, operation):
        result = 0b11111111111111111111111 if operation == "and" else 0b00000000000000000000000   
        
        for q in query.split(" "):
            line = numpy.transpose(self._freqs)[self._terms.index(q)]
            strBitL = ''.join('1' if x != 0 else '0' for x in line)
            result =  (int(strBitL, 2) & result) if operation == "and" else (int(strBitL, 2) | result)

        return self.find_indexes("{0:b}".format(result), '1')
    
    def bm_25(self, query): 
        sim_bm25 = []
        
        # numero de docs na colecao
        N = len(self._docs)
        # pra cada doc
        for doc in self._docs:
            # auxiliar que vai acumular o somatorio
            sum = 0        
            # pra cada termo
            for freq in self._frequency(query):                
                beta = self._betha(freq, doc)    
                print(self.weight_matrix())       
                break
                # ni = sum([1 for i in self._docs[word].values() if i > 0])
                # nj = 0
                # for k in range(N):
                #     valor = m_inc_cons.iloc[j,k]
                #     if valor > 0 :
                #     nj = nj + 1
                
                # p_log = (N - nj + 0.5)/(nj + 0.5)
                # som = som + beta * np.log2(p_log)
                
                # sim_bm25.append(som)
            
        return sim_bm25


def print_vector_result(query, documents, corpus):
    q = corpus.query_weight(query)
    m = corpus.weight_matrix()
    numpy.set_printoptions(precision=2, linewidth=90)
    print('query = ', repr(query))
    print(' w`=', numpy.array(q), end='\n\n')
    results = {}
    for i, doc in enumerate(documents):
        rank = corpus.vector_similarity(m[i], q)
        print('d%d =' % i, repr(doc))
        print(' w =', numpy.array(m[i]), end='\n')
        print(' rank =', rank, end='\n\n')

        if(rank > 0):
            results["d"+str(i)] = rank
        

    results = sorted(results.items(), key=operator.itemgetter(1))
    print("Documentos/Ranks", results)

def main():
    corpus = Corpus(separators, stopwords)
    for doc in documents:
        corpus.add_document(doc)


    
    and_boolean_result = corpus.boolean(query, "and")
    or_boolean_result = corpus.boolean(query, "or")
    # print(and_boolean_result)
    # print(or_boolean_result)
    
    # print_vector_result(query, documents, corpus)


    # corpus.bm_25(query)

    # pr = PageRank()
    # pr.rank()

    hits = HITS()
    hits.rank()



if __name__ == '__main__':
    main()