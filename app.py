from urllib import request
from flask import Flask,render_template,url_for,jsonify,request
# from flask_wtf import FlaskForm
# from wtforms import TextAreaField,SubmitField
import pandas as pd
import re
import time
app=Flask(__name__)
#Spell Checker's code
import time
import enchant
from transformers import AutoTokenizer, AutoModelForMaskedLM,pipeline,BertForMaskedLM,BertTokenizer

tokenizer = AutoTokenizer.from_pretrained("Rajan/NepaliBERT")
vocab_file_dir = './NepaliBERT/' 
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir,
                                        strip_accents=False,
                                         clean_text=False )
model = BertForMaskedLM.from_pretrained('./NepaliBERT')
fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)
import threading
corpus = []
threads=[]
text="म भात खान्छु । हामी भात खान्छौं । तँ भात खान्छस् । तपाईं भात खानुहुन्छ । म विध्यालय जान्छु । हामी विध्यालय जान्छौँ । म घुम्न जान्छु । हामी घुम्न जान्छौँ । राम भात खान्छ । तिमी पढ्न जाउ । स्पष्ट नेपालीले काम गर्छ । "
list1=text.split("।")
list2=[]
for _ in range(len(list1)-1):
  temp2=list1[_].split()
  # print(temp2)
  temp_tup=(temp2[0],temp2[-1][-1])
  list2.append(temp_tup)

def check_rule(str1):
  temp_list1=str1.split("।")
  temp_list2=[]
  err=[]
  for _ in range(len(temp_list1)-1):
    temp2=temp_list1[_].split()
    temp_tup=(temp2[0],temp2[-1][-1])
    temp_list2.append(temp_tup)
  for _ in temp_list2:
    # print(_)
    if _ in list2:
      continue
    else:
      err.append(_)
  return err

def tri_return_tuples(list1,x=0):
  temp_list={}
  pp=0
  for _ in range(0,len(list1)-2,1):
    if pp!=0:
      pp-=1
      continue
    temp21=(list1[_].split("\t")[x],list1[_+1].split("\t")[x],list1[_+2].split("\t")[x])
    if temp21[1]=="YF" or temp21[1].strip()=="।":# or temp21[1]=="YF" or temp21[1]=="।":
      pp=1
      continue
      temp21=(list1[_+1].split("\t")[x],list1[_+2].split("\t")[x],list1[_+3].split("\t")[x])
      # print(temp21[0])
      pp=1
    temp2=temp21
    if temp2 not in temp_list:
      temp_list[temp2]=1
    else:
      temp_list[temp2]+=1      
    # temp_list.append(temp2)
  return temp_list
@app.before_first_request

#rulee base approach


##tri gram



##

def loadCorpus():
    #function to load the dictionary/corpus and store it in a global list
    global corpus
    # path12='D:\Spasta Nepali data+files\dictionary.txt'
    path12='D:/Spasta Nepali data+files/project on/added_s/unique_words.txt'
    with open(path12,'r', encoding="utf-8") as csv_file:
        corpus =csv_file.readlines()#csv.reader(csv_file)
        # for line in corpus:
        #     corpus.append(line[1])
    # corpus=corpus[:160000]
    # return corpus
           
loadCorpus()       
def getLevenshteinDistance(s, t):

    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
                
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution

    return dist[row][col]

def getCorrectWord(word):
    min_dis=100
    correct_word=""
    for s in corpus:
        cur_dis = getLevenshteinDistance(s,word)
        if min_dis > cur_dis :
            min_dis = cur_dis
            correct_word = s
    return correct_word
def processInput(arg1):
    inputtext=arg1
    print("processing spelling")
    words=inputtext.strip().split()
    output=''     
    for word in words:            
      if word in corpus :
        output = output + word + ' '
          
      else:
        corrected= getCorrectWord(word)
        output = output + corrected + ' '          
    return(output)

#Spell Cheker ends


#Grammar Checker Starts

# from nepali_stemmer.stemmer import NepStemmer
# nepstem = NepStemmer()
start=time.time()
x=["cs.txt","gc_books.txt","gc_newspaper__extract.txt","gc_webtext__extract.txt"]
def reading(filename):
  with open(filename,"r",encoding="utf-8") as f:
    x=f.read().split("\n")
    return x


temp_path="D:/Spasta Nepali data+files/project on/added_s/"
temp0=reading(temp_path+x[0])
temp1=reading(temp_path+x[1])
temp1=temp1[:len(temp1)-3]
temp2=reading(temp_path+x[2])
temp2=temp2[:len(temp2)-3]
temp3=reading(temp_path+x[3])
temp3=temp3[:len(temp3)-3]
all_files_list=temp0+temp1+temp2+temp3
del temp1
del temp0
del temp2
del temp3
# print(len(all_files_list))
def return_tuples(list1):
  temp_list=[]
  # print(len(list1))
  for ii in list1:
    temp1=ii.split("\t")
    key1=temp1[0].strip()
    # print(temp1)
    key2=temp1[1].split("\n")[0].strip()
    # print(key2)
    temp2=(key1,key2)
    # print(type(temp2),temp2,temp1[0].strip())
    temp_list.append(temp2)
  return temp_list

get_tuples=return_tuples(all_files_list)

def get_frequency(tup1):

    freqs = {} # dictionary to be returned
    curr_words = set()
    ### START CODE HERE ###
    for tup in tup1:
        word, label = tup
        if word not in curr_words:
            new_dict = {}
            new_dict[label] = 1
            freqs[word] = new_dict
        else:
            cur_dict = freqs[word]
            if label in set(cur_dict.keys()):
                cur_dict[label] += 1
            else:
                cur_dict[label] = 1
            freqs[word] = cur_dict
        curr_words.add(word)
    ### END CODE HERE ###
    return freqs
freqs=get_frequency(get_tuples)
get_all_tags=reading(temp_path+"/111_tags.txt")
all_tags=[]
for _ in get_all_tags:
  temp=_.split("\t")[0].strip()
  if temp=='*' or temp=='':
    continue
  all_tags.append(temp)

del get_all_tags

import numpy as np
def transition_count(tup2):
  transition={}
  for _ in range(len(tup2)-1):
    key1=tup2[_][1]
    key2=tup2[_+1][1]
    key3=(key1,key2)
    # print(key3)
    temp_dict1={}
    if key3 not in transition.keys():
      transition[key3]=1
    else:
      transition[key3]+=1
  return transition

temp1_a=np.zeros((111,111))
temp_A=pd.DataFrame(temp1_a,index=all_tags, columns =all_tags)


get_transition_count=transition_count(get_tuples)
ttt=0
def create_transition_matrix(constant_val,tags_with_count,count_transition):
  global temp_A
  all_tags=list(tags_with_count.keys())#sorted
  A=np.zeros((len(all_tags),len(all_tags)))
  tuple1=set(count_transition.keys())
  for i in range(len(all_tags)):
    for j in range(len(all_tags)):
      count=0
      # if(all_tags[i]=="*"):
      #   continue
      temp_tuple1=(all_tags[i],all_tags[j])
      if temp_tuple1 in count_transition.keys():
        count=count_transition[temp_tuple1]
      count_prev_tag=tags_with_count[all_tags[i]]
      temp_calc=(count + constant_val) / (count_prev_tag + constant_val * len(all_tags))
    #   A[i,j] = temp_calc
      temp_A.loc[all_tags[i],all_tags[j]]=temp_calc

#   return A


def emission_count(tup2):
  emission={}
  set1=set()
  for _ in tup2:
    temp1=_
    key1=(temp1[0].strip(),temp1[1].strip())    
    if key1 not in emission.keys():
      emission[key1]=1
    else:
      emission[key1]+=1
  return emission

tag_with_count={}
ij=0
# print(len(all_files_list))
for i in all_files_list:
  ij+=1
  a=i.split("\t")[1].split("\n")[0]
  if a not in tag_with_count:
    if a=="*" or a=="":
      continue
    tag_with_count[a]=1
  else:
    tag_with_count[a]+=1

emission_counts=emission_count(get_tuples)
create_transition_matrix(0,tag_with_count,get_transition_count)
def create_emission_matrix(constant_val,tags_count,emission_count_data,vocab):
  all_tags=list(tags_count.keys())
  # print(all_tags[:20])
  B=np.zeros((len(all_tags),len(emission_count_data)))
  # B_temp=pd.DataFrame(B,index=all_tags,columns=)
  for i in range(len(all_tags)):
    for j in range(len(emission_count_data)):
      count=0
      tuple1=(vocab[j],all_tags[i])
      if tuple1 in emission_count_data.keys():
        count=emission_count_data[tuple1]
      count_tag=tags_count[all_tags[i]]

      B[i,j] = (count +constant_val) / (count_tag +constant_val * len(vocab))
  return B

only_text=[]
get_only_text=reading(temp_path+"/only_all_text.txt")
B=create_emission_matrix(0.00,tag_with_count,emission_counts,get_only_text)
words_name=[word for word,tag in emission_counts.keys()]
B_sub = pd.DataFrame(B,index=list(tag_with_count.keys()), columns =words_name)#pd.DataFrame(A, index=all_tags, columns = all_tags )

def Viterbi(sentence_list,state):
  global temp_A
  global B_sub
  global freqs
  p=[]
  for words in sentence_list:
    # print(len(freqs[words]))
    for j in freqs[words].keys():
      # print(j,words,freqs[words])
      transition_p=temp_A.loc[state[-1],j]
      emission_p=B_sub.loc[j,words][0]
      state_p=transition_p*emission_p
      # print(emission_p)
      p.append(state_p)
    # print(max(p))
  return list(freqs[words].keys())[p.index(max(p))]

def sentence_checker(sents):
  temp_pos=[]
  for x in sents:
    # temp_pos.append(list(x[1].keys())[0])
    temp_pos.append(x[1])
  return temp_pos

def stemmer(combined_words):
  global temp_path
  path2=temp_path+"/suffix.txt"
  with open(path2,"r",encoding="utf-8") as f:
    suffixes=f.readlines()
  modified_suffixes=[y.split("|")[0] for y in suffixes]
  sep_words=[]
  for x in modified_suffixes:
    if x in combined_words:
      temp_list=[combined_words.split(x)[0],x]
      if temp_list[0] in freqs.keys():
        sep_words.append(temp_list)
        break
  return(sep_words)

def check_unique_POS(token1,indx=0):
    # print(type(token1))
    # print(len(freqs[token1]))
    if token1 in freqs.keys() :#and len(freqs[token1])==1:
        # print(freqs[token1],len(freqs[token1])," : 1")
        key_list=list(freqs[token1].keys())
        value_list=list(freqs[token1].values())
        # print(key_list,value_list,max(value_list))
        temp_tuple=(token1,key_list[value_list.index(max(value_list))])
        # temp_tuple=(token1,freqs[token1])
    else:
        temp_list=stemmer(token1)
        # print(len(freqs[token1]))
        return temp_list
        # temp_tuple=(token1,freqs[token1])
    return temp_tuple

def pos_tag(texts):
    texts=texts.replace("|","।")
    sents=texts.split("।")
    print(f"There are {len(sents)-1} sentences in the provided texts")
    words=[sents[y].split() for y in range(len(sents)-1)]
    count=0
    tagging=[]
    state=[]
    for y in words:
        temp_tagging=[]
        for each_word in y:
            if each_word in freqs.keys():
                # print(freqs[each_word])
                # if 
                key_list=list(freqs[each_word].keys())
                value_list=list(freqs[each_word].values())
                # print(key_list,value_list,max(value_list))
                temp_tuple=(each_word,key_list[value_list.index(max(value_list))])
                temp_tagging.append(temp_tuple)
                # print(temp_tuple)
                # print("temp tuple 1st  ",temp_tuple,type(temp_tuple[1]))#yo 11th ma milako
                count+=1
            else:
                import nltk
                
                new_words=nltk.tokenize.word_tokenize(each_word)
                # print(new_words)
                for new_tokens in new_words:
                    temp_tuple=check_unique_POS(new_tokens)
                    if type(temp_tuple)!=list:
                      temp_tagging.append(temp_tuple)
                      # print(len(freqs[new_tokens])," :  second")
                    else:
                      # print(temp_tuple)
                      for xy in temp_tuple:
                        #  print("asdsad ",xy,type(xy))
                         for p in xy:
                           temp_tuple=check_unique_POS(p)
                          #  print("temp tuple 2nd ",temp_tuple)
                           count+=1
                           if type(temp_tuple)!=list:
                              temp_tagging.append(temp_tuple)
                              # tagging.append(temp_tagging)
                         

        tagging.append(temp_tagging)
        # print(count,tagging)
    # only_tags=[]
    # for _ in tagging:
    #   for aa in _:
    #     temp22=Viterbi(_,state)
    #     state.append(temp22)

    return tagging


def remove_pun(my_str):
  punctuations = '''!()[]\{\};:'"\,<>./?@#$%^&*_~'''
  no_punct = ""
  for char in my_str:
    if char not in punctuations:
        no_punct = no_punct + char
  return (no_punct)

def return_probab(pos_list):
  global temp_A
  probab=[]
  dict_probab={}
  # print(type(pos_list))
  for temp_pos_list in pos_list:
    temp_pos_list.insert(0,"--s--")
    temp_pos_list.append("YF")
    # print("temp_pos ",temp_pos_list)
    # temp_probab=A_sub.loc[]
  
  for y in pos_list:
    for x in range(len(y)-1):
      key1=(y[x],y[x+1])
      # print(freqs[key1])
      temp_calc=temp_A.loc[y[x],y[x+1]]#A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]
      # print(y[x],",",y[x+1],temp_calc,"   ",A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]," ",A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]==temp_calc)
      dict_probab[key1]=temp_calc
      probab.append(temp_calc)
  # print(dict_probab)#yo 11th ma milako
  return dict_probab
  # print(max(probab))

  # print(pos_list)



def sec(sents_lists):
  opening=["'","\"","[","(","{"]
  closing=["'","\"","]",")","}"]
  check=[]
  for i in sents_lists:
    for each_word,tags in i:
      if each_word in opening or each_word in closing:
        check.append(each_word)
  for i in range(len(check)):
    for j in range(i+1,len(check)):
      if check[i] in opening and check[j] in closing :
        if opening.index(str(check[i]))==closing.index(str(check[j])):  
          check[i]="*"
          check[j]="*"
          break
  return(check)


del temp1_a
del get_only_text
del words_name
##rule
# def tri_return_tuples(list1,x=0):
#   temp_list={}
#   pp=0
#   for _ in range(0,len(list1)-2,1):
#     # if pp==1:
#     #   pp=0
#     #   continue
#     temp21=(list1[_].split("\t")[x],list1[_+1].split("\t")[x],list1[_+2].split("\t")[x])
#     if temp21[0]=="YF" or temp21[0]=="।" or temp21[1]=="YF" or temp21[1]=="।":
#       continue
#       temp21=(list1[_+1].split("\t")[x],list1[_+2].split("\t")[x],list1[_+3].split("\t")[x])
#       # print(temp21[0])
#       pp=1
#     temp2=temp21
#     if temp2 not in temp_list:
#       temp_list[temp2]=1
#     else:
#       temp_list[temp2]+=1      
#     # temp_list.append(temp2)
#   return temp_list
get_tuples1=tri_return_tuples(all_files_list)
get_tuples2=tri_return_tuples(all_files_list,1)
def check_trigrams(sents1):
  global get_tuples1
  # print(sents1)#yo 11th ma milako
  temp12=sents1.split()
  temp_dict=[]
  for _ in range(0,len(temp12)-2):
    key=(temp12[_],temp12[_+1],temp12[_+2])
    if key in get_tuples1:
      pass
      # print(key,get_tuples1[key])#yo 11th ma milako
    else:
      temp_dict.append(key)
      # print("Not found: ",key)#yo 11th ma milako
  return temp_dict

##masking
sum12=[int(a) for a in get_tuples2.values()]
sum11=0
for _ in sum12:
  sum11+=_
def check_tri(temp_list):
  temp_list.insert(0,("--s--","--s--"))
  temp_list.append(("।","YF"))
  temp_list2=[]
  global sum11
  d={}
  k={}
  for _ in range(len(temp_list)-2):
    key1=(temp_list[_][1],temp_list[_+1][1],temp_list[_+2][1])
    key2=(temp_list[_][0],temp_list[_+1][0],temp_list[_+2][0])
    key3=(temp_list[_][0]+"\t"+temp_list[_][1],temp_list[_+1][0]+"\t"+temp_list[_+1][1],temp_list[_+2][0]+"\t"+temp_list[_+2][1])
    
    # print(key1)
    if key1 in get_tuples2.keys():
      d[key1]=get_tuples2[key1]    
    else:
      d[key1]=0
    if key2 in get_tuples1.keys():      
      k[key3]=get_tuples1[key2]
    else:
      k[key3]=0
  return d,k

def return_verb_mask(tagging):
  dict_temp={}
  count=-1
  count2=-1
  for _ in tagging:
    count+=1   
    mask=""
    text23=""
    line_no=0
    count2=0
    for _1 in _:
      count2+=1
      if _1[1][:2]!="VV":
        text23+=_1[0]+" "
      else:
        text23+="[MASK] "
        mask=_1[0]
    text23+="।"
    # print(text23)
    rrr=fill_mask(text23)
    temp111=[ab["token_str"] for ab in rrr]  #removed strip
      # return mask,line_no
    dict_temp[(count,count2,mask.strip())]=temp111       
    # print(temp111,mask.strip() in temp111)#nov 11
    if (mask.strip() in temp111)==True:
      dict_temp[(count,count2,mask.strip())]=[mask]
  return dict_temp


##
del all_files_list
# del get_all_tags
stop1=time.time()
print(start,stop1," Time elapsed is : ",stop1-start)
all_variables = dir()


###Yo chai 14th Feb ma add garya

def return_probab(pos_list):
  global temp_A
  probab=[]
  # dict_probab={}
  # print(type(pos_list))
  for temp_pos_list in pos_list:
    temp_pos_list.insert(0,"--s--")
    temp_pos_list.append("YF")
    # dict_probab={}
    # print("temp_pos ",temp_pos_list)
    # temp_probab=A_sub.loc[]
  
  for y in pos_list:
    dict_probab={}
    for x in range(len(y)-1):
      key1=(y[x],y[x+1])
      # print(freqs[key1])
      temp_calc=temp_A.loc[y[x],y[x+1]]#A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]
      # print(y[x],",",y[x+1],temp_calc,"   ",A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]," ",A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]==temp_calc)
      dict_probab[key1]=temp_calc
    # print(temp_calc)
    probab.append(dict_probab)#one indent done plz undo
  # print(dict_probab)
  return probab

def check_prob(text21):
  # text21="उसले मलाई झेले को छ |नेपालको निकासी व्यापार मा ऊनी गलैंचाले चालू आर्थिक वर्षमा पनि पहिलो स्थान ओगटेको छ |"#"नेपालको निकासी व्यापार मा ऊनी गलैंचाले चालू आर्थिक वर्षमा पनि पहिलो स्थान ओगटेको छ |  "#"जन रातो स्याउ जान्छ|"##"नेपालको निकासी व्यापारमा ऊनी गलैंचाले चालू आर्थिक वर्षमा पनि पहिलो स्थान ओगटेको छ ।"
  tagging=pos_tag(text21)
  get_pos=[]
  print("Tagging",tagging)
  for i in tagging:  
    get_pos.append(sentence_checker(i))
  print(get_pos,type(get_pos))
  dict2=return_probab(get_pos)
  print(dict2)
  for dict3 in dict2:
    pr=1
    count=10**(len(dict3)-2)
    most_probable_error=[]
    count=0
    for _ in dict3.values():
      if _<=0.01:
        most_probable_error.append(count)
        print(_)
      count+=1
    print(most_probable_error)
    for i in dict3.values():
      # count*=10
      pr*=i
    print(pr,pr*count)
    if pr<1.3398807621274485e-18:
      print("incorrect")
    else:
      print("correct")

####yaha samma

@app.route("/")
def index():        
    return render_template("ajax.html",)

@app.route("/spell_check",methods=["POST","GET"])
def spell_check():
  if request.method=="POST":
    data=request.get_json(force=True)
    input_movie_name1 = remove_pun(data['page_data'])
    if(len(input_movie_name1)==0):
      return jsonify("<span></span>")
    print("Spell checking has started: ",input_movie_name1)
    input_movie_name = input_movie_name1.split()

    #load the personal word list dictionary
    start=time.time()

    #load the personal word list dictionary
    movies_dict = enchant.PyPWL("D:/Spasta Nepali data+files/flask1/movies.txt")
    temp_list=[]
    index=-1
    html1=""
    #check if the word exists in the dictionary
    for _ in input_movie_name:
      index+=1
      word_exists = movies_dict.check(_)
      
      # print("word exists: ", word_exists," ",_ )#yo 11th ma milako


      if not word_exists:
        temp_dict={}
        #get suggestions for the input word if the word doesn't exist in the dictionary
        suggestions = movies_dict.suggest(_)
        temp_dict["index"]=index 
        temp_dict["word"]=_      
        temp_dict["suggestion"]=suggestions
        temp_list.append(temp_dict)
        temp_text=",".join(suggestions)
        html1+=f"""<span class="incorrect_word" id=\"{_}\" onclick='correction(\"{temp_text}\",\"{_}\",\"{index}\");'>{_}</span> 
        """#document.getElementById(\"mistakes\").innerHTML=\"{temp_text}\";
      else:
        html1+=f"""<span class='correct_word'>{_}</span>
        """
    # print(html1)
    stop=time.time()
    print("Spelling checking time",stop-start," Spell checking stopped")
    return jsonify(html1)



@app.route("/test_ajax1",methods=["POST","GET"])
def testing():
    global temp_A
    if request.method=="POST":
        start1=time.time()
        data2=request.get_json(force=True)
        # print(data2['page_data'])#yo 11th ma milako
        text21=data2['page_data']
        stem1=""
        
        temp_list24=text21.split("।")
        # for _ in text21.split():#yo 11th ma milako
        #   temp33=stemmer(_)#yo 11th ma milako
          # stem1+=temp33
          # print(temp33)#yo 11th ma milako
        # print(stem1)#yo 11th ma milako
        tagging=pos_tag(text21)
        ##14th Feb
        probab1=check_prob(text21)
        print(probab1)
        ##14th Feb
        get_pos=[]
        print("Tagging",tagging)#yo 11th ma milako
        for i in tagging:  
            get_pos.append(sentence_checker(i))
        # print(get_pos,type(get_pos))
        dict2=return_probab(get_pos)
        print("Masking started")
        verb1=return_verb_mask(tagging)#tagging thyo
        print(verb1)
        print(temp_list24)
        temp33=""
        for _ in verb1.keys():#10th nov
          print(type(verb1[_]),[verb1[_]],_)#nov 11,tagging[_])#10th
          # temp_list24[_[0]]=temp_list24[_[0]].replace(_[2],)
          # if verb1[_]==list:
          temp_text=",".join(verb1[_])
            # temp_text
          print("Verb is",verb1[_],"Temp text is :",temp_text," ",_[2])
          temp33+=f"""<span class="incorrect_word" id=\"{_[2]}\" onmouseup='correction(\"{temp_text}\",\"{_[2]}\",\"{index}\");'>{_[2]}</span> 
        """
          # print(verb1[_],tagging[_[0]][_[1]-1])#10th nov
        print("Masking ended")
        check_symbol=sec(tagging)
        # print(check_symbol)#yo 11th ma milako
        # print(dict2)#yo 11th ma milako
        # probab=""
        # for _ in list(dict2.values()):
        #   probab+=str(_)+" , "
        # print(check_symbol)#yo 11th ma milako
        text22=""
        c1=0
        temp_list2=text21.split()
        # for _ in range(0,len(temp_list2)):  
        #   # t1 = threading.Thread(target=processInput, args=(temp_list2[_],))
        #   # t2 = threading.Thread(target=processInput, args=(temp_list2[_+1],))        
        #   text22+=processInput(temp_list2[_])
        # print(text22)
        stop1=time.time()
        text22=check_rule(text21)
        # print(text22)#yo 11th ma milako
        temp_list22=text21.split()
        if(len(temp_list22)>0):
          temp44=tri_return_tuples(temp_list22)
          # print("Tri ",temp44)#yo 11th ma milako
        for _ in list2:
            # print(_[1])
            if len(text22)>0:
              # print(text22[0][0],_[0])
              if text22[0][0]==_[0]:
                temp33+=f"कृपया '{text22[0][0]}' कृयापदमा '{str( _[1])}' प्रयोग गर्नुहोस ।"
                break
        print(f"Index page load time {stop1-start1}")
        list_temp=text21.split()
        # for _ in range(len(list_temp)):
        #   pass
        # print(fill_mask(f"{text21} [MASK]"))
        # check_trigrams(text21) ##it's futuristic be ready for that
        return jsonify(temp33) #"त्रुटी  "+text22[0][1]+



if __name__=="main":
    app.run(debug=True)